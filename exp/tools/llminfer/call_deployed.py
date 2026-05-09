import traceback, re, subprocess, asyncio, os, signal, itertools
from typing import Literal, List, Dict, Any
from openai import OpenAI, AsyncOpenAI
from exp.tools import async_tqdm, async_sleep, async_gather
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"expandable_segments:True"
from exp.tools.llminfer.semaphore import get_semaphore_cls, get_max_concurrent, GLOBAL_ASYNC_SEMAPHORES, close_redis_semaphore

THINK_PATTERN = re.compile(r'<think>.*?</think>\s*', re.DOTALL)

GLOBAL_CLIENTS: "dict[str, OpenAI]" = dict() 
GLOBAL_ASYNC_CLIENTS: "dict[str, AsyncOpenAI]" = dict() 
GLOBAL_ASYNC_WORKERS: "dict[tuple[str, int, str], MultiGPUVLLMManager]" = dict()
DEPLOY_SEMAPHORE = asyncio.Semaphore(1)

def dict_to_args(params: Dict[str, Any]) -> List[str]:
    """
    Universal utility to convert a Python dictionary into a CLI argument list.
    - Converts underscores to hyphens (max_model_len -> --max-model-len)
    - Handles Boolean flags (True -> --flag, False/None -> Skip)
    - Handles valued arguments (port: 8000 -> --port 8000)
    """
    arg_list = []
    for key, value in params.items():
        # Skip internal or null values
        if value is None or value is False:
            continue

        flag = "--" + key.replace("_", "-")

        if value is True:
            # It's a standalone flag
            arg_list.append(flag)
        else:
            # It's a key-value pair
            arg_list.extend([flag, str(value)])

    return arg_list

def role_template(content:str, role: Literal["system", "user", "assistant"], system: "bool" = None):
    if system is not None:
        assert role != "system"
    system_template = [] if system is None else [dict(role = 'system', content = system)]
    return system_template + [dict(role = role, content = content)]



def get_deployed(model_name: str = None, port: int = None, api_key = 'None'):
    model_name = model_name.rstrip("/")
    global GLOBAL_CLIENTS
    if (model_name, port, api_key) not in GLOBAL_CLIENTS:
        GLOBAL_CLIENTS[(model_name, port, api_key)] = OpenAI(
            api_key = api_key, base_url = f"http://127.0.0.1:{port}/v1"
        )
    return GLOBAL_CLIENTS[(model_name, port, api_key)]



def query_deployed(user_query: str = None, messages: list = None, 
                   generate_kwargs: dict = dict(temperature = 0.7,top_p = 0.95, max_tokens = 100), max_tokens: int = None, enable_thinking: bool = False, discard_thinking: bool = False, tools = None, tool_choice = None, port:int = None, model_name: str = None, api_key = 'None', return_text: bool = True, return_dict: bool = False):
    model_name = model_name.rstrip("/")
    global GLOBAL_CLIENTS,THINK_PATTERN
    if (model_name, port, api_key) not in GLOBAL_CLIENTS:
        GLOBAL_CLIENTS[(model_name, port, api_key)] = OpenAI(
            api_key = api_key, base_url = f"http://127.0.0.1:{port}/v1"
        )
    client = GLOBAL_CLIENTS[(model_name, port, api_key)]

    assert (user_query is None) ^ (messages is None)
    if user_query is not None:
        assert isinstance(user_query, str)
        messages = [{"role": "user", "content": user_query}]
    if max_tokens is not None:
        generate_kwargs.update(dict(max_tokens = max_tokens))
    
    extras = dict() if enable_thinking else dict(extra_body = {
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        })
    tool_kwargs = dict()
    if tools is not None:
        tool_kwargs['tools']  = tools
    if tool_choice is not None:
        tool_kwargs['tool_choice'] = tool_choice

    response = client.chat.completions.create(
        model = model_name,
        messages = messages,
        **tool_kwargs,
        **generate_kwargs,
        **extras,
    )

    if return_text: 
        content = response.choices[0].message.content
        if enable_thinking and discard_thinking:
            content = THINK_PATTERN.sub('', content)
        return content
    elif return_dict:
        content = response.choices[0].message.content
        if enable_thinking and discard_thinking:
            content = THINK_PATTERN.sub('', content)
        ret = dict(content = content)
        ret['tool_calls'] = response.choices[0].message.tool_calls

        return ret
        
    return response


        

def close_workers():
    global GLOBAL_ASYNC_WORKERS
    for v in GLOBAL_ASYNC_WORKERS.values():
        v.close()
    GLOBAL_ASYNC_WORKERS = dict()



@async_tqdm
async def async_query(user_query: str = None, messages: list = None, 
                   generate_kwargs: dict = dict(temperature = 0.7,top_p = 0.95, max_tokens = 100),max_tokens: int = None, enable_thinking: bool = False, discard_thinking: bool = False, tools = None, tool_choice = None, port:int = None, model_name: str = None, traffic_port: int = 6379, api_key = 'None', return_text: bool = True, return_dict: bool = True, input_text: bool = False, logdir = "./async_logs",
                   gpu_list: list[int] = list(range(8)),
                   tp_size: int = 1, 
                   max_model_len: int = 40960,
                   max_num_seqs: int = 128,
                   gpu_memory_utilization = 0.98):
    

    model_name = model_name.rstrip("/")
    global GLOBAL_ASYNC_WORKERS, GLOBAL_ASYNC_SEMAPHORES, THINK_PATTERN
    if (model_name, port, api_key) not in GLOBAL_ASYNC_WORKERS:
        async with DEPLOY_SEMAPHORE:
            if (model_name, port, api_key) not in GLOBAL_ASYNC_WORKERS:
                GLOBAL_ASYNC_WORKERS[(model_name, port, api_key)] = await MultiGPUVLLMManager(
                    model_name ,gpu_list, tp_size, port, logdir, 
                    max_model_len = max_model_len,
                    max_num_seqs = max_num_seqs,
                    gpu_memory_utilization = gpu_memory_utilization
                ) .deploy()
        
    if (model_name, port, api_key) not in GLOBAL_ASYNC_SEMAPHORES:
        sema = get_semaphore_cls('asyncio')(
            redis_url=f"redis://localhost:{traffic_port}",
            key=f"{model_name}:{port} - {api_key}",
            max_concurrency = get_max_concurrent()
        )
        await sema.connect()
        GLOBAL_ASYNC_SEMAPHORES[(model_name, port, api_key)] = sema

    semaphore = GLOBAL_ASYNC_SEMAPHORES[(model_name, port, api_key)]

    client = GLOBAL_ASYNC_WORKERS[(model_name, port, api_key)].get_worker()


    assert (user_query is None) ^ (messages is None)
    if user_query is not None and (not input_text):
        assert isinstance(user_query, str)
        messages = [{"role": "user", "content": user_query}]
    if max_tokens is not None:
        generate_kwargs.update(dict(max_tokens = max_tokens))
    
    extras = dict() if enable_thinking else dict(extra_body = {
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    })


    tool_kwargs = dict()
    if tools is not None:
        tool_kwargs['tools']  = tools
    if tool_choice is not None:
        tool_kwargs['tool_choice'] = tool_choice

    exception = None
    retry_times = 1
    timeout = 0.5
    while retry_times > 0:
        try:
            await semaphore.acquire()
            try:
                if input_text:
                    response = await client.chat.completions.create(
                        model = model_name, timeout = None,
                        prompt = user_query,
                        **tool_kwargs,
                        **generate_kwargs,
                        **extras
                    )
                else:
                    response = await client.chat.completions.create(
                        model = model_name, timeout = None,
                        messages = messages,
                        **tool_kwargs,
                        **generate_kwargs,
                        **extras
                    )
            finally:
                await semaphore.release()
            break

        except Exception as e:
            exception = e
            retry_times -= 1

            if not retry_times:
                response = EmptyCompletion("", error = str(e) + ":" + str(e))

            await async_sleep(timeout)


    if return_text: 
        content = response.choices[0].message.content
        if enable_thinking and discard_thinking:
            content = THINK_PATTERN.sub('', content)
        return content
    elif return_dict:
        content = response.choices[0].message.content
        if enable_thinking and discard_thinking:
            content = THINK_PATTERN.sub('', content)
        ret = dict(content = content)
        ret['tool_calls'] = response.choices[0].message.tool_calls

        return ret

    return response




@async_tqdm
async def async_query_deployed(user_query: str = None, messages: list = None, 
                   generate_kwargs: dict = dict(temperature = 0.7,top_p = 0.95, max_tokens = 100),max_tokens: int = None, enable_thinking: bool = False, discard_thinking: bool = False, tools = None, tool_choice = None, port:int = None, model_name: str = None, traffic_port: int = 6379, api_key = 'None', return_text: bool = True, return_dict: bool = True, input_text: bool = False, semaphore_type: "Literal['asyncio', 'redis']" = 'asyncio'):
    model_name = model_name.rstrip("/")
    global GLOBAL_ASYNC_CLIENTS, GLOBAL_ASYNC_SEMAPHORES, THINK_PATTERN
    if (model_name, port, api_key) not in GLOBAL_ASYNC_CLIENTS:
        GLOBAL_ASYNC_CLIENTS[(model_name, port, api_key)] = AsyncOpenAI(
            api_key = api_key, base_url = f"http://127.0.0.1:{port}/v1"
        )
    if (model_name, port, api_key) not in GLOBAL_ASYNC_SEMAPHORES:
        sema = get_semaphore_cls(semaphore_type)(
            redis_url=f"redis://localhost:{traffic_port}",
            key=f"{model_name}:{port} - {api_key}",
            max_concurrency = get_max_concurrent()
        )
        await sema.connect()
        GLOBAL_ASYNC_SEMAPHORES[(model_name, port, api_key)] = sema

    semaphore = GLOBAL_ASYNC_SEMAPHORES[(model_name, port, api_key)]

    client = GLOBAL_ASYNC_CLIENTS[(model_name, port, api_key)]


    assert (user_query is None) ^ (messages is None)
    if user_query is not None and (not input_text):
        assert isinstance(user_query, str)
        messages = [{"role": "user", "content": user_query}]
    if max_tokens is not None:
        generate_kwargs.update(dict(max_tokens = max_tokens))
    
    extras = dict() if enable_thinking else dict(extra_body = {
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    })


    tool_kwargs = dict()
    if tools is not None:
        tool_kwargs['tools']  = tools
    if tool_choice is not None:
        tool_kwargs['tool_choice'] = tool_choice

    exception = None
    retry_times = 1
    timeout = 0.5
    while retry_times > 0:
        try:
            await semaphore.acquire()
            try:
                if input_text:
                    response = await client.chat.completions.create(
                        model = model_name, timeout = None,
                        prompt = user_query,
                        **tool_kwargs,
                        **generate_kwargs,
                        **extras
                    )
                else:
                    response = await client.chat.completions.create(
                        model = model_name, timeout = None,
                        messages = messages,
                        **tool_kwargs,
                        **generate_kwargs,
                        **extras
                    )
            finally:
                await semaphore.release()
            break

        except Exception as e:
            exception = e
            retry_times -= 1

            if not retry_times:
                response = EmptyCompletion("", error = str(e) + ":" + str(e))

            await async_sleep(timeout)


    # await semaphore.release()

    # if isinstance(response, EmptyCompletion):
    #     print(exception, end = " ")

    if return_text: 
        content = response.choices[0].message.content
        if enable_thinking and discard_thinking:
            content = THINK_PATTERN.sub('', content)
        return content
    elif return_dict:
        content = response.choices[0].message.content
        if enable_thinking and discard_thinking:
            content = THINK_PATTERN.sub('', content)
        ret = dict(content = content)
        ret['tool_calls'] = response.choices[0].message.tool_calls

        return ret

    return response



class Msg: ...
class EmptyCompletion:
    def __init__(self, text, error):
        self.choices = [Msg()]
        self.choices[0].message = Msg()
        self.choices[0].message.content = text
        self.choices[0].message.tool_calls = []
        self.error = error


def async_save_exit(func):
    '''
    decorate the main function of the event loop
    '''
    async def wrapper(*args, **kwargs):
        try: return await func(*args, **kwargs)
        finally: 
            await close_redis_semaphore()
            close_workers()
    return wrapper
