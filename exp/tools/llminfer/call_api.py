import asyncio, re, time, httpcore
from openai import OpenAI, AsyncOpenAI, RateLimitError, BadRequestError
from exp.tools import async_tqdm
from exp.tools.utils.tools import *
from exp.tools.llminfer.semaphore import get_semaphore_cls, get_max_concurrent, GLOBAL_ASYNC_SEMAPHORES, close_redis_semaphore
THINK_PATTERN = re.compile(r'<think>.*?</think>\s*', re.DOTALL)

GLOBAL_CLIENTS: "dict[str, OpenAI]" = dict() 
GLOBAL_ASYNC_CLIENTS: "dict[str, AsyncOpenAI]" = dict() 

def query_api(user_query: str = None, messages: list = None, 
              generate_kwargs: dict = dict(temperature = 0.7,top_p = 0.95, max_tokens = 100), max_tokens: int = None, enable_thinking: bool = False, discard_thinking: bool = False, base_url:str = None, model_name: str = None, api_key = 'None', return_text: bool = True):
    model_name = model_name.rstrip("/")
    global GLOBAL_CLIENTS, THINK_PATTERN
    if (model_name, base_url, api_key) not in GLOBAL_CLIENTS:
        GLOBAL_CLIENTS[(model_name, base_url, api_key)] = OpenAI(
            api_key = api_key, base_url = base_url
        )
    client = GLOBAL_CLIENTS[(model_name, base_url, api_key)]

    assert (user_query is None) ^ (messages is None)
    if user_query is not None:
        assert isinstance(user_query, str)
        messages = [{"role": "user", "content": user_query}]

    if max_tokens is not None:
        generate_kwargs.update(dict(max_tokens = max_tokens))

    extras = dict() if enable_thinking else dict(extra_body = {
            "chat_template_kwargs": {
                "enable_thinking": False
            },
            # "thinking": {"type": "disabled"}
        })

    response = client.chat.completions.create(
        model = model_name,
        messages = messages,
        **generate_kwargs,
        **extras,
    )
    # print(response.choices[0].message.content)

    if return_text: 
        content = response.choices[0].message.content
        if enable_thinking and discard_thinking:
            content = THINK_PATTERN.sub('', content)
        return content

    return response

@async_tqdm
async def async_query_api(user_query: str = None, messages: list = None, 
                   generate_kwargs: dict = dict(temperature = 0.7,top_p = 0.95, max_tokens = 100), max_tokens: int = None, enable_thinking: bool = False, discard_thinking: bool = False, base_url:str = None, model_name: str = None, traffic_port: int = 6379, api_key = 'None', return_text: bool = True, semaphore_type: "Literal['asyncio', 'redis']" = 'asyncio'):
    model_name = model_name.rstrip("/")
    global GLOBAL_ASYNC_CLIENTS, GLOBAL_ASYNC_SEMAPHORES,  THINK_PATTERN
    if (model_name, base_url, api_key) not in GLOBAL_ASYNC_CLIENTS:
        GLOBAL_ASYNC_CLIENTS[(model_name, base_url, api_key)] = AsyncOpenAI(
            api_key = api_key, base_url = base_url
        )

    if (model_name, base_url, api_key) not in GLOBAL_ASYNC_SEMAPHORES:
        sema = get_semaphore_cls('asyncio')(
            redis_url=f"redis://localhost:{traffic_port}",
            key=f"{model_name}:{base_url} - {api_key}",
            max_concurrency = get_max_concurrent()
        )
        await sema.connect()
        GLOBAL_ASYNC_SEMAPHORES[(model_name, base_url, api_key)] = sema

    semaphore = GLOBAL_ASYNC_SEMAPHORES[(model_name, base_url, api_key)]

    client = GLOBAL_ASYNC_CLIENTS[(model_name, base_url, api_key)]


    assert (user_query is None) ^ (messages is None)
    if user_query is not None:
        assert isinstance(user_query, str)
        messages = [{"role": "user", "content": user_query}]
    if max_tokens is not None:
        generate_kwargs.update(dict(max_tokens = max_tokens))

    extras = dict() if enable_thinking else dict(extra_body = {
        "chat_template_kwargs": {
            "enable_thinking": False
        },
        "thinking": {"type": "disabled"}
    })

    

    exception = None
    retry_times = 3
    timeout = 0.5
    while retry_times > 0:
        try:
            ratelimit = False
            connecterror = False
            await semaphore.acquire()
            response = await client.chat.completions.create(
                model = model_name, timeout = None,
                messages = messages,
                **generate_kwargs,
                **extras
            )
            retry_times -= 1
        except BadRequestError as e:
            exception = e
            response = EmptyCompletion("", error = str(e) + ":" + str(e))
            # traceback.print_exc()
            ratelimit = False
            connecterror = False
            print(e)
        except RateLimitError as e:
            exception = e
            response = EmptyCompletion("", error = str(e) + ":" + str(e))
            ratelimit = True
            connecterror = False
            print(e)
        except httpcore.ConnectError as e:
            exception = e
            response = EmptyCompletion("", error = str(e) + ":" + str(e))
            connecterror = True
            ratelimit = False
            print(e)
        except Exception as e:
            exception = e
            response = EmptyCompletion("", error = str(e) + ":" + str(e))
            # traceback.print_exc()
            ratelimit = False
            connecterror = False
            # raise e
            print(e)
        finally:
            retry_times -= 1
            await semaphore.release()
            if ratelimit:
                time.sleep(randint(10, get_max_concurrent() * 1000) / get_max_concurrent() / 1000)
            # if connecterror:
            #     time.sleep(randint(20, 50)/100)

    if return_text: 
        content = response.choices[0].message.content
        if enable_thinking and discard_thinking:
            content = THINK_PATTERN.sub('', content)
        return content

    return response



class Msg: ...
class EmptyCompletion:
    def __init__(self, text, error):
        self.choices = [Msg()]
        self.choices[0].message = Msg()
        self.choices[0].message.content = text
        self.error = error
        