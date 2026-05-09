from typing import Literal
import traceback, re
from openai import OpenAI, AsyncOpenAI
from exp.tools import async_tqdm
from exp.tools.llminfer.semaphore import RedisDistributedSemaphore, MAX_CONCURRENT, GLOBAL_ASYNC_SEMAPHORES

THINK_PATTERN = re.compile(r'<think>.*?</think>\s*', re.DOTALL)

GLOBAL_CLIENTS: "dict[str, OpenAI]" = dict() 
GLOBAL_ASYNC_CLIENTS: "dict[str, AsyncOpenAI]" = dict() 


GLOBAL_DEPLOYED = dict()

def query_embedding(text: "str | list[str]" = None, type: Literal["query", "document"] = "document", messages: list = None, port:int = None, model_name: str = None, api_key = 'None', return_text: bool = True, device = None):
    model_name = model_name.rstrip("/")
    if port is None:
        from sentence_transformers import SentenceTransformer
        global GLOBAL_DEPLOYED
        if model_name not in GLOBAL_DEPLOYED:
            GLOBAL_DEPLOYED[model_name] = SentenceTransformer(model_name, device = device)
        model = GLOBAL_DEPLOYED[model_name]
        if type == 'query':
            embeded = model.encode(text, prompt_name="query")
        else:
            embeded = model.encode(text)

        return embeded


    else:
        global GLOBAL_CLIENTS,THINK_PATTERN
        if (model_name, port, api_key) not in GLOBAL_CLIENTS:
            GLOBAL_CLIENTS[(model_name, port, api_key)] = OpenAI(
                api_key = api_key, base_url = f"http://127.0.0.1:{port}/v1"
            )
        client = GLOBAL_CLIENTS[(model_name, port, api_key)]


        response = client.embeddings.create(
            input = text,
            model = model_name
        )

        return response.data[0].embedding



async def async_query_embedding(user_query: str = None, port:int = None, model_name: str = None, api_key = 'None', return_text: bool = True):
    model_name = model_name.rstrip("/")
    global GLOBAL_ASYNC_CLIENTS, GLOBAL_ASYNC_SEMAPHORES, MAX_CONCURRENT
    if (model_name, port, api_key) not in GLOBAL_CLIENTS:
        GLOBAL_CLIENTS[(model_name, port, api_key)] = AsyncOpenAI(
            api_key = api_key, base_url = f"http://127.0.0.1:{port}/v1"
        )

    if (model_name, port, api_key) not in GLOBAL_ASYNC_SEMAPHORES:
        sema = RedisDistributedSemaphore(
            redis_url="redis://localhost:6379",
            key=f"{model_name}:{port} - {api_key}",
            max_concurrency = MAX_CONCURRENT
        )
        await sema.connect()
        GLOBAL_ASYNC_SEMAPHORES[(model_name, port, api_key)] = sema

    semaphore = GLOBAL_ASYNC_SEMAPHORES[(model_name, port, api_key)]

    client = GLOBAL_CLIENTS[(model_name, port, api_key)]

    semaphore.acquire()
    response = client.embeddings.create(
        input = [user_query],
        model = model_name
    )
    semaphore.release()

    return response.data[0].embedding
