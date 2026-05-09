import asyncio
from tqdm import tqdm

def async_tqdm(func):
    async def wrapper(*args, pbar = None, **kwargs):
        result = await func(*args, **kwargs)
        if pbar is not None:
            assert isinstance(pbar, tqdm)
            pbar.update(1)
        return result
    return wrapper



def async_sleep(delay: "float"):
    return asyncio.sleep(delay)

def async_create_task(coro, *, name: "str | None" = None):
    return asyncio.create_task(coro, name = name)
    
def async_run(main, *, debug: "bool | None" = None):
    return asyncio.run(main, debug = debug)

def async_gather(*coro_or_future1):
    return asyncio.gather(*coro_or_future1)
    

async def async_await_gather(tasks: list, desc: "str | None" = None):
    results = []
    for f in tqdm(asyncio.as_completed(tasks), total = len(tasks), desc = desc):
         results += [await f]
    return results

    