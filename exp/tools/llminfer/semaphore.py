import asyncio, aiohttp, time, random
try:
    import aioredis
except Exception:
    from redis import asyncio as aioredis
from typing import Literal

MAX_CONCURRENT = 256
GLOBAL_ASYNC_SEMAPHORES: "dict[str, AsyncioSemaphore | RedisDistributedSemaphore]" = dict()



def set_max_concurrent(value: int = 256):
    global MAX_CONCURRENT
    MAX_CONCURRENT = value

def get_max_concurrent():
    global MAX_CONCURRENT
    return MAX_CONCURRENT

class AsyncioSemaphore:
    def __init__(self, redis_url=None, key=None, max_concurrency=256, timeout=300):
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self._local_semaphore = asyncio.Semaphore(max_concurrency)
    
    async def connect(self):
        pass
    
    async def acquire(self):
        try:
            await asyncio.wait_for(self._local_semaphore.acquire(), timeout=self.timeout)
            return True
        except asyncio.TimeoutError:
            raise TimeoutError(f"Local Semaphore wait timeout after {self.timeout}s. vLLM is processing too slow!")
    
    async def release(self):
        self._local_semaphore.release()
    
    async def close(self):
        pass

class RedisDistributedSemaphore:
    def __init__(self, redis_url, key, max_concurrency, timeout = 300):

        self.redis_url = redis_url
        self.key = key
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.redis = None
    
    async def connect(self):
        self.redis = await aioredis.from_url(self.redis_url)
    
    async def acquire(self):
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            current = await self.redis.incr(self.key)
            if current <= self.max_concurrency:
                await self.redis.expire(self.key, self.timeout + 10)
                return True
        
            await self.redis.decr(self.key)
            
            sleep_time = random.uniform(0.05, 0.15) 
            await asyncio.sleep(sleep_time)
            
        raise TimeoutError(f"Redis Semaphore wait timeout after {self.timeout}s")

    async def release(self):
        current = await self.redis.decr(self.key)
        if current < 0:
            await self.redis.set(self.key, 0)
    
    async def close(self):
        if self.redis:
            await self.redis.close()


def get_semaphore_cls(name: "Literal['asyncio', 'redis']"):
    if name == "asyncio":
        return AsyncioSemaphore
    elif name == "redis":
        return RedisDistributedSemaphore
    else:
        raise ValueError(f"Unknown semaphore type: {name}")

async def close_redis_semaphore():
    global GLOBAL_ASYNC_SEMAPHORES
    for semaphore in GLOBAL_ASYNC_SEMAPHORES.values():
        await semaphore.close()
