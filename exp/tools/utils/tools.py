from typing import Type, TypeVar, Callable, List, Literal, Iterable
import os, random, itertools
from enum import Enum, auto
from collections import defaultdict
from exp.tools.utils.iotools import save_pkl, read_pkl

CALLABLE = TypeVar("CALLABLE")

class Marks(Enum):
    ckpt = auto()
    stop = auto()


def chunks(lst: list, chunk_num: int):
    """Yield successive n-sized chunks from lst."""
    chunk_width = len(lst) // chunk_num
    ones = chunk_num - len(lst) % chunk_num 
    p = 0
    for i in range(chunk_num):
        if i == ones: chunk_width += 1
        yield lst[p: p + chunk_width]
        p += chunk_width


def randint(a: int, b: int) -> int:
    return random.randint(a, b)

def random_sample(population, k: int) -> List[int]:
    return random.sample(population, k)

def shuffle(lst: list) -> list:
    random.shuffle(lst)
    return lst

def mapping(objs: Iterable, 
            worker: Callable, 
            tool: Callable = lambda :tuple(),
            num_processes: int = 8):
    '''tools : a function building global variables costing time, like tokenizer'''
    from tqdm import tqdm
    import multiprocessing as mp
    def mapping_worker(wk, tl, get_queue: mp.Queue, to_queue: mp.Queue): 
        t = tl() 
        while True:
            obj = get_queue.get()
            if obj == Marks.stop: break
            to_queue.put(wk(obj, *t))

    to_queue = mp.Queue()
    get_queue = mp.Queue()

    processes = []
    for i in range(num_processes):
        p = mp.Process(target=mapping_worker,
                       args=(worker, tool, to_queue, get_queue))
        p.start()
        processes += [p]

    total, ret = 0, []
    for obj in tqdm(objs, desc = "Preparing"): 
        total += 1
        to_queue.put(obj)
    for _ in range(num_processes): to_queue.put(Marks.stop)
    for _ in tqdm(range(total), desc = "Mapping"): ret += [get_queue.get()]
    for p in processes: p.join()

    return ret
    

def filtering(objs: Iterable, 
              filter: Callable, 
              tool: Callable = lambda :tuple(), 
              num_processes: int = 8):
    '''tools : a function building global variables costing time, like tokenizer'''
    from tqdm import tqdm
    import multiprocessing as mp
    def filtering_worker(ft, tl, get_queue: mp.Queue, to_queue: mp.Queue): 
        t = tl() 
        while True:
            obj = get_queue.get()
            if obj == Marks.stop: break
            if ft(obj, *t): to_queue.put(obj)
            else: to_queue.put(Marks.stop)

    to_queue = mp.Queue()
    get_queue = mp.Queue()

    processes = []
    for i in range(num_processes):
        p = mp.Process(target=filtering_worker,
                       args=(filter, tool, to_queue, get_queue))
        p.start()
        processes += [p]

    total, ret = 0, []
    for obj in tqdm(objs, desc = "Preparing"): 
        total += 1
        to_queue.put(obj)
    for _ in range(num_processes): to_queue.put(Marks.stop)
    for _ in tqdm(range(total), desc="Filtering"): 
        obj = get_queue.get()
        if obj != Marks.stop: ret += [obj]
    for p in processes: p.join()

    return ret
    


def CTX_MAPPING(length, ks = [0, 1, 2, 4, 8, 16, 32, 64, 128]):
    length /= 1000
    distances = [abs(length-k) for k in ks]
    min_ids = min(distances)
    for dis, l in zip(reversed(distances), reversed(ks)):
        if min_ids == dis:
            return f"{l}k"

