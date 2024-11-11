import time
import torch

class Timer:
    handler = print
    def __init__(self, desc, is_torch = False, trace=True):
        self.desc = desc
        self.is_torch = is_torch
        self.trace = trace

    def __enter__(self):
        if not self.trace:return self
        if self.is_torch:torch.cuda.synchronize()
        self.start = time.time()
        return self

    def __exit__(self, *args):
        if not self.trace:return
        if self.is_torch:torch.cuda.synchronize()
        self.end = time.time()
        self.interval = self.end - self.start
        self.handler(f'{self.desc} takes {self.interval} seconds')

    @classmethod
    def set_handler(cls, new_handler):
        cls.handler = new_handler

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        with Timer(func.__name__):
            return func(*args, **kwargs)
    return wrapper

class Tick:
    times = {}
    @staticmethod
    def start(name):
        Tick.times[name] = time.time()
    
    @staticmethod
    def end(name, update = False):
        if not name in Tick.times.keys():return -1
        end = time.time()
        if update:
            start = Tick.times[name]
            Tick.times[name] = end
        else:
            start = Tick.times.pop(name)
        return end - start
