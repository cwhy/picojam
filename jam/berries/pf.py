import functools
from typing import Callable
from better_partial import _, partial
from jax import vmap
from jax import lax

def compose(f, g):
    return lambda *args, **kw: g(f(*args, **kw))

class PointFreeFunction:
    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)
    
    def f(self, *args, **kw):
        return PointFreeFunction(partial(self.func)(*args, **kw))
    
    def vmap(self, *args, **kw):
        return PointFreeFunction(vmap(self.func, *args, **kw))
    
    def map(self):
        return PointFreeFunction(lambda *args, **kw: lax.map(self.func, *args, **kw))

    def __call__(self, *args, **kw):
        return self.func(*args, **kw)
    
    def __rshift__(self, other: 'PointFreeFunction' | Callable):
        if isinstance(other, PointFreeFunction):
            return PointFreeFunction(compose(self.func, other.func))
        elif callable(other):
            return PointFreeFunction(compose(self.func, other))
        else:
            raise TypeError("other must be callable or PointFreeFunction")
    
    def __getitem__(self, key):
        return PointFreeFunction(lambda *args: self.func(*args)[key])
    
    def swap(self):
        if self.func.__code__.co_argcount != 2:
            print(self.func.__code__.co_argcount)
            raise ValueError("Function must have exactly two arguments to use swap")
        return PointFreeFunction(lambda x, y: self.func(y, x))



F = PointFreeFunction
