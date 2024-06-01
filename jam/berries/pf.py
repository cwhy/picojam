import functools
from typing import Callable
from better_partial import _, partial
from jax import vmap

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
        return PointFreeFunction(lambda x: self.func(x)[key])

F = PointFreeFunction
