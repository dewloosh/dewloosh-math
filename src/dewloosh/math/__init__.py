# -*- coding: utf-8 -*-
from typing import Callable
import numpy as np

from dewloosh.math.hist import *
from dewloosh.math.utils import *

__version__ = "0.0.1-alpha6"
__description__ = "A math library for numerical calculations."


def squeeze_if_array(arr): return np.squeeze(
    arr) if isinstance(arr, np.ndarray) else arr


def squeeze(default=True):
    def decorator(fnc: Callable):
        def inner(*args, **kwargs):
            if kwargs.get('squeeze', default):
                res = fnc(*args, **kwargs)
                if isinstance(res, tuple):
                    return list(map(squeeze_if_array, res))
                return squeeze_if_array(res)
            else:
                return fnc(*args, **kwargs)
        return inner
    return decorator


def config(*args, **kwargs):
    def decorator(fnc: Callable):
        def inner(*args, **kwargs):
            return fnc(*args, **kwargs)
        return inner
    return decorator


def is_none_or_false(a):
    if isinstance(a, bool):
        return not a
    elif a is None:
        return True
    return False
