# app/utils/timed.py
import functools
import time
from typing import Any, Callable
import sys


def timed(fn: Callable) -> Callable:
    """Log execution time of any method only if run from __main__."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        # Check if the top-level script is __main__
        if sys.modules["__main__"].__file__ == sys.modules[fn.__module__].__file__:
            from rich import print

            print(f"⏱ {fn.__name__}: {elapsed:.2f}s")
        return result

    return wrapper
