import time
from functools import wraps


def timer_time(orig_func):
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        var = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        print(f"   Function '{orig_func.__name__}' ran in {t2 * 1000} ms. (time) ")
        return var
    return wrapper