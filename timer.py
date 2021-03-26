import time
from functools import wraps


def timer_time(orig_func):
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        var = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        print("   Function '{}' ran in {} s. (time) ".format(orig_func.__name__, t2))
        return var
    return wrapper