import functools
import inspect
import time

class TimeRecorder:
    # data has the pair `[filename, lineno, func_name, milliseconds]`
    data = []

    @classmethod
    def register(cls, name: str):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                frame = inspect.currentframe().f_back
                info = inspect.getframeinfo(frame)
                begin = time.perf_counter()
                ret = func(*args, **kwargs)
                end = time.perf_counter()
                print(f"record {name}")
                cls.data.append([info.filename, info.lineno, name, (end - begin) * 1000])
                return ret
            return wrapper
        return decorator


if __name__ == '__main__':
    @TimeRecorder.register('record_for10000')
    def for10000():
        for i in range(0, 10000):
            pass

    for10000()
    print(TimeRecorder.data)
