import functools
import time
import sys
def progress_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Starting '{func.__name__}'...")
        start_time = time.time()  # 시작 시간 기록
        result = func(*args, **kwargs)
        end_time = time.time()  # 종료 시간 기록
        elapsed_time = end_time - start_time
        print(f"Finished '{func.__name__}' in {elapsed_time:.2f} seconds.")
        return result
    return wrapper

def progress_bar(current, total):
    percent = (current / total) * 100
    bar_length = 40 
    block = int(round(bar_length * percent / 100))
    progress = '#' * block + '-' * (bar_length - block)
    sys.stdout.write(f'\rProgress: |{progress}| {percent:.2f}% Complete')
    sys.stdout.flush()