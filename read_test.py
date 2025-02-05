import mmap
import subprocess
import fire
import io
import torch
import time
import os

def main(path: str):

    buf_st = time.perf_counter()
    buffer = io.BytesIO()
    with open(path, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            buffer.write(chunk)
    buffer.seek(0)
    loaded = torch.load(buffer)
    buf_end = time.perf_counter()
    print(f"With Buffer {buf_end - buf_st}", flush=True)

    '''
    direct_st = time.perf_counter()
    loaded2 = torch.load(path)
    direct_end = time.perf_counter()
    print(f"Without Buffer {direct_end - direct_st}", flush=True)
    '''

    '''
    mmap_st = time.perf_counter()
    with open(path, 'rb', buffering=2*1024*1024) as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        # mm[:] 를 호출하면 OS가 파일 전체 내용을 메모리에 읽어옵니다.
        data = mm[:]
        # BytesIO 객체에 전체 데이터를 담습니다.
        buffer = io.BytesIO(data)
        # torch.load 를 호출할 때는 적절한 encoding 인자를 사용 (예: 'latin1')
        loaded3 = torch.load(buffer)
    mmap_end = time.perf_counter()
    print(f"With mmap {mmap_end - mmap_st}")
    '''

    '''
    os_st = time.perf_counter()
    buffer = io.BytesIO()
    fd = os.open(path, os.O_RDONLY)
    try:
        while True:
            chunk = os.read(fd, 10 * 1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)
    finally: 
        os.close(fd)
    buffer.seek(0)
    loaded4 = torch.load(buffer)
    os_end = time.perf_counter()
    print(f"With OS READ {os_end - os_st}")
    '''

    '''
    dd_st = time.perf_counter()
    
    result = subprocess.run(['dd', f"if={path}", "bs=1M"], stdout=subprocess.PIPE, check=True)
    buffer = io.BytesIO(result.stdout)
    #loaded5 = torch.load(buffer)

    dd_end = time.perf_counter()

    print(f"with dd: {dd_end - dd_st}")
    '''







if __name__ == "__main__":
    fire.Fire(main)
