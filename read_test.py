import mmap
import subprocess
import fire
import io
import torch
import time
import os

def main():

    buf_st = time.perf_counter()
    buffer = io.BytesIO()
    with open('data/cache/doc_1619.txt-0.pt', 'rb') as f:
        while chunk := f.read(1024 * 1024):
            buffer.write(chunk)
    buffer.seek(0)
    load_st = time.perf_counter()
    loaded = torch.load(buffer, weights_only=True, map_location="cuda:0")
    load_end = time.perf_counter()
    buf_end = time.perf_counter()
    print(f"With Buffer {buf_end - buf_st}", flush=True)
    print(f"  Load: {load_end - load_st}")

    direct_st = time.perf_counter()
    loaded2 = torch.load('data/cache/doc_1620.txt-0.pt', weights_only=True, map_location="cuda:0")
    direct_end = time.perf_counter()
    print(f"Without Buffer {direct_end - direct_st}", flush=True)

    mmap_st = time.perf_counter()
    with open('data/cache/doc_1621.txt-0.pt', 'rb', buffering=2*1024*1024) as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        # mm[:] 를 호출하면 OS가 파일 전체 내용을 메모리에 읽어옵니다.
        data = mm[:]
        # BytesIO 객체에 전체 데이터를 담습니다.
        buffer = io.BytesIO(data)
        # torch.load 를 호출할 때는 적절한 encoding 인자를 사용 (예: 'latin1')
        load_st = time.perf_counter()
        loaded3 = torch.load(buffer, weights_only=True, map_location="cuda:0")
        load_end = time.perf_counter()
    mmap_end = time.perf_counter()
    print(f"With mmap {mmap_end - mmap_st}")
    print(f"  Load: {load_end - load_st}")

    os_st = time.perf_counter()
    buffer = io.BytesIO()
    fd = os.open('data/cache/doc_1622.txt-0.pt', os.O_RDONLY)
    try:
        while True:
            chunk = os.read(fd, 10 * 1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)
    finally: 
        os.close(fd)
    buffer.seek(0)
    load_st = time.perf_counter()
    loaded4 = torch.load(buffer, weights_only=True, map_location="cuda:0")
    load_end = time.perf_counter()
    os_end = time.perf_counter()
    print(f"With OS READ {os_end - os_st}")
    print(f"  Load: {load_end - load_st}")


if __name__ == "__main__":
    fire.Fire(main)
