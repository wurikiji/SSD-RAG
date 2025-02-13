import torch

print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")
print(f"현재 사용중인 GPU: {torch.cuda.current_device()}")
