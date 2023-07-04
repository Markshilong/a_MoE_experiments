import torch

gpu_available = torch.cuda.is_available()
if gpu_available:
    num_gpus = torch.cuda.device_count()
    print("PyTorch can use GPU.")
    print("Number of available GPUs:", num_gpus)
else:
    print("PyTorch cannot use GPU.")
