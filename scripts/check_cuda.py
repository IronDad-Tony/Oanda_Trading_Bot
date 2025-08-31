import os
import torch

print('torch_version=', torch.__version__)
print('torch_cuda_version=', getattr(torch.version, 'cuda', None))
print('cuda_available=', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device=', torch.cuda.get_device_name(0))
    print('capability=', torch.cuda.get_device_capability(0))

