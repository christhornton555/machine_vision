# import torch
# print(torch.cuda.is_available())  # This should return True if CUDA is available
# print(torch.cuda.current_device())  # This should return the GPU device ID
# print(torch.cuda.get_device_name(torch.cuda.current_device()))  # This should return the name of your GPU

import torch
print(torch.__version__)
print(torch.version.cuda)

from torchvision import ops
print(ops.nms.__module__)
