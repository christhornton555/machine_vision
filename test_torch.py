import torch
print(torch.cuda.is_available())  # This should return True if CUDA is available
print(torch.cuda.current_device())  # This should return the GPU device ID
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # This should return the name of your GPU
