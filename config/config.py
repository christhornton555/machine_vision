import torch

def select_device(prefer_gpu=True):
    """
    Select the device to use for processing: GPU (if available) or CPU.

    Args:
        prefer_gpu (bool): If True, will prefer GPU if available. If False, always use CPU.

    Returns:
        torch.device: The device (CPU or GPU) to be used.
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Device selected: GPU (CUDA)")
    else:
        device = torch.device('cpu')
        print("Device selected: CPU")

    return device
