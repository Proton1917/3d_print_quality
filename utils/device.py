"""
设备选择工具函数
"""
import torch


def get_device(use_gpu=True, verbose=True):
    """
    获取最佳可用设备
    
    参数:
        use_gpu: 是否使用GPU
        verbose: 是否打印设备信息
        
    返回:
        torch.device: 选择的设备
    """
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            if verbose:
                print(f"使用CUDA GPU：{torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            if verbose:
                print("使用Apple Silicon MPS")
        else:
            device = torch.device('cpu')
            if verbose:
                print("GPU不可用，使用CPU")
    else:
        device = torch.device('cpu')
        if verbose:
            print("使用CPU")
    
    return device


def is_mps_device(device):
    """
    检查是否为MPS设备
    
    参数:
        device: torch.device对象
        
    返回:
        bool: 是否为MPS设备
    """
    return str(device).startswith('mps')


def move_to_device(tensor_or_module, device):
    """
    将张量或模块移动到指定设备
    
    参数:
        tensor_or_module: 张量或模块
        device: 目标设备
        
    返回:
        移动后的张量或模块
    """
    if hasattr(tensor_or_module, 'to'):
        return tensor_or_module.to(device)
    else:
        return tensor_or_module
