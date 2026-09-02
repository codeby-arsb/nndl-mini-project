import torch
from typing import Optional, Tuple, Dict, Any

def is_cuda_available() -> bool:
    """Return True if NVIDIA CUDA is available."""
    return torch.cuda.is_available()

def is_mps_available() -> bool:
    """Return True if Apple Silicon MPS backend is built and available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

def get_device() -> torch.device:
    """
    Select the optimal available compute device.
    Priority: CUDA -> MPS -> CPU
    """
    if is_cuda_available():
        return torch.device("cuda")
    elif is_mps_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_device_name(device: Optional[torch.device] = None) -> str:
    """Return a descriptive human-readable name for the device."""
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)
        
    if device.type == "cuda":
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NVIDIA CUDA"
    elif device.type == "mps":
        return "Apple Silicon MPS"
    else:
        return "CPU"

def device_synchronize(device: Optional[torch.device] = None) -> None:
    """Synchronize compute stream on the selected device if applicable."""
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)
        
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()

def empty_device_cache(device: Optional[torch.device] = None) -> None:
    """Release unallocated cached memory on the accelerator."""
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)
        
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

def get_amp_device_type(device: Optional[torch.device] = None) -> Optional[str]:
    """
    Return the backend string for torch.amp (e.g. 'cuda', 'mps') if supported,
    or None if AMP is not applicable.
    """
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)
        
    if device.type in ("cuda", "mps"):
        return device.type
    return None

def get_memory_stats(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Return memory allocation statistics in MB for the active device.
    Returns empty dict for CPU.
    """
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)
        
    stats = {}
    if device.type == "cuda" and torch.cuda.is_available():
        stats["allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
        stats["reserved_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)
    elif device.type == "mps" and hasattr(torch, "mps"):
        if hasattr(torch.mps, "current_allocated_memory"):
            stats["allocated_mb"] = torch.mps.current_allocated_memory() / (1024 ** 2)
        if hasattr(torch.mps, "driver_allocated_memory"):
            stats["driver_allocated_mb"] = torch.mps.driver_allocated_memory() / (1024 ** 2)
    return stats
