from gpu_config import gpu_config
import torch

def print_gpu_status():
    """Print current GPU status"""
    if gpu_config.is_gpu_available():
        print(f"\nGPU Status: {gpu_config.get_num_gpus()} GPUs available")
        for i in gpu_config.get_available_gpus():
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
    else:
        print("No GPUs available - using CPU")

def clear_gpu_cache():
    """Clear GPU cache on all devices"""
    if gpu_config.is_gpu_available():
        for gpu_id in gpu_config.get_available_gpus():
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
        print("GPU cache cleared on all devices")