import torch
import os
from typing import List, Optional

class GPUConfig:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GPUConfig, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.available_gpus = self._detect_available_gpus()
            self.current_gpu_index = 0
            self.gpu_lock = None
            self._initialized = True
    
    def _detect_available_gpus(self) -> List[int]:
        if not torch.cuda.is_available():
            return []
        
        num_gpus = torch.cuda.device_count()
        print(f"Detected {num_gpus} GPUs available")

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpus)))
        
        return list(range(num_gpus))
    
    def get_available_gpus(self) -> List[int]:
        return self.available_gpus.copy()
    
    def get_num_gpus(self) -> int:
        return len(self.available_gpus)
    
    def get_next_gpu(self) -> Optional[int]:
        if not self.available_gpus:
            return None
        
        gpu_id = self.available_gpus[self.current_gpu_index]
        self.current_gpu_index = (self.current_gpu_index + 1) % len(self.available_gpus)
        return gpu_id
    
    def set_current_gpu(self, gpu_id: int):
        if gpu_id in self.available_gpus:
            torch.cuda.set_device(gpu_id)
            self.current_gpu_index = self.available_gpus.index(gpu_id)
    
    def get_current_gpu(self) -> Optional[int]:
        if not self.available_gpus:
            return None
        return self.available_gpus[self.current_gpu_index]
    
    def is_gpu_available(self) -> bool:
        """Check if any GPU is available"""
        return len(self.available_gpus) > 0
    
    def get_device(self, gpu_id: Optional[int] = None) -> torch.device:
        if gpu_id is not None and gpu_id in self.available_gpus:
            return torch.device(f"cuda:{gpu_id}")
        elif self.available_gpus:
            return torch.device(f"cuda:{self.available_gpus[0]}")
        else:
            return torch.device("cpu")

gpu_config = GPUConfig()