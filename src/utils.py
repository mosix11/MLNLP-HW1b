import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Any
def get_gpu_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else : return None
    
def get_cpu_device():
    return torch.device('cpu')

class SimpleListDataset(Dataset):
    
    def __init__(self, data_list:list) -> None:
        super().__init__()
        self.data_list = data_list
        
    def __getitem__(self, index) -> Any:
        return self.data_list[index]
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    