import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return