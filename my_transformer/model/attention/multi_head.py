import torch
import torch.nn as nn
import torch.nn.functional as F

from scaled_dot_product import ScaledDotProductAttention

class AttentionHead(nn.Module):
    def __init__(self, HP):
        super().__init__()
        self.Q = nn.Linear(HP.emb_dim, HP.d_k)
        self.K = nn.Linear(HP.emb_dim, HP.d_k)
        self.V = nn.Linear(HP.emb_dim, HP.d_k)
        self.d_k = HP.d_k

    def forward(self, input1, input2):
        return ScaledDotProductAttention(self.Q(input2), self.K(input1), self.V(input1), self.d_k)

class MultiHeadAttention(nn.Module):
    def __init__(self, HP, mask = False):
        super().__init__()
        self.h = HP.h
        self.mask = mask

    def forward(self, input1, input2):
        return