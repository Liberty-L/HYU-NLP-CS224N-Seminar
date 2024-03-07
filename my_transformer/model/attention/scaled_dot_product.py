import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class ScaledDotProductAttention(nn.Module):
    def __init__(self, Q, K, V, d_k):
        super().__init__()
        self.Q = Q
        self.K = K
        self.V = V
        self.d_k = d_k

    def forward(self):
        score = torch.softmax(torch.matmul(self.Q, torch.transpose(self.K)) / sqrt(self.d_k))
        return torch.matmul(score, self.V)

