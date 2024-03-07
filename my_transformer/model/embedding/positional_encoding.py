import torch
import torch.nn as nn
import torch.nn.functional as F

class Positional_Encoding(nn.Module):
    def __init__(self, HP):
        super().__init__()
        self.embedding = nn.Embedding(vocab, HP.emb_dim)

    def forward(self, input):
        return