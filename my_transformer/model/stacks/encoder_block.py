import torch
import torch.nn as nn
import torch.nn.functional as F

from attention.layer_normalization import LayerNorm
from attention.multi_head import MultiHeadAttention
from attention.positionwise_fully_connected import FullyConnectedLayer
from attention.residual_connection import ResidualConnection
from attention.scaled_dot_product import ScaledDotProductAttention


class EncoderBlock(nn.Module):
    def __init__(self, HP):
        super().__init__()
        