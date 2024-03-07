import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embedding.positional_encoding import Positional_Encoding
from model.embedding.token_embedding import Embedding

from model.stacks import encoder_block, decoder_block

class HyperParameters:
    def __init__(self, N=6, emb_dim=512, ff=2048, h=8, d_k=64, dropout=0.1):
        self.N = N
        self.emb_dim = emb_dim
        self.ff = ff
        self.h = h
        self.d_k = d_k
        self.dropout = dropout

class Encoder(nn.Module):
    def __init__(self, HP : HyperParameters):
        super().__init__()

        self.embedding = Embedding(HP)
        self.position = Positional_Encoding(HP)
        self.blocks = nn.ModuleList(
            [encoder_block.EncoderBlock(HP) for _ in range(HP.N)]
        )

    def forward(self, input):
        return

class Decoder(nn.Module):
    def __init__(self, HP : HyperParameters):
        super().__init__()

        self.embedding = Embedding(HP)
        self.position = Positional_Encoding(HP)
        self.blocks = nn.ModuleList(
            [encoder_block.EncoderBlock(HP) for _ in range(HP.N)]
        )

    def forward(self, input):
        return


class Transformer(nn.Module):
    def __init__(self, HP : HyperParameters):
        super().__init__()
        self.encoder = Encoder(HP)
        self.decoder = Decoder(HP)
        self.linear = nn.Linear()

    def forward(self, input, output):
        enc = self.encoder(input)
        dec = self.decoder(output, enc)
        
        return



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hp = HyperParameters()
    model = Transformer(hp).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    loss = nn.CrossEntropyLoss()

    pass