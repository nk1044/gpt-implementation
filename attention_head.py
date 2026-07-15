import torch
import torch.nn as nn
from torch.nn import functional as F
from config import N_EMBEDDINGS

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(N_EMBEDDINGS, head_size, bias=False)
        self.key = nn.Linear(N_EMBEDDINGS, head_size, bias=False)
        self.value = nn.Linear(N_EMBEDDINGS, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(N_EMBEDDINGS, N_EMBEDDINGS)))
     
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) # (B, T, head_size)
        k = self.key(x) # (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, h_s) @ (B, h_s, T) -> (B, T, T)
        wei = F.softmax(wei)

        v = self.value(x)
        out = wei @ v
        return out

