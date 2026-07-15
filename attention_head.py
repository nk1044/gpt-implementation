import torch
import torch.nn as nn
from torch.nn import functional as F
from config import DROPOUTS, CONTEXT_WINDOW

class Head(nn.Module):
    def __init__(self, num_embd, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(num_embd, head_size, bias=False)
        self.key = nn.Linear(num_embd, head_size, bias=False)
        self.value = nn.Linear(num_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(CONTEXT_WINDOW, CONTEXT_WINDOW)))
        self.dropout = nn.Dropout(DROPOUTS)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) # (B, T, head_size)
        k = self.key(x) # (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5 # (B, T, h_s) @ (B, h_s, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # causal mask
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_embd, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(num_embd, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_embd, num_embd)
        self.dropout = nn.Dropout(DROPOUTS)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
