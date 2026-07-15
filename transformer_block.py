import torch.nn as nn
from torch.nn import functional as F
from attention_head import MultiHeadAttention
from config import DROPOUTS


class FeedForward(nn.Module):
    def __init__(self, num_embd, ffn_factor):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embd, num_embd*ffn_factor),
            nn.ReLU(),
            nn.Linear(num_embd*ffn_factor, num_embd),
            nn.Dropout(DROPOUTS)
        )
    def forward(self, x):
        return self.net(x)
    


class Block(nn.Module):
    def __init__(self, num_embd, num_heads, ffn_factor):
        super().__init__()
        head_size = num_embd // num_heads
        self.self_attention_layer = MultiHeadAttention(num_embd, num_heads, head_size)
        self.feed_forward_layer = FeedForward(num_embd, ffn_factor)
        self.layer_norm1 = nn.LayerNorm(num_embd)
        self.layer_norm2 = nn.LayerNorm(num_embd)
    
    def forward(self, x):
        x = self.layer_norm1(x)
        x = self.self_attention_layer(x)
        x = self.layer_norm1(x)
        x = self.feed_forward_layer(x)
        return x