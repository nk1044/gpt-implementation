import torch
import torch.nn as nn
from torch.nn import functional as F

from config import N_EMBEDDINGS, CONTEXT_WINDOW
from tokenizer import VOCAB_SIZE


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, VOCAB_SIZE)

    def forward(self, x, target=None):
        logits = self.token_embedding(x)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)

        return logits, loss
    
    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            x = x[:, -CONTEXT_WINDOW:]
            logits, _ = self.forward(x) # (B, T, C)
            logits = logits[:, -1, :] # (B, C)
            prob = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(prob, num_samples=1) # (B, 1)
            x = torch.cat((x, x_next), dim=-1) # (B, T+1)
        return x



