import torch
import torch.nn as nn
from torch.nn import functional as F

from config import N_EMBEDDINGS, CONTEXT_WINDOW
from tokenizer import VOCAB_SIZE


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, N_EMBEDDINGS)
        self.pos_embedding = nn.Embedding(CONTEXT_WINDOW, N_EMBEDDINGS)
        self.lm_head = nn.Linear(N_EMBEDDINGS, VOCAB_SIZE)

    def forward(self, x, target=None):
        B, T = x.shape
        tok_embd = self.token_embedding(x) # (B, T) -> (B, T, C)
        pos_embd = self.pos_embedding(torch.arange(T)) # (T, C)
        x = tok_embd+pos_embd
        logits = self.lm_head(x) # (B, T, C) -> (B, T, vocab_size)

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



