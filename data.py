import torch
from config import CONTEXT_WINDOW, BATCH_SIZE
from tokenizer import train_data, val_data

torch.manual_seed(1337)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    idx_x = torch.randint(len(data)-CONTEXT_WINDOW, (BATCH_SIZE,))
    x = torch.stack([data[i:i+CONTEXT_WINDOW] for i in idx_x]) # (B, T)
    y = torch.stack([data[i+1:i+1+CONTEXT_WINDOW] for i in idx_x])
    return x, y
