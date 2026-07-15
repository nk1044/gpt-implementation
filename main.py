import torch
from model import LanguageModel
from data import get_batch
from tokenizer import decode
from config import N_TRAIN

model = LanguageModel()
print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M Parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for _ in range(N_TRAIN):
    x, y = get_batch("train")
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
print(decode(model.generate(x = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))