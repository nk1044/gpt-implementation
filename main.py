import torch
from model import LanguageModel
from data import get_batch
from tokenizer import decode

model = LanguageModel()

x, y = get_batch("train")
print(x.shape, y.shape)
logits, loss = model(x, y)
print(logits.shape)
print(loss)

print(decode(model.generate(x = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))