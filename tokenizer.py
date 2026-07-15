import torch

with open('pg100.txt', 'r', encoding='utf-8') as file:
    data = file.read()

chars = sorted(list(set(data)))

# print("Length of the code:", len(data))
# print("Number of unique characters:", len(chars))
# print("Unique characters:", "".join(chars))

VOCAB_SIZE = len(chars) # vocab_size = 101, for current data

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# print(encode("hii there"))
# print(decode(encode("hii there")))

data = torch.tensor(encode(data), dtype=torch.long)

# print(data.shape, data.dtype)

train_split = int(len(data)*0.9)

train_data = data[:train_split]
val_data = data[train_split:]