import torch

with open ("data/shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]

def decode(tokens):
    return "".join([itos[t] for t in tokens])

# example = "Goodbye, my love!"
# tokens = encode(example)

# print(tokens)
# print(decode(tokens))

data = torch.tensor(encode(text), dtype=torch.long)

def get_batch(data, batch_size, context_length):
    ix = torch.randint(len(data) - context_length, (batch_size,))
    
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])
    
    return x, y

x, y = get_batch(data, 4, 8)

for i in range(4):
    print("INPUT :", decode(x[i].tolist()))
    print("TARGET:", decode(y[i].tolist()))
    print()