words = open('Names.txt', 'r').read().splitlines()
words = [x.lower() for x in words]
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['<>'] = 0


#creating the training sets for the bigram neural network
xs, ys = [], []

for w in words:
    chs = ['<>'] + list(w) + ['<>']
    for ch1, ch2 in zip(chs, chs[1:]):
        x1 = stoi[ch1]
        x2 = stoi[ch2]
        xs.append(x1)
        ys.append(x2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print(f'Total number of input samples: {num}')


# Random initialisation of the weights of the neurons for the first run (ONLY FIRST TIME!)
g = torch.Generator().manual_seed(9845165659)
W = torch.randn((27,27), generator = g, requires_grad=True)


# Gradient descent optimisation of the neural network

for Optimisation in range (100):

    # Forward pass of the neural network
    import torch.nn.functional as F
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim = True)
    loss = - probs[torch.arange(num), ys].log().mean()
    print(f'Loss=: {loss}')
    
    # Backward pass of the neural netword
    W.grad = None
    loss.backward()
    
    # Finetuning the weights of the neural netword
    learning_rate = 10
    W.data += -learning_rate * W.grad


# Sampling from the Neural Network

g = torch.Generator().manual_seed(666)
for i in range (10):
    out = []
    index = 0
    while True:
        xenc = F.one_hot(torch.tensor([index]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims = True)
    
        index = torch.multinomial(probs, num_samples = 1, replacement = True, generator = g).item()
        out.append(itos[index])
        if index == 0:
            break
    print(''.join(out))


