import time
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

label_1, label_2 = 4, 9

train_set = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)

idx = (train_set.targets == label_1) + (train_set.targets == label_2)
train_set.data = train_set.data[idx]
train_set.targets = train_set.targets[idx]
train_set.targets[train_set.targets == label_1] = -1
train_set.targets[train_set.targets == label_2] = 1

test_set = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=True)

idx = (test_set.targets == label_1) + (test_set.targets == label_2)
test_set.data = test_set.data[idx]
test_set.targets = test_set.targets[idx]
test_set.targets[test_set.targets == label_1] = -1
test_set.targets[test_set.targets == label_2] = 1


class LR(nn.Module):
    def __init__(self, input_size = 28*28):
        super().__init__()
        self.linear = nn.Linear(input_size, 1, bias = True)

    def forward(self, x):
        return self.linear(x.float().view(-1, 28*28))


model_LR1 = LR().to(device)
model_LR2 = LR().to(device)


def KL_loss(output, target):
    return torch.mean(-torch.nn.functional.logsigmoid(output.view(-1)*target.view(-1)))


def SQUARE_loss(output, target):
    output, target = output.reshape(-1), target.reshape(-1)
    return torch.mean((1-target)*sigmoid(output)**2 + (1+target)*sigmoid(-output)**2)


loss_function1 = KL_loss
loss_function2 = SQUARE_loss
optimizer1 = torch.optim.SGD(model_LR1.parameters(), lr=1024*1e-4)
optimizer2 = torch.optim.SGD(model_LR2.parameters(), lr=1024*1e-4)

train_loader = DataLoader(dataset=train_set, batch_size=1024, shuffle=True)
for e in range(10):
    t = time.time()
    for i, l in train_loader:
        i, l = i.to(device), l.to(device)
        optimizer1.zero_grad()

        train_loss = loss_function1(model_LR1(i), l.float())
        train_loss.backward()

        optimizer1.step()
    print(f'For epoch {e+1}, time = {time.time()-t:.4f}')
test_loss, correct = 0, 0

test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

for i, l in test_loader:
    i, l = i.to(device), l.to(device)
    output = model_LR1(i)

    test_loss += loss_function1(output, l.float()).item()

    if output.item() * l.item() >= 0 :
        correct += 1

print(f'CE Loss Accuracy: {correct}/{len(test_loader)}({100*correct/len(test_loader):.2f})%')

for e in range(10):
    t = time.time()
    for i, l in train_loader:
        i, l = i.to(device), l.to(device)
        optimizer2.zero_grad()

        train_loss = loss_function2(model_LR2(i), l.float())
        train_loss.backward()

        optimizer2.step()
    print(f'For epoch {e+1}, time = {time.time()-t:.4f}')
test_loss, correct = 0, 0

test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

for i, l in test_loader:
    i, l = i.to(device), l.to(device)
    output = model_LR2(i)

    test_loss += loss_function2(output, l.float()).item()

    if output.item() * l.item() >= 0 :
        correct += 1

print(f'Square Loss Accuracy: {correct}/{len(test_loader)}({100*correct/len(test_loader):.2f})%')

