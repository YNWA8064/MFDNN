import torch
import torch.utils.data as data 
import torch.nn as nn
from torch.distributions.normal import Normal 
from torch.distributions.uniform import Uniform
import numpy as np
import matplotlib.pyplot as plt

epochs = 50
lr = 5e-3
batch_size = 128
n_components = 5
target_distribution = Normal(0.0, 1.0)

class Flow1d(nn.Module):
    def __init__(self, num):
        super(Flow1d, self).__init__()
        self.mus = nn.Parameter(torch.randn(num), requires_grad=True)
        self.taus = nn.Parameter(torch.randn(num), requires_grad=True)
        self.ws = nn.Parameter(torch.randn(num), requires_grad=True)

    def forward(self, x):
        x = x.view(-1,1)
        weights = self.ws.exp()
        dists = Normal(loc=self.mus, scale=self.taus.exp())
        z = (weights*(dists.cdf(x)-0.5)).sum(dim=1)
        dz_by_dx = (dists.log_prob(x).exp() * weights).sum(dim=1)
        return z, dz_by_dx


################################################
# STEP 2: Create Dataset and Create Dataloader #
################################################ 

def mixture_of_gaussians(num, mu_var=(-1,0.25, 0.2,0.25, 1.5,0.25)):
    n = num // 3
    m1,s1,m2,s2,m3,s3 = mu_var
    gaussian1 = np.random.normal(loc=m1, scale=s1, size=(n,))
    gaussian2 = np.random.normal(loc=m2, scale=s2, size=(n,))
    gaussian3 = np.random.normal(loc=m3, scale=s3, size=(num-n,))
    return np.concatenate([gaussian1, gaussian2, gaussian3])

class MyDataset(data.Dataset):
    def __init__(self, array):
        super().__init__()
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return self.array[index]

def loss_function(target_dist, z, dz_by_dx):
    log_likelihood = target_dist.log_prob(z) + dz_by_dx.log()
    return -log_likelihood.mean()

n_train, n_test = 5000, 1000
train_data, test_data = mixture_of_gaussians(n_train), mixture_of_gaussians(n_test)
train_loader = data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = data.DataLoader(test_data, shuffle=True, batch_size=batch_size)

flow = Flow1d(n_components)
optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

train_losses, test_losses = [], []

for epoch in range(epochs):
    mean_loss = 0
    for idx, x in enumerate(train_loader):
        z, dz_by_dx = flow(x)
        loss = loss_function(target_distribution, z, dz_by_dx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss += loss.item()

    train_losses.append(mean_loss / (idx + 1))

    flow.eval()
    mean_loss = 0
    for idx, x in enumerate(train_loader):
        z, dz_by_dx = flow(x)
        loss = loss_function(target_distribution, z, dz_by_dx)

        mean_loss += loss.item()
    test_losses.append(mean_loss / (idx + 1))

plt.plot(train_losses, label='train_loss')
plt.plot(test_losses, label='test_loss')
plt.legend()
plt.show()

plt.hist(train_data, bins=50)
plt.show()

with torch.no_grad():
    z, _ = flow(torch.FloatTensor(train_data))

plt.hist(np.array(z), bins=50)
plt.show()