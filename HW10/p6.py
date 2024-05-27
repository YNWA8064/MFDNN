import torch
from torch import exp, sin, cos
import matplotlib.pyplot as plt

N = 500
lr = 1e-2
batch_size = 16

mu, tau = torch.tensor(0.), torch.tensor(0.)
mu_history, tau_history = torch.zeros(N+1), torch.zeros(N+1)

for i in range(N):
    x = torch.normal(mu, exp(tau), size=(batch_size,1))
    df_by_dmu = (x * (x-mu) * torch.sin(x) / (torch.exp(2*tau)) + mu - 1).mean()
    df_by_dtau = (((x-mu)**2*exp(-2*tau)-1)*x*sin(x) + exp(tau)-1).mean()
    mu -= lr*df_by_dmu
    tau -= lr*df_by_dtau
    mu_history[i+1] = mu
    tau_history[i+1] = tau

print('Log-derivative trick')
print(f'mu : {mu.item():.4f}, sigma : {tau.exp().item():.4f}')
print()

plt.plot(mu_history, exp(tau_history))
plt.show()

mu, tau = torch.tensor(0.), torch.tensor(0.)
mu_history, tau_history = torch.zeros(N+1), torch.zeros(N+1)

for i in range(N):
    x = torch.normal(0, 1, size=(batch_size,1))
    df_by_dmu = (sin(mu+exp(tau)*x) + (mu + exp(tau) * x) * cos(mu + exp(tau) * x)).mean() + mu - 1
    df_by_dtau = (exp(tau) * x * sin(mu + exp(tau) * x) + (mu + exp(tau) * x) * cos(mu + exp(tau) * x) * exp(tau) * x).mean() + exp(tau) - 1
    mu -= lr*df_by_dmu
    tau -= lr*df_by_dtau
    mu_history[i+1] = mu
    tau_history[i+1] = tau

print('Reparameterization trick')
print(f'mu : {mu.item():.4f}, sigma : {tau.exp().item():.4f}')

plt.plot(mu_history, exp(tau_history))
plt.show()
