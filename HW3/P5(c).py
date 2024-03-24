import torch
import numpy as np

L=5
np.random.seed(80)
n_list = np.random.randint(1,11, size=L)
x = torch.tensor(np.random.randn(n_list[0]), dtype=torch.float)

def sig(z):
    return 1/(1+np.exp(-z))

def tanh(z):
    return (1-np.exp(-2*z))/(1+np.exp(-2*z))

class MLP(torch.nn.Module):
    def __init__(self, n_list, act_f):
        super(MLP, self).__init__()
        self._n_list = n_list
        self.act_f = act_f
        self.fc1 = torch.nn.Linear(self._n_list[0], self._n_list[1], bias=True)
        self.fc2 = torch.nn.Linear(self._n_list[1], self._n_list[2], bias=True)
        self.fc3 = torch.nn.Linear(self._n_list[2], self._n_list[3], bias=True)
        self.fc4 = torch.nn.Linear(self._n_list[3], self._n_list[4], bias=True)

    def forward(self, x):
        if self.act_f == 'sig':
            x = torch.sigmoid(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
            x = self.fc4(x)
        elif self.act_f == 'tanh':
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            x = self.fc4(x)
        return x

sig_model = MLP(n_list, 'sig')
tanh_model = MLP(n_list, 'tanh')

tanh_model.fc1.weight.data = sig_model.fc1.weight.data/2
tanh_model.fc1.bias.data = sig_model.fc1.bias.data/2
tanh_model.fc2.weight.data = sig_model.fc2.weight.data/4
tanh_model.fc2.bias.data = sig_model.fc2.bias.data/2 + sig_model.fc2.weight.data@torch.tensor([1 for i in range(n_list[1])],dtype=torch.float)/4
tanh_model.fc3.weight.data = sig_model.fc3.weight.data/4
tanh_model.fc3.bias.data = sig_model.fc3.bias.data/2 + sig_model.fc3.weight.data@torch.tensor([1 for i in range(n_list[2])],dtype=torch.float)/4
tanh_model.fc4.weight.data = sig_model.fc4.weight.data/2
tanh_model.fc4.bias.data = sig_model.fc4.bias.data + sig_model.fc4.weight.data@torch.tensor([1 for i in range(n_list[3])],dtype=torch.float)/2

print(sig_model(x) == tanh_model(x))
print(sig_model(x))
print(tanh_model(x))
