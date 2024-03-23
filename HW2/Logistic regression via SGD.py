import numpy as np
import matplotlib.pyplot as plt

N, p = 30, 20
np.random.seed(0)
X = np.random.randn(N, p)
Y = 2*np.random.randint(2, size = N) - 1

theta = np.zeros(p)
alpha = 0.3

def cal(theta):
    return sum([np.log(1+np.exp(-Y[i]*X[i,:]@theta)) for i in range(N)])/N

def f(theta, index):
    return np.exp(-Y[index]*X[index,:]@theta)*(-Y[index])/(1+np.exp(-Y[index]*X[index,:]@theta))

K = 100000
f_val = []
for _ in range(K):
    idx = np.random.randint(N)
    theta -= alpha*X[idx]*f(theta,idx)
    f_val.append(cal(theta))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size = 14)
plt.plot(list(range(K)),f_val, color = "green", label = "Stochastic Gradient Descent")
plt.xlabel('Iterations')
plt.ylabel(r'$f(\theta^k)$')
plt.legend()
plt.show()