import numpy as np
import matplotlib.pyplot as plt

N, p = 30, 20
np.random.seed(0)
X = np.random.randn(N, p)
Y = 2*np.random.randint(2, size = N) - 1

theta = np.random.randn(p)
# theta = np.zeros(p)
alpha = 0.01
lam = 0.1
cnt = 0

def cal(theta):
    return np.average(np.maximum(1-Y*(X@theta), 0)) + lam*np.sum(theta**2)

def f(theta, index):
    global cnt
    _x = 1 - Y[index]*X[index,:]@theta
    cnt += 1 if _x==0 else 0
    return 0*(_x<=0) - Y[index]*(_x>0)

K = 100000
f_val = []
for _ in range(K):
    idx = np.random.randint(N)
    theta -= alpha*(X[idx]*f(theta,idx) + lam*2*theta)
    f_val.append(cal(theta))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size = 14)
plt.plot(list(range(K)),f_val, color = "green", label = "Stochastic Gradient Descent")
plt.xlabel('Iterations')
plt.ylabel(r'$f(\theta^k)$')
plt.legend()
plt.show()
print(cnt)