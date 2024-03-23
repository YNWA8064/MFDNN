import numpy as np
import matplotlib.pyplot as plt

N = 30
np.random.seed(0)
X = np.random.randn(2, N)
y = np.sign(X[0, :] ** 2 + X[1, :] ** 2 - 0.7)
theta = 0.5
c, s = np.cos(theta), np.sin(theta)
X = np.array([[c, -s], [s, c]]) @ X
X = X + np.array([[1], [1]])

def kernel(x):
    l = np.size(x[0])
    return np.asarray([[1, x[0,i], x[0,i]**2, x[1,i], x[1,i]**2] for i in range(l)])

k_x = kernel(X)


METHOD = 'LOG REG' # 'SVM' or 'LOG REG'

if METHOD == 'SVM':
    theta = np.zeros(5)
    alpha = 0.01
    lam = 0.1


    def cal(theta):
        return np.average(np.maximum(1 - y * (k_x @ theta), 0)) # + lam * np.sum(theta ** 2)


    def f(theta, index):
        global cnt
        _x = 1 - y[index] * k_x[index, :] @ theta
        return 0 * (_x <= 0) - y[index] * (_x > 0)


    K = 100000
    f_val = []
    for _ in range(K):
        idx = np.random.randint(N)
        theta -= alpha * (k_x[idx] * f(theta, idx))# + lam * 2 * theta)
        f_val.append(cal(theta))
    print(f_val[-1])

elif METHOD == 'LOG REG':
    theta = np.zeros(5)
    alpha = 0.1
    def cal(theta):
        return sum([np.log(1 + np.exp(-y[i] * k_x[i, :] @ theta)) for i in range(N)]) / N


    def f(theta, index):
        return np.exp(-y[index] * k_x[index, :] @ theta) * (-y[index]) / (1 + np.exp(-y[index] * k_x[index, :] @ theta))


    K = 100000
    f_val = []
    for _ in range(K):
        idx = np.random.randint(N)
        theta -= alpha * k_x[idx] * f(theta, idx)
        f_val.append(cal(theta))

xx = np.linspace(-4, 4, 1024)
yy = np.linspace(-4, 4, 1024)
xx, yy = np.meshgrid(xx, yy)
Z = theta[0] + (theta[1] * xx + theta[2] * xx **2) + (theta[3] * yy + theta[4] * yy **2)
plt. contour(xx, yy, Z, 0)

for i in range(np.size(X[0])):
    color = 'red' if y[i]>0 else 'blue'
    plt.scatter(X[0, i], X[1, i], c=color)
plt.show()
