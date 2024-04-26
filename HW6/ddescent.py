import matplotlib.pyplot as plt 
import numpy as np
from time import time

ttt = time()

"""
Step 1 : Generate Toy data
"""

d = 35
n_train, n_val, n_test = 300, 60, 30
np.random.seed(0)
beta = np.random.randn(d)
beta_true = beta / np.linalg.norm(beta)
# Generate and fix training data
X_train = np.array([np.random.multivariate_normal(np.zeros(d), np.identity(d)) for _ in range(n_train)])
Y_train = X_train @ beta_true + np.random.normal(loc = 0.0, scale = 0.5, size = n_train)
# Generate and fix validation data (for tuning lambda). 
X_val = np.array([np.random.multivariate_normal(np.zeros(d), np.identity(d)) for _ in range(n_val)])
Y_val = X_val @ beta_true 
# Generate and fix test data
X_test = np.array([np.random.multivariate_normal(np.zeros(d), np.identity(d)) for _ in range(n_test)])
Y_test = X_test @ beta_true


"""
Step 2 : Solve the problem
"""    


lambda_list = [2 ** i for i in range(-6, 6)]
num_params = np.arange(1, 1501, 10)

def cal_error(X, Y, theta):
    return np.sum([np.linalg.norm(X[i]@theta-Y[i]) for i in range(len(Y))])/2

def cal_theta(X_tilda, lamb, Y):
    p = X_tilda.shape[1]
    return np.linalg.inv(np.transpose(X_tilda)@X_tilda + lamb*np.identity(p)) @ np.transpose(X_tilda) @ Y

errors_opt_lambda = []
errors_fixed_lambda = []
for p in num_params:
    W = np.random.normal(loc=0, scale=np.sqrt(1/p), size=(p, d))
    X_tilda_train = np.array([np.maximum(0, W @ x_train) for x_train in X_train])
    theta_fixed_lambda = cal_theta(X_tilda_train, 0.01, Y_train)

    theta_per_lambda_list = [cal_theta(X_tilda_train, lamb, Y_train) for lamb in lambda_list]
    X_tilda_val = np.array([np.maximum(0, W @ x_val) for x_val in X_val])
    min_index = np.argmin([cal_error(X_tilda_val, Y_val, theta) for theta in theta_per_lambda_list])
    theta = theta_per_lambda_list[min_index]

    X_tilda_test = np.array([np.maximum(0, W @ x_test) for x_test in X_test])
    errors_fixed_lambda.append(cal_error(X_tilda_test, Y_test, theta_fixed_lambda))
    errors_opt_lambda.append(cal_error(X_tilda_test, Y_test, theta))
    print(f'time for p={p} : {time()-ttt}')
    ttt = time()

"""
Step 3 : Plot the results
"""

plt.figure(figsize = (24, 8))
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.rc('font', size = 24)


plt.scatter(num_params, errors_fixed_lambda, color = 'black',
            label = r"Test error with fixed $\lambda = 0.01$",
            )
plt.legend()

plt.plot(num_params, errors_opt_lambda, 'k', label = r"Test error with tuned $\lambda$")
plt.legend()
plt.xlabel(r'$\#$ parameters')
plt.ylabel('Test error')
plt.title(r'Test error vs. $\#$ params')

plt.savefig('double_descent.png')
plt.show()