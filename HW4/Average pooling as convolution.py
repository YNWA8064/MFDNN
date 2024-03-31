import numpy as np


def avpoll(a, x_polling=2, y_polling=2, stride=2):
    b = np.ones((x_polling, y_polling))
    result = np.zeros((a.shape[0], a.shape[1]//y_polling, a.shape[2]//x_polling))
    for l in range(a.shape[0]):
        for i in range(0, a.shape[1]//y_polling):
            for j in range(0, a.shape[2]//x_polling):
                result[l][i][j] = a[l][i:i+x_polling,j:j+y_polling].reshape(-1)@b.reshape(-1) / (x_polling*y_polling)
    return result


np.random.seed(0)
k = 2
C = 5
X = np.random.randint(0, 255, size=(C, 8, 4))
print(avpoll(X))