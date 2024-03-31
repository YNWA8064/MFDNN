import numpy as np

def conv(a, b, padding=0):
    a = np.pad(a, (padding, padding), constant_values=0)
    layer = b.shape[0]
    x_size, y_size = b.shape[1], b.shape[2]
    result = np.zeros((layer, a.shape[0] - b[0].shape[0] + 1,a.shape[1] - b[0].shape[1] + 1))
    for l in range(layer):
        for i in range(a.shape[0] - b[l].shape[0] + 1):
            for j in range(a.shape[1] - b[l].shape[1] + 1):
                result[l][i][j] = a[i:i+x_size,j:j+y_size].reshape(-1)@b[l].reshape(-1)
    return result

np.random.seed(0)
w = np.asarray([[[0,0,0],[0,-1,1],[0,0,0]],[[0,0,0],[0,-1,0],[0,1,0]]])
X = np.random.randint(0, 255, size=(5,5))
padd = 1
print(conv(X, w, padd))