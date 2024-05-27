import numpy as np

N = 3000
K = 600
p = 18/37
q = 0.55

rich = 0

for i in range(N):
    BUDGET = 100
    for j in range(K):
        a = np.random.uniform(0, 1)
        BUDGET = BUDGET + 1 if a <= p else BUDGET - 1

        if BUDGET == 0:
            break
        if BUDGET == 200:
            rich += 1
            break
print('Without IS, ', rich/N)

rich = 0

for i in range(N):
    BUDGET = 100
    l = 1
    for j in range(K):
        a = np.random.uniform(0, 1)
        if a <= q:
            BUDGET += 1
            l *= p/q
        else:
            BUDGET -= 1
            l *= (1-p)/(1-q)

        if BUDGET == 0:
            break
        if BUDGET == 200:
            rich += l
            break
print('With IS, ', rich/N)
