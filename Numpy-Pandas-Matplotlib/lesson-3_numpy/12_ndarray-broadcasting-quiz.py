import numpy as np

# Use Broadcasting to create a 4 x 4 ndarray that has its first
# column full of 1s, its second column full of 2s, its third
# column full of 3s, etc..


X = np.ones((4, 1))
for c in range(2, 5):
    X = np.append(X, [[c], [c], [c], [c]], axis=1)
    c += 1
