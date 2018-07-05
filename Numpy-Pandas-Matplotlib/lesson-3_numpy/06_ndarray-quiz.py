import numpy as np

# Using the Built-in functions you learned about in the
# previous lesson, create a 4 x 4 ndarray that only
# contains consecutive even numbers from 2 to 32 (inclusive)

X = np.linspace(2, 32, 16, dtype=int).reshape(4, 4)
print(X)
