import numpy as np

inputs = np.array([
    [1.1, 2, 3, 0.14],
    [4.9, 0.3, 5.7, 2]
])

weights = np.array([
    [0.3, 0.1, 0.76],
    [2.1, 0.3, 0.1],
    [0.1, 0.1, 0.6],
    [0.9, 0.55, 0.12]
])

print(np.dot(inputs, weights) + np.array([1, 1, 2]))