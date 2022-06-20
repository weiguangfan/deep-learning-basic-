import numpy as np

y = [0.1, 0.005, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)


# print(mean_squared_error(np.array(y), np.array(t)))
# y = [0.1, 0.005, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
# print(mean_squared_error(np.array(y), np.array(t)))


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


print(cross_entropy_error(np.array(y), np.array(t)))
y = [0.1, 0.005, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))



