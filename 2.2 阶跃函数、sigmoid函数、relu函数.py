"""
阶跃函数
"""


# def step_function(x):
#     if x > 0:
#         return 1
#     else:
#         return 0

"""
改写为支持numpy数组
"""
import numpy as np


# def step_function(x):
#     y = x > 0
#     return y.astype(np.int)

# x = np.array([-1.0, 1.0, 2.0])
# print(x)
# y = x > 0
# print(y)

import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x > 0, dtype=np.int)


# x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()



def sigmoid(x):
    return 1/(1 + np.exp(-x))

# x = np.array([-1.0, 1.0, 2.0])
# print(sigmoid(x))

# t = np.array([1.0, 2.0, 3.0])
# print((1.0 + t))
# print((1.0 / t))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

"""
relu函数
"""


def relu(x):
    return np.maximum(0, x)
