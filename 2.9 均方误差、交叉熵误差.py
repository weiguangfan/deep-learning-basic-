"""
损失函数：
均方误差mean squared error：输出结果和监督数据越吻合，误差越小；
交叉熵误差cross entropy error：值是正确解标签所对应的输出结果决定的；
"""


import numpy as np
# 设2为正确解
y = [0.1, 0.005, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]


def mean_squared_error(y, t):
    """均方误差的实现：0.5是系数"""
    return 0.5 * np.sum((y - t)**2)


# print(mean_squared_error(np.array(y), np.array(t)))
# 7的概率最高的情况
# y = [0.1, 0.005, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
# print(mean_squared_error(np.array(y), np.array(t)))


def cross_entropy_error(y, t):
    """交叉熵损失的实现：log是ln，和式前有负号；delta 防止负无限大；"""
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


print(cross_entropy_error(np.array(y), np.array(t)))
# 7的概率最高的情况
y = [0.1, 0.005, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))
