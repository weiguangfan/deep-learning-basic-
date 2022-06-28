"""
数值微分numerical differentiation
"""

import numpy as np
import matplotlib.pyplot as plt

# def numerical_diff(f, x):
#     """函数的导数"""
#     h = 10e-50  # h无限接近0，由于舍入误差，会省略小数点的精细部分的数值（小数点第8位以后的数值），导致结果上的误差
#     return (f(x + h) - f(x))/h  # 0.0

# 舍入误差，无法正确表达出来
print(np.float32(1e-50))


def numerical_diff(f, x):
    """改进版的函数的导数"""
    h = 1e-4  # 改为1e-4， 就能正确表达
    return (f(x + h) - f(x - h))/(2 * h)  # 数值微分存在误差，采用中心差分


# 简单函数构造
def function_1(x):
    return 0.01 * x**2 + 0.1*x


# 简单函数图形
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, y)
plt.show()
# x=5,x=10处的数值微分，误差小到近似等于真导数
print(numerical_diff(function_1, 5))  # 0.1999999999990898
print(numerical_diff(function_1, 10))  # 0.2999999999986347



