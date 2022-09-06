"""
刚才登场的 h（x）函数会将输入信号的总和转换为输出信号，这种函数一般称为激活函数（activation function）。
如“激活”一词所示，激活函数的作用在于决定如何来激活输入信号的总和。
激活函数是连接感知机和神经网络的桥梁。

“朴素感知机”是指单层网络，指的是激活函数使用了阶跃函数的模型。
“多层感知机”是指神经网络，即使用 sigmoid 函数（后述）等平滑的激活函数的多层网络。
阶跃函数是指以阈值为界,一旦输入超过阈值，就切换输出的函数。
因此，可以说感知机中使用了阶跃函数作为激活函数。
也就是说，在激活函数的众多候选函数中，感知机使用了阶跃函数。
阶跃函数数学表达式的实现
"""


# def step_function(x):
#     if x > 0:  #参数 x 只能接受实数（浮点数）。
#         return 1
#     else:
#         return 0

"""
改写为支持numpy数组
对 NumPy 数组进行不等号运算后，数组的各个元素都会进行不等号运算，生成一个布尔型数组。
这里，数组 x 中大于 0 的元素被转换为 True，小于等于 0的元素被转换为 False，从而生成一个新的数组 y。
数组 y 是一个布尔型数组，但是我们想要的阶跃函数是会输出 int 型的 0或 1 的函数。
因此，需要把数组 y 的元素类型从布尔型转换为 int 型。
"""
import numpy as np


# def step_function(x):
#     y = x > 0  #允许参数取 NumPy 数组
#     return y.astype(np.int)


# x = np.array([-1.0, 1.0, 2.0])
# print(x)
# y = x > 0
# print(y)
# print(step_function(y))
# y = y.astype(np.int)
# print(y)

import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x > 0, dtype=np.int)

# x = np.array([-1.0, 1.0, 2.0])
# print(x)
# y = x > 0
# print(y)
# print(step_function(y))
# y = y.astype(np.int)
# print(y)


"""
阶跃函数的图形
阶跃函数以 0 为界，输出从 0 切换为 1（或者从 1 切换为0）。
它的值呈阶梯式变化
阶跃函数以 0 为界，输出发生急剧性的变化。
阶跃函数只能返回 0 或 1
感知机中神经元之间流动的是 0 或 1 的二元信号
输出信号的值都在 0 到 1 之间。
"""

# x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1) # 指定y轴的范围
# plt.show()

"""
sigmoid 函数的实现
参数 x 为 NumPy 数组时，这个sigmoid 函数中输出一个 NumPy 数组
据 NumPy 的广播功能，如果在标量和 NumPy 数组之间进行运算，则标量会和 NumPy 数组的各个元素进行运算。
"""


def sigmoid(x):
    return 1/(1 + np.exp(-x))


# x = np.array([-1.0, 1.0, 2.0])
# print(sigmoid(x))

# t = np.array([1.0, 2.0, 3.0])
# print((1.0 + t))
# print((1.0 / t))

"""
sigmoid 函数画在图上
sigmoid 函数是一条平滑的曲线，输出随着输入发生连续性的变化。
sigmoid 函数可以返回0.731 ...、0.880 ... 等实数
神经网络中流动的是连续的实数值信号。
输出信号的值都在 0 到 1 之间。
"""

# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmoid(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()

"""
relu函数的实现
"""


def relu(x):
    return np.maximum(0, x)


a = np.random.randn(2, 3)
print(a)
b = np.maximum(0, a)
print(b)
