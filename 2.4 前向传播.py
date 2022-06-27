"""
3 层神经网络的实现
"""
import numpy as np
#输入信号、权重、偏置设置成任意值。
x = np.array([1.0, 0.5])
w1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
b1 = np.array([0.1, 0.2, 0.3])
print(w1.shape)
print(x.shape)
print(b1.shape)
a1 = np.dot(x, w1) + b1
print(a1)
#隐藏层的加权和（加权信号和偏置的总和）用 a 表示，被激活函数转换后的信号用 z 表示。
def sigmoid(x):
    return 1/(1 + np.exp(-x))

z1 = sigmoid(a1)
print(z1)
#实现第 1 层到第 2 层的信号传递
w2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
b2 = np.array([0.1, 0.2])
print(z1.shape)
print(w2.shape)
print(b2.shape)
a2 = np.dot(z1, w2) + b2
print(a2)
z2 = sigmoid(a2)
print(z2)
#第 2 层到输出层的信号传递
def identity_function(x):
    return x

w3 = np.array([[0.1, 0.3], [0.2, 0.4]])
b3 = np.array([0.1, 0.2])
a3 = np.dot(z2, w3) + b3
print(a3)
z3 = identity_function(a3)
print(z3)
"""
3 层神经网络的实现。现在我们把之前的代码实现全部整理一下。
init_network() 函数会进行权重和偏置的初始化，并将它们保存在字典变量 network 中。
这个字典变量 network 中保存了每一层所需的参数（权重和偏置）。
forward() 函数中则封装了将输入信号转换为输出信号的处理过程。
"""


def init_network():
    network = {}
    network['w1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['w2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['w3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


def forward(network, x):
    w1, w2, w3 = network['w1'], network['w2'], network['w3'],
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = identity_function(a3)
    return y


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
