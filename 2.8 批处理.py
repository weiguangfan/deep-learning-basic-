"""
多维数组的对应维度的元素个数是否一致；
输出为元素个数为10的一维数组；
批处理一次性计算大型数组要比分开逐步计算各个小型数组速度更快；
"""

import pickle
import numpy as np
from dataset.mnist import load_mnist

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def soft_max(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = soft_max(a3)
    return y

# 输入为一张图像时的处理流程；
# x, _ = get_data()
# network = init_network()
# w1, w2, w3 = network['W1'], network['W2'], network['W3']
# print(x.shape)  # (10000, 784)
# print(x[0].shape)  # (784,)
# print(w1.shape)  # (784, 50)
# print(w2.shape)  # (50, 100)
# print(w3.shape)  # (100, 10)

# 输入多张图像的处理流程
x, t = get_data()
network = init_network()
batch_size = 100
accuracy_cnt = 0
for i in range(0, len(x), batch_size):  # 步长为batch_size
    x_batch = x[i:i+batch_size]  # 切片：取100个数，抽取批数据
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])


# range(start,end,stride)函数
print(list(range(0, 10)))
print(list(range(0, 10, 3)))

# 获取数组值最大的索引，按照不同的轴
x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
y = np.argmax(x, axis=1)
print(y)

y = np.array([1, 2, 1, 0])
t = np.array([1, 2, 0, 0])
print(y == t)
print(np.sum(y == t))

