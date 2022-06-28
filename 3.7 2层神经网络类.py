"""
神经网络的学习：
1:min-batch
    从训练数据中随机选出一部分数据，这一部分数据称为mini-batch。
    目标是减小mini-batch的损失函数的值。
2:compute gradient
    为了减小mini-batch的损失函数的值，需要求出各个权重参数的梯度。
    梯度表示损失函数值减小最多的方向。
3:update parameters
    将权重参数沿梯度方向进行微小更新。
4:repeat
    重复1、2、3

随机梯度下降法stochastic gradient descent(SGD)
"""

import sys, os
sys.path.append(os.pardir)
import numpy as np


def sigmoid(x):
    """sigmoid激活函数"""
    return 1/(1 + np.exp(-x))


def numerical_gradient(f, x):
    '''遍历数组的每一个元素，进行数值微分'''
    h = 1e-4
    grad = np.zeros_like(x)


    # for idx in range(x.size):
    #     tmp_val = x[idx]
    #     x[idx] = tmp_val + h
    #     fxh1 = f(x)
    #
    #     x[idx] = tmp_val - h
    #     fxh2 = f(x)
    #     grad[idx] = (fxh1 - fxh2)/(2*h)
    #     x[idx] = tmp_val

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
        it.iternext()

    return grad


def soft_max(a):
    """改进版的输出层的激活函数"""
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def cross_entropy_error(y, t):
    """改进版的交叉熵损失"""
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


class TwoLayerNet(object):
    """2层神经网络，1层隐藏层"""
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.0):
        """
        初始化权重；
        params:保存神经网络的参数的字典变量；
        :param input_size: 输入层的神经元数
        :param hidden_size: 隐藏层的神经元数
        :param output_size: 输出层的神经元数
        :param weight_init_std:
        """
        self.params = {}
        # 第一层的权重，高斯分布初始化
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # 第一层的偏置，0进行初始化
        self.params['b1'] = np.zeros(hidden_size)
        # 第二层的权重
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # 第二层的偏置
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        """进行推理"""
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = soft_max(a2)

        return y

    def loss(self, x, t):
        """
        损失函数
        :param x: 输入数据
        :param t: 标签数据
        :return:
        """
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accurary(self, x, t):
        """
        识别精度；
        :param x:
        :param t: 标签数据
        :return:
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(y, axis=1)
        accurary = np.sum(y == t)/float(x.shape[0])
        return accurary

    def numerical_gradient(self, x, t):
        """
        根据数值微分，计算各个权重、偏置参数相对于损失函数的梯度；
        grads:保存梯度的字典变量；
        :param x: 输入数据
        :param t: 标签数据
        :return:
        """
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        # 第一层权重的梯度
        grads['w1'] = numerical_gradient(loss_W, self.params['w1'])
        # 第一层偏置的梯度
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        # 第二层权重的梯度
        grads['w2'] = numerical_gradient(loss_W, self.params['w2'])
        # 第二层偏置的梯度
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads


# 实例化对象
net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print(net.params['w1'].shape)
print(net.params['b1'].shape)
print(net.params['w2'].shape)
print(net.params['b2'].shape)

# 伪输入数据
x = np.random.rand(100, 784)

# 伪正确标签
t = np.random.rand(100, 10)

# 推理
y = net.predict(x)
print(y.shape)

# 梯度计算
grads = net.numerical_gradient(x, t)
print(grads['w1'].shape)
print(grads['b1'].shape)
print(grads['w2'].shape)
print(grads['b2'].shape)

