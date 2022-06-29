"""
组装已实现的层构建神经网络；
"""

import sys, os
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
from dataset.mnist import load_mnist

class Relu(object):
    """ReLu层"""
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid(object):
    """sigmoid层"""
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine(object):
    """affine层"""
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.w) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss(object):
    """输出层的softmax-with-loss层"""
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = soft_max(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


def sigmoid(x):
    """sigmoid函数"""
    return 1/(1 + np.exp(-x))


def numerical_gradient(f, x):
    """遍历数组的各个元素，进行数值微分"""
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
    """softmax函数"""
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def cross_entropy_error(y, t):
    """改进版的交叉熵损失函数"""
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


class TwoLayerNet(object):
    """
    误差反向传播求梯度和梯度确认；
    1: mini-batch
        从训练数据中随机选择一部分数据
    2: compute gradient
        计算损失函数关于各个权重参数的梯度；误差传播；
    3: update parameters
        将权重参数沿梯度方向进行微小的更新
    4: repeat
        重复1、2、3
    """
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        进行初始化；
        params:保存神经网络的参数的字典型变量；
        layers:保存神经网络的层的有序字典型变量；
        :param input_size: 输入层神经元数
        :param hidden_size: 隐藏层神经元数
        :param output_size: 输出层神经元数
        :param weight_init_std: 初始化权重时的高斯分布的规模
        """
        # 初始化权重
        self.params = {}
        # 第一层的权重；
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # 第一层的偏置；
        self.params['b1'] = np.zeros(hidden_size)
        # 第二层的权重；
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # 第二层的偏置；
        self.params['b2'] = np.zeros(output_size)
        # 生成层
        self.layers = OrderedDict()
        # affine1层
        self.layers['Affine1'] = Affine(self.params['w1'], self.params['b1'])
        # relu1层
        self.layers['Relu1'] = Relu()
        # affine2层
        self.layers['Affine2'] = Affine(self.params['w2'], self.params['b2'])
        # 最后一层softmax-with-loss层
        self.lastLayer = SoftmaxWithLoss()


    def predict(self, x):
        """进行推理"""
        # w1, w2 = self.params['w1'], self.params['w2']
        # b1, b2 = self.params['b1'], self.params['b2']
        #
        # a1 = np.dot(x, w1) + b1
        # z1 = sigmoid(a1)
        # a2 = np.dot(z1, w2) + b2
        # y = soft_max(a2)

        # return y
        # 神经网络的正向传播只需要按照添加元素的顺序调用各层的forward()方法就可以完成处理；
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """计算损失函数的值"""
        y = self.predict(x)
        # return cross_entropy_error(y, t)
        return self.lastLayer.forward(y, t)

    def accurary(self, x, t):
        """计算识别精度"""
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        # t = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(y, axis=1)
        accurary = np.sum(y == t)/float(x.shape[0])
        return accurary

    def numerical_gradient(self, x, t):
        """通过数值微分计算关于权重参数的梯度"""
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['w1'] = numerical_gradient(loss_W, self.params['w1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['w2'] = numerical_gradient(loss_W, self.params['w2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradients(self, x, t):
        """通过误差反向传播法计算关于权重参数的梯度"""
        self.loss(x, t)
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        # 神经网络的反向传播，只需要按照相反的顺序调用各层forward()方法即可；
        for layer in layers:
            dout = layer.backward(dout)
        # 设定
        grads = {}
        grads['w1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affine1'].db
        grads['w2'] = self.layers['Affine2'].dw
        grads['b2'] = self.layers['Affine2'].db

        return grads

# gredient check:用数值微分求梯度的结果，验证误差反向传播求梯度的结果；
# 读入数据
(x_train, t_train),(x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# 实例化对象
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
x_batch = x_train[:3]
t_batch = t_train[:3]
# 数值微分求梯度
grad_numerial = network.numerical_gradient(x_batch, t_batch)
# 误差反向传播求梯度
grad_backprop = network.gradients(x_batch, t_batch)
# 求各个权证的绝对误差的平均值：MAE(mean absolute error)
for key in grad_numerial.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerial[key]))
    print(key + ': ' + str(diff))
