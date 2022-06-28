"""

"""

import sys, os
sys.path.append(os.pardir)
import numpy as np
from  dataset.mnist import load_mnist


def sigmoid(x):
    """sigmoid激活函数"""
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
    """输出层的softmax激活函数"""
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
    """2层网络的实现"""
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.0):
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = soft_max(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accurary(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(y, axis=1)
        accurary = np.sum(y == t)/float(x.shape[0])
        return accurary

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['w1'] = numerical_gradient(loss_W, self.params['w1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['w2'] = numerical_gradient(loss_W, self.params['w2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads


# 获取mini-=batch
# 加载数据
(x_train, t_train), (x_test, t_test), = load_mnist(normalize=True, one_hot_label=True)
train_loss_list = []

# 超参数
# 最大迭代次数
iters_num = 10000
# 输入层神经元
train_size = x_train.shape[0]
# 批数量
batch_size = 100
# 学习率
learning_rate = 0.1
# 初始化实例对象
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 计算梯度
# 随机梯度下降
for i in range(iters_num):
    # 随机选择样本，返回样本索引的数组
    batch_mask = np.random.choice(train_size, batch_size)
    # 输入批数据
    x_batch = x_train[batch_mask]
    # 标签批数据
    t_batch = t_train[batch_mask]
    # 计算各个参数相对于损失函数的梯度
    grad = network.numerical_gradient(x_batch, t_batch)

    # 更新参数
    # 更新权重和偏置参数，并保存到字典network.params
    for key in ('w1', 'b1', 'w2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 记录学习过程
    # 计算批数据的总损失
    loss = network.loss(x_batch, t_batch)
    # 每批数据的损失，均添加到列表
    train_loss_list.append(loss)

print("train_loss_list: ", train_loss_list)

