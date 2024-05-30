"""
神经网络的学习的实现使用的是前面介绍过的 mini-batch 学习。
所谓mini-batch 学习，就是从训练数据中随机选择一部分数据（称为 mini-batch），
再以这些 mini-batch 为对象，使用梯度法更新参数的过程。
下面，我们就以 TwoLayerNet 类为对象，使用 MNIST 数据集进行学习（源代码在ch04/train_neuralnet.py 中）。

"""

import os
import sys

sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist


def sigmoid(x):
    """sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))


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
        grad[idx] = (fxh1 - fxh2) / (2 * h)
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
        accurary = np.sum(y == t) / float(x.shape[0])
        return accurary

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['w1'] = numerical_gradient(loss_W, self.params['w1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['w2'] = numerical_gradient(loss_W, self.params['w2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads


"""
这里，mini-batch 的大小为 100，需要每次从 60000 个训练数据中随机取出 100 个数据（图像数据和正确解标签数据）。
然后，对这个包含 100 笔数据的mini-batch 求梯度，使用随机梯度下降法（SGD）更新参数。
这里，梯度法的更新次数（循环的次数）为 10000。
每更新一次，都对训练数据计算损失函数的值，并把该值添加到数组中。
用图像来表示这个损失函数的值的推移，如图 4-11 所示。

图 4-11　损失函数的推移：左图是 10000 次循环的推移，右图是 1000 次循环的推移

观察图 4-11，可以发现随着学习的进行，损失函数的值在不断减小。
这是学习正常进行的信号，表示神经网络的权重参数在逐渐拟合数据。
也就是说，神经网络的确在学习！通过反复地向它浇灌（输入）数据，神经网络正在逐渐向最优参数靠近。

"""

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
