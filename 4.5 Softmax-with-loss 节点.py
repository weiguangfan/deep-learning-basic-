"""
神经网络最后一层affine层，推理阶段不需要softmax的正规化，训练阶段需要softmax的正规化；
softmax-with-loss层:计算图中，softmax层 + cross-entropy-error层
"""

import numpy as np


def soft_max(a):
    """改进版的softmax"""
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def cross_entropy_error(y, t):
    """交叉熵损失函数"""
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))



class SoftmaxWithLoss(object):
    """
    softmax-with-loss层的实现；
    反向传播中，输入的导数，等于输出与标签的差值；
    """

    def __init__(self):
        self.loss = None  # 损失
        self.y = None  # softmax输出
        self.t = None  # 监督数据

    def forward(self, x, t):
        """前向传播"""
        self.t = t
        self.y = soft_max(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        """反向传播"""
        batch_size = self.t.shape[0]
        #  将要传播的值除以批的大小后，传递给前面的层的是单个数据误差
        dx = (self.y - self.t) / batch_size
        return dx

