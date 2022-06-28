"""ReLu函数、激活函数层的实现"""
import numpy as np


class Relu(object):
    """rectified linear unit"""
    def __init__(self):
        """初始化实例变量self.mask,由True\False组成的numpy数组"""
        self.mask = None

    def forward(self, x):
        """<=0,输出0；>0，输出x"""
        self.mask = (x <= 0)  # boolean 数组，符合条件为True
        out = x.copy()  # 拷贝
        out[self.mask] = 0  # 将符合boolean数组中为True的元素，更新为0
        return out

    def backward(self, dout):
        """<=0，输出0；>0，输出1"""
        dout[self.mask] = 0  # 将保存的符合条件的输入数据的boolean数组，元素中为True的元素，更新为0；
        dx = dout
        return dx


class Sigmoid(object):
    """sigmoid层"""
    def __init__(self):
        self.out = None

    def forward(self, x):
        """正向传播"""
        out = 1 / (1 + np.exp(-x))
        self.out = out  # 正向传播时将输出保存在实例变量out中
        return out

    def backward(self, dout):
        """反向传播，会使用正向传播的输出结果"""
        dx = dout * (1.0 - self.out) * self.out
        return dx

x = np.array([[1.0,-0.5],[-2.0,3.0]])
print(x)
mask = (x<=0)
print(mask)

