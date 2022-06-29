"""
神经网络的正向传播中，为了计算加权信号和，使用了矩阵的乘积运算（numpy 中是np.dot()）；

"""
# 单个数据对象的affine层
# 计算图各个节点流动的是标量
import numpy as np
# x = np.random.rand(2)
# w = np.random.rand(2,3)
# b = np.random.rand(3)

# print(x.shape)
# print(w.shape)
# print(b.shape)

# y = np.dot(x, w) + b
# print(y)
# print(y.shape)

# 批版本的affine层
# 计算图中各个节点流动是矩阵
# 求导变成了矩阵求导
x_dot_w = np.array([[0,0,0],[10,10,10]])
print(x_dot_w)
print(x_dot_w.shape)
b = np.array([1,2,3])
print(b.shape)
print(b)
# 正向传播时，偏置被加到x.w的各个数据上
y = x_dot_w + b
print(y)
print(y.shape)
# 根据求导公式，
dy = np.array([[1,2,3],[4,5,6]])
print(dy.shape)
print(dy)
# 反向传播时，各个数据的反向传播的值需要汇总为偏置的元素
db = np.sum(dy, axis=0)
print(db.shape)
print(db)

class Affine(object):
    """矩阵乘积运算，改写为affine层，affine的实现"""
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
