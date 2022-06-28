"""
神经网络的学习也要求梯度；
梯度是指损失函数关于权重参数的梯度；
梯度数组的元素由各个元素关于W的偏导数构成；
"""

import sys, os
import numpy as np
sys.path.append(os.pardir)


def cross_entropy_error(y, t):
    """单个数据的交叉熵损失函数"""
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def soft_max(a):
    """改进版的输出层softmax激活函数"""
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def numerical_gradient(f, x):
    """遍历数组中各个元素，进行数值微分"""
    print("x: ", x)
    h = 1e-4
    grad = np.zeros_like(x)
    print("grad: ", grad)
    print("grad.shape: ", grad.shape)
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
    print("it: ", it)
    while not it.finished:
        idx = it.multi_index
        print("idx: ", idx)
        tmp_val = x[idx]
        print('tmp_val: ', tmp_val)
        x[idx] = tmp_val + h
        print('++ x[idx]: : ', x[idx])
        fxh1 = f(x)

        x[idx] = tmp_val - h
        print('-- x[idx]: : ', x[idx])
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        print('grad[idx]: ', grad[idx])
        x[idx] = tmp_val
        it.iternext()

    return grad


class simpleNet(object):
    """定义一个简单的神经网络，求梯度"""
    def __init__(self):
        """定义一个权重参数self.w"""
        self.w = np.random.randn(2, 3)

    def predict(self, x):
        """进行预测"""
        print("self.w: ", self.w)
        print("predict input x: ", x)
        return np.dot(x, self.w)

    def loss(self, x, t):
        """求损失"""
        print("loss_input x: ", x)
        print("loss_input t: ", t)
        z = self.predict(x)
        print("z: ", z)
        print("z.shape: ", z.shape)
        y = soft_max(z)
        print("y: ", y)
        print("y.shape: ", y.shape)
        loss = cross_entropy_error(y, t)
        print("loss: ", loss)
        return loss

# 实例化对象
net = simpleNet()
print("net.w: ", net.w)
print("net.w.shape: ", net.w.shape)

# 输入数据
x = np.array([0.6, 0.9])
print("x: ", x)
print("x.shape: ", x.shape)

# 输出值
p = net.predict(x)
print(p)
print(p.shape)
# 返回数组各元素最大值的下标
print(np.argmax(p))

# 标签数据
t = np.array([0, 0, 1])
print("t: ", t)
print("t.shape: ", t.shape)

# 损失值
print(net.loss(x, t))


# 求梯度
# f = lambda w:net.loss(x, t)
def f(W):
    """参数w是一个伪参数"""
    return net.loss(x, t)


# 求含有参数w的损失函数的偏导
# net.w传入函数f
dw = numerical_gradient(f, net.w)
print(dw)

