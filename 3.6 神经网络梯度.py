"""
神经网络的学习也要求梯度。
这里所说的梯度是指损失函数关于权重参数的梯度。
比如，有一个只有一个形状为 2 × 3 的权重 W 的神经网络，损失函数用 L表示。
此时，梯度可以用 e_L/e_W 表示。
用数学式表示的话，如下所示。

W = w11  w12  w13
    w21  w22  w23

e_L/e_W = e_L/e_w11  e_L/e_w12  e_L/e_w13
          e_L/e_w21  e_L/e_w22  e_L/e_w23

e_L/e_W 的元素由各个元素关于 W 的偏导数构成。
比如，第 1 行第 1 列的元素 e_L/e_w11 表示当 w11 稍微变化时，损失函数 L 会发生多大变化。
这里的重点是，e_L/e_W 的形状和 W 相同。
实际上，式（4.8）中的 e_L/e_W 和 W 都是 2 × 3 的形状。

下面，我们以一个简单的神经网络为例，来实现求梯度的代码。
为此，我们要实现一个名为 simpleNet 的类（源代码在 ch04/gradient_simplenet.py中）。

"""

import os
import sys

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

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        print('grad[idx]: ', grad[idx])

        x[idx] = tmp_val
        it.iternext()

    return grad


"""
这里使用了 common/functions.py 中的 softmax 和cross_entropy_error 方法，
以及 common/gradient.py 中的numerical_gradient 方法。
simpleNet 类只有一个实例变量，即形状为 2×3的权重参数。
它有两个方法，一个是用于预测的 predict(x)，另一个是用于求损失函数值的 loss(x,t)。
这里参数 x 接收输入数据，t 接收正确解标签。
现在我们来试着用一下这个 simpleNet。

"""


class simpleNet(object):
    """定义一个简单的神经网络，求梯度"""

    def __init__(self):
        """定义一个权重参数self.w"""
        self.w = np.random.randn(2, 3)  # 用高斯分布进行初始化

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
print("net.w: ", net.w)  # 权重参数
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

"""
接下来求梯度。
和前面一样，我们使用 numerical_gradient(f, x) 求梯度（这里定义的函数 f(W) 的参数 W 是一个伪参数。
因为numerical_gradient(f, x) 会在内部执行 f(x)，为了与之兼容而定义了f(W)）。

"""


# 求梯度
f = lambda w:net.loss(x, t)
# def f(W):
#     """参数w是一个伪参数"""
#     return net.loss(x, t)


"""
numerical_gradient(f, x) 的参数 f 是函数，x 是传给函数 f 的参数。
因此，这里参数 x 取 net.W，并定义一个计算损失函数的新函数 f，然后把这个新定义的函数传递给 numerical_gradient(f, x)。

numerical_gradient(f, net.W) 的结果是 dW，一个形状为 2 × 3 的二维数组。
观察一下 dW 的内容，例如，会发现 e_L/e_W 中的 e_L/e_w11 的值大约是 0.2，这表示如果将 w11 增加 h，那么损失函数的值会增加 0.2h。
再如，w23 对应的值大约是 -0.5，这表示如果将 w23 增加 h，损失函数的值将减小 0.5h。
因此，从减小损失函数值的观点来看，w23 应向正方向更新，w11 应向负方向更新。
至于更新的程度，w23 比 w11 的贡献要大。
"""
# 求含有参数w的损失函数的偏导
# net.w传入函数f
dw = numerical_gradient(f, net.w)
print(dw)

"""
另外，在上面的代码中，定义新函数时使用了“def f(x):...”的形式。
实际上，Python 中如果定义的是简单的函数，可以使用 lambda 表示法。
使用lambda 的情况下，上述代码可以如下实现。

求出神经网络的梯度后，接下来只需根据梯度法，更新权重参数即可。
在下一节中，我们会以 2 层神经网络为例，实现整个学习过程。
为了对应形状为多维数组的权重参数 W，这里使用的numerical_gradient() 和之前的实现稍有不同。
不过，改动只是为了对应多维数组，所以改动并不大。
这里省略了对代码的说明，想知道细节的读者请参考源代码（common/gradient.py）。
"""

# f = lambda w: net.loss(x, t)

# dw = numerical_gradient(f, net.w)
