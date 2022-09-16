"""
最后，我们来看一下使用了误差反向传播法的神经网络的学习的实现。
和之前的实现相比，不同之处仅在于通过误差反向传播法求梯度这一点。
这里只列出了代码，省略了说明（源代码在 ch05/train_neuralnet.py 中）。

"""
import sys, os
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
from dataset.mnist import load_mnist
class Relu(object):
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
    return 1/(1 + np.exp(-x))


def numerical_gradient(f, x):
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
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


class TwoLayerNet(object):

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['w1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['w2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()


    def predict(self, x):
        # w1, w2 = self.params['w1'], self.params['w2']
        # b1, b2 = self.params['b1'], self.params['b2']
        #
        # a1 = np.dot(x, w1) + b1
        # z1 = sigmoid(a1)
        # a2 = np.dot(z1, w2) + b2
        # y = soft_max(a2)

        # return y

        for layer in self.layers.values():
            x = layer.forward(x)
        return x


    def loss(self, x, t):
        y = self.predict(x)
        # return cross_entropy_error(y, t)
        return self.lastLayer.forward(y, t)

    def accurary(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        # t = np.argmax(y, axis=1)
        if t.ndim != 1:
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

    def gradients(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        grads['w1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affine1'].db
        grads['w2'] = self.layers['Affine2'].dw
        grads['b2'] = self.layers['Affine2'].db

        return grads


(x_train, t_train),(x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
iters_num = 100
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01
train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size,1)
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # 通过误差反向传播法求梯度
    grad = network.gradients(x_batch, t_batch)
    # 更新
    for key in ('w1','b1','w2','b2',):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accurary(x_train, t_train)
        test_acc = network.accurary(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

"""
本章我们介绍了将计算过程可视化的计算图，
并使用计算图，介绍了神经网络中的误差反向传播法，
并以层为单位实现了神经网络中的处理。
我们学过的层有ReLU 层、Softmax-with-Loss 层、Affine 层、Softmax 层等，
这些层中实现了 forward 和 backward 方法，
通过将数据正向和反向地传播，
可以高效地计算权重参数的梯度。
通过使用层进行模块化，神经网络中可以自由地组装层，轻松构建出自己喜欢的网络。

本章所学的内容
通过使用计算图，可以直观地把握计算过程。
计算图的节点是由局部计算构成的。局部计算构成全局计算。
计算图的正向传播进行一般的计算。通过计算图的反向传播，可以计算各个节点的导数。
通过将神经网络的组成元素实现为层，可以高效地计算梯度（反向传播法）。
通过比较数值微分和误差反向传播法的结果，可以确认误差反向传播法的实现是否正确（梯度确认）。
"""

















