"""
通过像组装乐高积木一样组装上一节中实现的层，可以构建神经网络。
本节我们将通过组装已经实现的层来构建神经网络。

在进行具体的实现之前，我们再来确认一下神经网络学习的全貌图。
神经网络学习的步骤如下所示。
前提
神经网络中有合适的权重和偏置，调整权重和偏置以便拟合训练数据的过程称为学习。
神经网络的学习分为下面 4 个步骤。

步骤 1
（mini-batch）从训练数据中随机选择一部分数据。
步骤 2
（计算梯度）计算损失函数关于各个权重参数的梯度。
步骤 3
（更新参数）将权重参数沿梯度方向进行微小的更新。
步骤 4
（重复）重复步骤 1、步骤 2、步骤 3。

现在来进行神经网络的实现。
这里我们要把2层神经网络实现为TwoLayerNet。
首先，将这个类的实例变量和方法整理成表 5-1 和表 5-2。

表 5-1 TwoLayerNet类的实例变量
params  保存神经网络的参数的字典型变量。
        params['W1'] 是第 1 层的权重，params['b1'] 是第 1 层的偏置。
        params['W2'] 是第 2 层的权重，params['b2'] 是第 2层的偏置

layers  保存神经网络的层的有序字典型变量。
        以 layers['Affine1']、layers['ReLu1']、layers['Affine2']的形式，通过有序字典保存各个层

lastLayer  神经网络的最后一层。本例中为 SoftmaxWithLoss 层


表 5-2 TwoLayerNet类的方法
__init__(self, input_size, hidden_size, output_size, weight_init_std)   进行初始化。
    参数从头开始依次是输入层的神经元数、隐藏层的神经元数、输出层的神经元数、初始化权重时的高斯分布的规模


predict(self, x)    进行识别（推理）。
    参数 x 是图像数据

loss(self, x, t)    计算损失函数的值。
    参数 X 是图像数据、t 是正确解标签

accuracy(self, x, t)    计算识别精度

numerical_gradient(self, x, t)  通过数值微分计算关于权重参数的梯度（同上一章）

gradient(self, x, t)    通过误差反向传播法计算关于权重参数的梯度

这个类的实现稍微有一点长，但是内容和 4.5 节的学习算法的实现有很多共通的部分，不同点主要在于这里使用了层。
通过使用层，获得识别结果的处理（predict()）和计算梯度的处理（gradient()）只需通过层之间的传递就能完成。
下面是 TwoLayerNet 的代码实现。

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
"""
请注意这个实现中的粗体字代码部分，尤其是将神经网络的层保存为OrderedDict 这一点非常重要。
OrderedDict 是有序字典，“有序”是指它可以记住向字典里添加元素的顺序。
因此，神经网络的正向传播只需按照添加元素的顺序调用各层的 forward() 方法就可以完成处理，
而反向传播只需要按照相反的顺序调用各层即可。
因为 Affine 层和 ReLU 层的内部会正确处理正向传播和反向传播，所以这里要做的事情仅仅是以正确的顺序连接各层，
再按顺序（或者逆序）调用各层。

像这样通过将神经网络的组成元素以层的方式实现，可以轻松地构建神经网络。这个用层进行模块化的实现具有很大优点。
因为想另外构建一个神经网络（比如 5 层、10 层、20 层......的大的神经网络）时，只需像组装乐高积木那样添加必要的层就可以了。
之后，通过各个层内部实现的正向传播和反向传播，就可以正确计算进行识别处理或学习所需的梯度。

到目前为止，我们介绍了两种求梯度的方法。
一种是基于数值微分的方法，另一种是解析性地求解数学式的方法。
后一种方法通过使用误差反向传播法，即使存在大量的参数，也可以高效地计算梯度。
因此，后文将不再使用耗费时间的数值微分，而是使用误差反向传播法求梯度。

数值微分的计算很耗费时间，而且如果有误差反向传播法的（正确的）实现的话，就没有必要使用数值微分的实现了。
那么数值微分有什么用呢？
实际上，在确认误差反向传播法的实现是否正确时，是需要用到数值微分的。


数值微分的优点是实现简单，因此，一般情况下不太容易出错。
而误差反向传播法的实现很复杂，容易出错。
所以，经常会比较数值微分的结果和误差反向传播法的结果，以确认误差反向传播法的实现是否正确。
确认数值微分求出的梯度结果和误差反向传播法求出的结果是否一致（严格地讲，是非常相近）的操作称为梯度确认（gradient check）。
梯度确认的代码实现如下所示（源代码在ch05/gradient_check.py 中）。

"""
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
# 求各个权重的绝对误差的平均值：MAE(mean absolute error)
# 实现正确的话，误差是一个接近0的很小的值；
# 如果这个值很大，说明误差反向传播法的实现存在错误；
for key in grad_numerial.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerial[key]))
    print(key + ': ' + str(diff))
"""
和以前一样，读入 MNIST 数据集。
然后，使用训练数据的一部分，确认数值微分求出的梯度和误差反向传播法求出的梯度的误差。
这里误差的计算方法是求各个权重参数中对应元素的差的绝对值，并计算其平均值。
运行上面的代码后，会输出如下结果。

从这个结果可以看出，通过数值微分和误差反向传播法求出的梯度的差非常小。
比如，第 1 层的偏置的误差是 9.7e-13（0.00000000000097）。
这样一来，我们就知道了通过误差反向传播法求出的梯度是正确的，误差反向传播法的实现没有错误。


数值微分和误差反向传播法的计算结果之间的误差为 0 是很少见的。
这是因为计算机的计算精度有限（比如，32 位浮点数）。
受到数值精度的限制，刚才的误差一般不会为 0，但是如果实现正确的话，可以期待这个误差是一个接近 0 的很小的值。
如果这个值很大，就说明误差反向传播法的实现存在错误。

"""