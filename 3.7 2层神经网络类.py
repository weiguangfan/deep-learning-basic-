"""
关于神经网络学习的基础知识，到这里就全部介绍完了。
“损失函数”“mini-batch”“梯度”“梯度下降法”等关键词已经陆续登场，这里我们来确认一下神经网络的学习步骤，顺便复习一下这些内容。
神经网络的学习步骤如下所示。

前提神经网络存在合适的权重和偏置，调整权重和偏置以便拟合训练数据的过程称为“学习”。
神经网络的学习分成下面 4 个步骤。

1:min-batch
    从训练数据中随机选出一部分数据，这一部分数据称为mini-batch。
    目标是减小mini-batch的损失函数的值。
2:compute gradient
    为了减小mini-batch的损失函数的值，需要求出各个权重参数的梯度。
    梯度表示损失函数值减小最多的方向。
3:update parameters
    将权重参数沿梯度方向进行微小更新。
4:repeat
    重复1、2、3

神经网络的学习按照上面 4 个步骤进行。
这个方法通过梯度下降法更新参数，
不过因为这里使用的数据是随机选择的mini batch 数据，
所以又称为随机梯度下降法（stochastic gradient descent）。
“随机”指的是“随机选择的”的意思，因此，随机梯度下降法是“对随机选择的数据进行的梯度下降法”。
深度学习的很多框架中，随机梯度下降法一般由一个名为 SGD 的函数来实现。
SGD 来源于随机梯度下降法的英文名称的首字母。

下面，我们来实现手写数字识别的神经网络。
这里以 2 层神经网络（隐藏层为1 层的网络）为对象，使用 MNIST 数据集进行学习。

首先，我们将这个 2 层神经网络实现为一个名为 TwoLayerNet 的类，实现过程如下所示 。
源代码在 ch04/two_layer_net.py 中。

"""

import sys, os
sys.path.append(os.pardir)
import numpy as np


def sigmoid(x):
    """sigmoid激活函数"""
    return 1/(1 + np.exp(-x))


def numerical_gradient(f, x):
    '''遍历数组的每一个元素，进行数值微分'''
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
    """改进版的输出层的激活函数"""
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def cross_entropy_error(y, t):
    """改进版的交叉熵损失"""
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

"""
TwoLayerNet 的实现参考了斯坦福大学 CS231n 课程提供的 Python 源代码。

"""




class TwoLayerNet(object):
    """2层神经网络，1层隐藏层"""
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.0):
        """
        初始化权重；
        params:保存神经网络的参数的字典变量；
        :param input_size: 输入层的神经元数
        :param hidden_size: 隐藏层的神经元数
        :param output_size: 输出层的神经元数
        :param weight_init_std:
        """
        self.params = {}
        # 第一层的权重，高斯分布初始化
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # 第一层的偏置，0进行初始化
        self.params['b1'] = np.zeros(hidden_size)
        # 第二层的权重
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # 第二层的偏置
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        """进行推理"""
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = soft_max(a2)

        return y

    def loss(self, x, t):
        """
        损失函数
        :param x: 输入数据
        :param t: 标签数据
        :return:
        """
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accurary(self, x, t):
        """
        识别精度；
        :param x:
        :param t: 标签数据
        :return:
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(y, axis=1)
        accurary = np.sum(y == t)/float(x.shape[0])
        return accurary

    def numerical_gradient(self, x, t):
        """
        根据数值微分，计算各个权重、偏置参数相对于损失函数的梯度；
        grads:保存梯度的字典变量；
        :param x: 输入数据
        :param t: 标签数据
        :return:
        """
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        # 第一层权重的梯度
        grads['w1'] = numerical_gradient(loss_W, self.params['w1'])
        # 第一层偏置的梯度
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        # 第二层权重的梯度
        grads['w2'] = numerical_gradient(loss_W, self.params['w2'])
        # 第二层偏置的梯度
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads
"""
虽然这个类的实现稍微有点长，但是因为和上一章的神经网络的前向处理的实现有许多共通之处，所以并没有太多新东西。
我们先把这个类中用到的变量和方法整理一下。
表 4-1 中只罗列了重要的变量，表 4-2 中则罗列了所有的方法。

表 4-1 TwolayerNet类中使用的变量
params  保存神经网络的参数的字典型变量（实例变量）。
        params['W1'] 是第 1 层的权重，params['b1'] 是第 1 层的偏置。
        params['W2'] 是第 2 层的权重，params['b2'] 是第 2 层的偏置
        
grads   保存梯度的字典型变量（numerical_gradient() 方法的返回值）。       
        grads['W1'] 是第 1 层权重的梯度，grads['b1'] 是第 1 层偏置的梯度。
        grads['W2'] 是第 2 层权重的梯度，grads['b2'] 是第 2 层偏置的梯度

表 4-2 TwoLayerNet类的方法
__init__(self, input_size, hidden_size, output_size)    进行初始化。
    参数从头开始依次表示输入层的神经元数、隐藏层的神经元数、输出层的神经元数

predict(self, x)    进行识别（推理）。
    参数 x 是图像数据

loss(self, x, t)    计算损失函数的值。
    参数 x 是图像数据，t 是正确解标签（后面 3 个方法的参数也一样）
    
accuracy(self, x, t)       计算识别精度
 
numerical_gradient(self, x, t)  计算权重参数的梯度

gradient(self, x, t)    计算权重参数的梯度。
    
numerical_gradient() 的高速版，将在下一章实现


TwoLayerNet 类有 params 和 grads 两个字典型实例变量。
params 变量中保存了权重参数，比如 params['W1'] 以 NumPy 数组的形式保存了第 1 层的权重参数。
此外，第 1 层的偏置可以通过 param['b1'] 进行访问。
这里来看一个例子。

"""

# 实例化对象
net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print(net.params['w1'].shape)
print(net.params['b1'].shape)
print(net.params['w2'].shape)
print(net.params['b2'].shape)
"""
如上所示，params 变量中保存了该神经网络所需的全部参数。
并且，params变量中保存的权重参数会用在推理处理（前向处理）中。
顺便说一下，推理处理的实现如下所示。

"""
# 伪输入数据
x = np.random.rand(100, 784)
y = net.predict(x)

"""
此外，与 params 变量对应，grads 变量中保存了各个参数的梯度。
如下所示，使用 numerical_gradient() 方法计算梯度后，梯度的信息将保存在 grads变量中。

"""
# 伪输入数据
x = np.random.rand(100, 784)

# 伪正确标签
t = np.random.rand(100, 10)

# 梯度计算
grads = net.numerical_gradient(x, t)
print(grads['w1'].shape)
print(grads['b1'].shape)
print(grads['w2'].shape)
print(grads['b2'].shape)

"""
接着，我们来看一下 TwoLayerNet 的方法的实现。
首先是 __init__(self,input_size, hidden_size, output_size) 方法，
它是类的初始化方法（所谓初始化方法，就是生成 TwoLayerNet 实例时被调用的方法）。
从第 1 个参数开始，依次表示输入层的神经元数、隐藏层的神经元数、输出层的神经元数。
另外，因为进行手写数字识别时，输入图像的大小是 784（28 × 28），输出为 10 个类别，
所以指定参数 input_size=784、output_size=10，
将隐藏层的个数hidden_size 设置为一个合适的值即可。

此外，这个初始化方法会对权重参数进行初始化。
如何设置权重参数的初始值这个问题是关系到神经网络能否成功学习的重要问题。
后面我们会详细讨论权重参数的初始化，这里只需要知道，权重使用符合高斯分布的随机数进行初始化，偏置使用 0 进行初始化。
predict(self, x) 和 accuracy(self, x, t) 的实现和上一章的神经网络的推理处理基本一样。
如果仍有不明白的地方，请再回顾一下上一章的内容。
另外，loss(self, x, t) 是计算损失函数值的方法。
这个方法会基于 predict() 的结果和正确解标签，计算交叉熵误差。

剩下的 numerical_gradient(self, x, t) 方法会计算各个参数的梯度。
根据数值微分，计算各个参数相对于损失函数的梯度。
另外，gradient(self, x,t) 是下一章要实现的方法，该方法使用误差反向传播法高效地计算梯度。

numerical_gradient(self, x, t) 基于数值微分计算参数的梯度。
下一章，我们会介绍一个高速计算梯度的方法，称为误差反向传播法。
用误差反向传播法求到的梯度和数值微分的结果基本一致，但可以高速地进行处理。
使用误差反向传播法计算梯度的 gradient(self, x, t) 方法会在下一章实现，
不过考虑到神经网络的学习比较花时间，
想节约学习时间的读者可以替换掉这里的 numerical_gradient(self, x, t)，
抢先使用gradient(self, x, t) ！


"""