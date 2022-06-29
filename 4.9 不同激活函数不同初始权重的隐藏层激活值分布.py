"""
观察权重初始值如何影响隐藏层的激活值的分布的；
一个5层神经网络，激活函数是sigmoid函数；
各层的激活值的分布都要求有适当地广度。
因为通过在各层间传递多样性的数据，神经网络可以进行高效的学习。
如果传递的是有所偏向的数据，就会出现梯度消失或者表现力受限的问题，导致学习可能无法顺利进行；
xavier 初始值：如果前一层的节点数为n，则初始值使用标准差为np.sqrt(n)的分布；
使用xavier初始值，前一层的节点数越多，要设定的目标节点的初始值的权重尺度就越小；
xavier 初始值是以激活函数是线性函数为前提而推导出来的。适合sigmoid， tanh函数；
激活函数ReLu，推荐使用ReLu专用的初始值，He初始值；
He初始值：当前一层的节点为n时，He初始值使用标准差为 1/ np.sqrt(2/n)的高斯分布；
"""
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)

input_data = np.random.randn(1000, 100)  # 一千个数据
node_num = 100  # 隐藏层的神经元数
hidden_layer_size = 5  # 隐藏层数
activations = {}  # 激活值的结果保存在这里

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # 通过改变这个尺寸，观察激活值的分布如何变化
    # 标准差为1的高斯分布
    # 各层的激活值呈偏向0和1的分布；
    # w = np.random.randn(node_num, node_num) * 1

    # 这次呈现在0.5附近的分布。
    # 因为不像刚才的例子那样偏向0和1，所以不会发生梯度消失的问题；
    # 但是，激活值的分布有所偏向，说明在表现力上会有很大问题。
    # 因为如果有多个神经元都输出几乎相同的值，那它们就没有存在的意义了；
    # w = np.random.randn(node_num, node_num) * 0.01

    # 使用xavier初始值，前一层的节点数越多，要设定的目标节点的初始值的权重尺度就越小；
    # 越是后面的层，图像变得越歪斜，但是呈现了比之前更有广度的分布。
    # 因为各层间传递的数据有适当地广度；所以sigmoid函数的表现力不受限制，有望进行高效的学习；
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)

    # 激活函数ReLu，推荐使用ReLu专用的初始值，He初始值；
    # He初始值：当前一层的节点为n时，He初始值使用标准差为 1/ np.sqrt(2/n)的高斯分布；
    w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)


    a = np.dot(x, w)


    # 激活函数
    # sigmoid函数是s型函数，随着输出不断地靠近0（或者1），它的导数的值逐渐接近0；
    # 因此，偏向0和1的数据分布会造成反向传播中梯度的值不断变小，最后消失。
    # 这个问题称为梯度消失gradient vanishing;
    # 层次加深的深度学习中，梯度消失的问题可能会更加严重；
    # z = sigmoid(a)

    z = ReLU(a)
    # 用tanh()代替sigmoid()，这个稍微歪斜的问题就能得到改善，会呈现吊钟型分布；
    # z = tanh(a)

    activations[i] = z

# 绘制直方图
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()

