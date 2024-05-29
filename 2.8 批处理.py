"""
以上就是处理 MNIST 数据集的神经网络的实现，现在我们来关注输入数据和权重参数的“形状”。
再看一下刚才的代码实现。

下面我们使用 Python 解释器，输出刚才的神经网络的各层的权重的形状。
"""

import pickle
import numpy as np
from dataset.mnist import load_mnist

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def soft_max(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = soft_max(a3)
    return y

# 输入为一张图像时的处理流程；
# x, _ = get_data()
# network = init_network()
# w1, w2, w3 = network['W1'], network['W2'], network['W3']
# print(x.shape)  # (10000, 784)
# print(x[0].shape)  # (784,)
# print(w1.shape)  # (784, 50)
# print(w2.shape)  # (50, 100)
# print(w3.shape)  # (100, 10)
"""
图 3-26　数组形状的变化
我们通过上述结果来确认一下多维数组的对应维度的元素个数是否一致（省略了偏置）。
用图表示的话，如图 3-26 所示。
可以发现，多维数组的对应维度的元素个数确实是一致的。
此外，我们还可以确认最终的结果是输出了元素个数为 10的一维数组。

从整体的处理流程来看，
图 3-26 中，输入一个由 784 个元素（原本是一个28 × 28 的二维数组）构成的一维数组后，
输出一个有 10 个元素的一维数组。
这是只输入一张图像数据时的处理流程。


现在我们来考虑打包输入多张图像的情形。
比如，我们想用 predict() 函数一次性打包处理 100 张图像。
为此，可以把  的形状改为 100 × 784，将 100张图像打包作为输入数据。
用图表示的话，如图 3-27 所示。
图 3-27　批处理中数组形状的变化

如图 3-27 所示，输入数据的形状为 100 × 784，输出数据的形状为 100 ×10。
这表示输入的 100 张图像的结果被一次性输出了。
比如，x[0] 和 y[0] 中保存了第 0 张图像及其推理结果，x[1] 和 y[1] 中保存了第 1 张图像及其推理结果，等等。

这种打包式的输入数据称为批（batch）。
批有“捆”的意思，图像就如同纸币一样扎成一捆。

批处理对计算机的运算大有利处，可以大幅缩短每张图像的处理时间。
那么为什么批处理可以缩短处理时间呢？这是因为大多数处理数值计算的库都进行了能够高效处理大型数组运算的最优化。
并且，在神经网络的运算中，当数据传送成为瓶颈时，
批处理可以减轻数据总线的负荷（严格地讲，相对于数据读入，可以将更多的时间用在计算上）。
也就是说，批处理一次性计算大型数组要比分开逐步计算各个小型数组速度更快。

下面我们进行基于批处理的代码实现。
这里用粗体显示与之前的实现的不同之处。




"""
# 输入多张图像的处理流程
# x, t = get_data()
# network = init_network()
# batch_size = 100  # 批数量
# accuracy_cnt = 0
# for i in range(0, len(x), batch_size):  # 步长为batch_size
#     x_batch = x[i:i+batch_size]  # 切片：取100个数，抽取批数据
#     y_batch = predict(network, x_batch)
#     p = np.argmax(y_batch, axis=1)
#     accuracy_cnt += np.sum(p == t[i:i+batch_size])
# print(accuracy_cnt)
"""
我们来逐个解释粗体的代码部分。
首先是 range() 函数。
range() 函数若指定为 range(start, end)，则会生成一个由 start 到 end-1 之间的整数构成的列表。
若像 range(start, end, step) 这样指定 3 个整数，则生成的列表中的下一个元素会增加 step 指定的值。
我们来看一个例子。
"""

# range(start,end,stride)函数
# print(list(range(0, 10)))
# print(list(range(0, 10, 3)))
"""
在 range() 函数生成的列表的基础上，通过 x[i:i+batch_size] 从输入数据中抽出批数据。
x[i:i+batch_n] 会取出从第 i 个到第 i+batch_n 个之间的数据。
本例中是像 x[0:100]、x[100:200]......这样，从头开始以 100 为单位将数据提取为批数据。

然后，通过 argmax() 获取值最大的元素的索引。
不过这里需要注意的是，我们给定了参数 axis=1。
矩阵的第 0 维是列方向，第 1 维是行方向。
这指定了在 100 × 10 的数组中，沿着第 1 维方向（以第 1 维为轴）找到值最大的元素的索引（第 0 维对应第 1 个维度）。
这里也来看一个例子。
"""
# 获取数组值最大的索引，按照不同的轴
# x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
# print(x)
# y = np.argmax(x, axis=1)
# print(y)
"""
最后，我们比较一下以批为单位进行分类的结果和实际的答案。
为此，需要在NumPy 数组之间使用比较运算符（==）生成由 True/False 构成的布尔型数组，并计算 True 的个数。
我们通过下面的例子进行确认。

"""
y = np.array([1, 2, 1, 0])
t = np.array([1, 2, 0, 0])
print(y == t)
print(np.sum(y == t))
"""
至此，基于批处理的代码实现就介绍完了。
使用批处理，可以实现高速且高效的运算。
下一章介绍神经网络的学习时，我们将把图像数据作为打包的批数据进行学习，届时也将进行和这里的批处理一样的代码实现。

本章介绍了神经网络的前向传播。
本章介绍的神经网络和上一章的感知机在信号的按层传递这一点上是相同的，
但是，向下一个神经元发送信号时，改变信号的激活函数有很大差异。
神经网络中使用的是平滑变化的 sigmoid 函数，而感知机中使用的是信号急剧变化的阶跃函数。
这个差异对于神经网络的学习非常重要，我们将在下一章介绍。

本章所学的内容
神经网络中的激活函数使用平滑变化的 sigmoid 函数或ReLU函数。
通过巧妙地使用NumPy多维数组，可以高效地实现神经网络。
机器学习的问题大体上可以分为回归问题和分类问题。
关于输出层的激活函数，回归问题中一般用恒等函数，分类问题中一般用softmax 函数。
分类问题中，输出层的神经元的数量设置为要分类的类别数。
输入数据的集合称为批。通过以批为单位进行推理处理，能够实现高速的运算。
"""
