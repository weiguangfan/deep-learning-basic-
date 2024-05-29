"""
神经网络的推理处理
下面，我们对这个 MNIST 数据集实现神经网络的推理处理。
神经网络的输入层有 784 个神经元，输出层有 10 个神经元。
输入层的 784 这个数字来源于图像大小的 28 × 28 = 784，输出层的 10 这个数字来源于 10 类别分类（数字 0到 9，共 10 类别）。
此外，这个神经网络有 2 个隐藏层，第 1 个隐藏层有 50个神经元，第 2 个隐藏层有 100 个神经元。
这个 50 和 100 可以设置为任何值。

下面我们先定义 get_data()、init_network()、predict() 这 3 个函数（代码在 ch03/neuralnet_mnist.py 中）。

因为之前我们假设学习已经完成，所以学习到的参数被保存下来。
假设保存在 sample_weight.pkl 文件中，在推理阶段，我们直接加载这些已经学习到的参数。

init_network() 会读入保存在 pickle 文件 sample_weight.pkl 中的学习到的权重参数 。
这个文件中以字典变量的形式保存了权重和偏置参数。
剩余的2 个函数，和前面介绍的代码实现基本相同，无需再解释。
现在，我们用这 3 个函数来实现神经网络的推理处理。
然后，评价它的识别精度（accuracy），即能在多大程度上正确分类。

另外，在这个例子中，我们把 load_mnist 函数的参数 normalize 设置成了True。
将 normalize 设置成 True 后，函数内部会进行转换，将图像的各个像素值除以 255，使得数据的值在 0.0～1.0 的范围内。
像这样把数据限定到某个范围内的处理称为正规化（normalization）。
此外，对神经网络的输入数据进行某种既定的转换称为预处理（pre-processing）。
这里，作为对输入图像的一种预处理，我们进行了正规化。

预处理在神经网络（深度学习）中非常实用，其有效性已在提高识别性能和学习的效率等众多实验中得到证明。
在刚才的例子中，作为一种预处理，我们将各个像素值除以 255，进行了简单的正规化。
实际上，很多预处理都会考虑到数据的整体分布。
比如，利用数据整体的均值或标准差，移动数据，使数据整体以 0 为中心分布，
或者进行正规化，把数据的延展控制在一定范围内。
除此之外，还有将数据整体的分布形状均匀化的方法，即数据白化（whitening）等。

"""
import pickle
import numpy as np
from dataset.mnist import load_mnist


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def soft_max(a):
    """进化的防止溢出的激活函数"""
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def get_data():
    """获取数据"""
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    """读入保存的pickle文件"""
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

x, t = get_data()  # t输出真实类别标签的数组
print(x.shape)
network = init_network()
print(network.keys())


accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])  # 以数组形式输出各个标签对应的概率
    if i == 6:
        print(y.shape)
        print(y)
    p = np.argmax(y)  # 取出概率列表最大值的索引
    if p == t[i]:
        accuracy_cnt += 1
print("accuracy_cnt: " + str(float(accuracy_cnt / len(x))))

"""
首先获得 MNIST 数据集，生成网络。
接着，用 for 语句逐一取出保存在 x中的图像数据，用 predict() 函数进行分类。
predict() 函数以 NumPy 数组的形式输出各个标签对应的概率。
比如输出 [0.1, 0.3, 0.2, ..., 0.04] 的数组，该数组表示“0”的概率为 0.1，“1”的概率为 0.3，等等。
然后，我们取出这个概率列表中的最大值的索引（第几个元素的概率最高），作为预测结果。
可以用 np.argmax(x) 函数取出数组中的最大值的索引，np.argmax(x) 将获取被赋给参数 x 的数组中的最大值元素的索引。
最后，比较神经网络所预测的答案和正确解标签，将回答正确的概率作为识别精度。

执行上面的代码后，会显示“Accuracy:0.9352”。这表示有 93.52 % 的数据被正确分类了。
目前我们的目标是运行学习到的神经网络，所以不讨论识别精度本身，
不过以后我们会花精力在神经网络的结构和学习方法上，思考如何进一步提高这个精度。
实际上，我们打算把精度提高到 99 % 以上。

"""
