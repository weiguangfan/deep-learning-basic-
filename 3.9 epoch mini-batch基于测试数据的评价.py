"""
根据图 4-11 呈现的结果，我们确认了通过反复学习可以使损失函数的值逐渐减小这一事实。
不过这个损失函数的值，严格地讲是“对训练数据的某个 mini-batch 的损失函数”的值。
训练数据的损失函数值减小，虽说是神经网络的学习正常进行的一个信号，
但光看这个结果还不能说明该神经网络在其他数据集上也一定能有同等程度的表现。

神经网络的学习中，必须确认是否能够正确识别训练数据以外的其他数据，即确认是否会发生过拟合。
过拟合是指，虽然训练数据中的数字图像能被正确辨别，但是不在训练数据中的数字图像却无法被识别的现象。

神经网络学习的最初目标是掌握泛化能力，
因此，要评价神经网络的泛化能力，就必须使用不包含在训练数据中的数据。
下面的代码在进行学习的过程中，会定期地对训练数据和测试数据记录识别精度。
这里，每经过一个 epoch，我们都会记录下训练数据和测试数据的识别精度。

epoch 是一个单位。
一个 epoch 表示学习中所有训练数据均被使用过一次时的更新次数。
比如，对于 10000 笔训练数据，用大小为 100 笔数据的 mini-batch 进行学习时，
重复随机梯度下降法 100 次，所有的训练数据就都被“看过”了 。
此时，100 次就是一个 epoch。

实际上，一般做法是事先将所有训练数据随机打乱，
然后按指定的批次大小，按序生成 mini-batch。
这样每个 mini-batch 均有一个索引号，比如此例可以是 0, 1, 2, ... , 99，然后用索引号可以遍历所有的 mini-batch。
遍历一次所有数据，就称为一个epoch。
请注意，本节中的mini-batch 每次都是随机选择的，所以不一定每个数据都会被看到。

为了正确进行评价，我们来稍稍修改一下前面的代码。
与前面的代码不同的地方，我们用粗体来表示。

"""
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

def sigmoid(x):
    """sigmoid激活函数"""
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
    """输出层的softmax激活函数"""
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def cross_entropy_error(y, t):
    """单个数据的损失函数"""
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

# def cross_entropy_error(y, t):
#     delta = 1e-7
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
#     # 抽出正确标签对应的输出值
#     batch_size = y.shape[0]
#     print('batch_size: ', batch_size)
#     return -np.sum(np.log(y[np.arange(batch_size), t] + delta))/batch_size

class TwoLayerNet(object):
    """epoch mini-batch SGD"""
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.0):
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = soft_max(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accurary(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
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

# 获取mini-batch
(x_train, t_train), (x_test, t_test), = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 超参数
iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

iter_per_epoch = max(train_size/batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
# 这个是按照mini-batch，完成一个epoch;
# for i in range(0, train_size, batch_size):
#     x_batch = x_train[i:i+batch_size]
#     t_batch = t_train[i:i+batch_size]

# 重复一千次，每次取mini-batch数据
for i in range(iters_num):
    # 随机选取的mini-batch，不是严格的epoch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)

    # 更新参数
    for key in ('w1', 'b1', 'w2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 计算损失
    loss = network.loss(x_batch, t_batch)

    # 记录学习过程
    train_loss_list.append(loss)

    # 每经历一次epoch，计算一次精确度，记录一次学习过程；
    if i % iter_per_epoch == 0:
        train_acc = network.accurary(x_train, t_train)
        test_acc = network.accurary(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc) )

print("train_loss_list: ", train_loss_list)
"""
在上面的例子中，每经过一个 epoch，就对所有的训练数据和测试数据计算识别精度，并记录结果。
之所以要计算每一个 epoch 的识别精度，是因为如果在for 语句的循环中一直计算识别精度，会花费太多时间。
并且，也没有必要那么频繁地记录识别精度（只要从大方向上大致把握识别精度的推移就可以了）。
因此，我们才会每经过一个 epoch 就记录一次训练数据的识别精度。

把从上面的代码中得到的结果用图表示的话，如图 4-12 所示。

图 4-12　训练数据和测试数据的识别精度的推移（横轴的单位是 epoch）

图 4-12 中，实线表示训练数据的识别精度，虚线表示测试数据的识别精度。
如图所示，随着 epoch 的前进（学习的进行），我们发现使用训练数据和测试数据评价的识别精度都提高了，
并且，这两个识别精度基本上没有差异（两条线基本重叠在一起）。
因此，可以说这次的学习中没有发生过拟合的现象。

"""



# 绘图观察训练集和测试集的精确度；
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
"""
本章中，我们介绍了神经网络的学习。
首先，为了能顺利进行神经网络的学习，我们导入了损失函数这个指标。
以这个损失函数为基准，找出使它的值达到最小的权重参数，就是神经网络学习的目标。
为了找到尽可能小的损失函数值，我们介绍了使用函数斜率的梯度法。

本章所学的内容
机器学习中使用的数据集分为训练数据和测试数据。
神经网络用训练数据进行学习，并用测试数据评价学习到的模型的泛化能力。
神经网络的学习以损失函数为指标，更新权重参数，以使损失函数的值减小。
利用某个给定的微小值的差分求导数的过程，称为数值微分。
利用数值微分，可以计算权重参数的梯度。
数值微分虽然费时间，但是实现起来很简单。

下一章中要实现的稍微复杂一些的误差反向传播法可以高速地计算梯度。


"""
