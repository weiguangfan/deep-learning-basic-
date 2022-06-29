"""
机器学习的问题中，过拟合是一个很常见的问题，过拟合指的是只能拟合训练数据，但不能很好地拟合不包含在训练数据中的其他数据的装填。
机器学习的目标是提高泛化能力，即便是没有包含在训练数据里的未观测数据，也希望模型可以进行正确地识别；
发生过拟合的原因：
    1 模型拥有大量参数，变现力强
    2 训练数据少
制造过拟合现象：
    数据集60000，只选300个，7层网络（每层100个神经元，激活函数ReLu)

不使用权重衰减：
按epoch分别算出所有训练集和测试集的识别精度；
过了100个epoch后，用训练数据测量到的识别精度几乎都为100%；
但是，测试数据，离100%的识别精度还有较大的差距。
如此大的识别精度差距，是只拟合了训练数据的结果。
模型对训练时没有使用的一般数据拟合得不是很好。

权值衰减是一直以来经常被使用的一种抑制过拟合的方法。
该方法通过在学习的过程中对大的权重进行惩罚，来抑制过拟合。
很多过拟合原本就是因为权重参数取值过大才发生的。
为损失函数加上权重的平方范数（L2范数）。这样一来，就可以抑制权重变大。
超参数lambda,lambda设置得越大，对大的权重施加惩罚就越重。
权重为w,L2范数为 1/2 * lambda * w**2, 将这个式子加到损失函数上。
L2范数相当于各个元素的平方和。
L1范数相当于各个元素的绝对值之和。
L inf 也称Max范数，相当于各个元素的绝对值最大的那一个。

使用权重衰减：
虽然训练数据的识别精度和测试数据的识别精度之间有差距，但是与没有使用权重值衰减的结果相比，差距变小了。
说明过拟合受到了抑制。


"""
# coding: utf-8
import os
import sys

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 再现过拟合，减少谢谢数据
x_train = x_train[:300]
t_train = t_train[:300]

# 权重衰减，惩罚系数
# 按epoch分别算出所有训练集和测试集的识别精度；
# 过了100个epoch后，用训练数据测量到的识别精度几乎都为100%；
# 但是，测试数据，离100%的识别精度还有较大的差距。
# 如此大的识别精度差距，是只拟合了训练数据的结果。
# 模型对训练时没有使用的一般数据拟合得不是很好。
# weight_decay_lambda = 0 # 权重不衰减


# 虽然训练数据的识别精度和测试数据的识别精度之间有差距，但是与没有使用权重值衰减的结果相比，差距变小了。
# 说明过拟合受到了抑制。
weight_decay_lambda = 0.1 # 惩罚系数
# ====================================================

network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                        weight_decay_lambda=weight_decay_lambda)
optimizer = SGD(lr=0.01)

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break


# 3.グラフの描画==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()