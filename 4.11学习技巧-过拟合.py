"""
机器学习的问题中，过拟合是一个很常见的问题。
过拟合指的是只能拟合训练数据，但不能很好地拟合不包含在训练数据中的其他数据的状态。
机器学习的目标是提高泛化能力，即便是没有包含在训练数据里的未观测数据，也希望模型可以进行正确的识别。
我们可以制作复杂的、表现力强的模型，但是相应地，抑制过拟合的技巧也很重要。

发生过拟合的原因，主要有以下两个。
模型拥有大量参数、表现力强。
训练数据少。
这里，我们故意满足这两个条件，制造过拟合现象。
为此，要从 MNIST 数据集原本的 60000 个训练数据中只选定 300 个，
并且，为了增加网络的复杂度，使用 7 层网络（每层有 100 个神经元，激活函数为 ReLU）。

下面是用于实验的部分代码（对应文件在 ch06/overfit_weight_decay.py中）。
首先是用于读入数据的代码。

接着是进行训练的代码。
和之前的代码一样，按 epoch 分别算出所有训练数据和所有测试数据的识别精度。

train_acc_list 和 test_acc_list 中以 epoch 为单位（看完了所有训练数据的单位）保存识别精度。
现在，我们将这些列表（train_acc_list、test_acc_list）绘成图，结果如图 6-20 所示。

图 6-20　训练数据（train）和测试数据（test）的识别精度的变化

过了 100 个 epoch 左右后，用训练数据测量到的识别精度几乎都为 100%。
但是，对于测试数据，离 100% 的识别精度还有较大的差距。
如此大的识别精度差距，是只拟合了训练数据的结果。
从图中可知，模型对训练时没有使用的一般数据（测试数据）拟合得不是很好。

权值衰减是一直以来经常被使用的一种抑制过拟合的方法。
该方法通过在学习的过程中对大的权重进行惩罚，来抑制过拟合。
很多过拟合原本就是因为权重参数取值过大才发生的。

复习一下，神经网络的学习目的是减小损失函数的值。
这时，例如为损失函数加上权重的平方范数（L2 范数）。
这样一来，就可以抑制权重变大。
用符号表示的话，
如果将权重记为 W，L2 范数的权值衰减就是 0.5 * lambda * math.pow(W)，
然后将这个 0.5 * lambda * math.pow(W) 加到损失函数上。
这里，λ 是控制正则化强度的超参数。
λ 设置得越大，对大的权重施加的惩罚就越重。
此外，0.5 * lambda * math.pow(W) 开头的 0.5 是用于将 0.5 * lambda * math.pow(W) 的求导结果变成 lambda * W 的调整用常量。

对于所有权重，权值衰减方法都会为损失函数加上 0.5 * lambda * math.pow(W)。
因此，在求权重梯度的计算中，要为之前的误差反向传播法的结果加上正则化项的导数 lambda * W 。

L2 范数相当于各个元素的平方和。
用数学式表示的话，假设有权重W = (w1,w2,...,wn)，则 L2 范数可用math.sqrt(math.pow(w1) + ... + math.pow(wn))  计算出来。
除了L2 范数，还有 L1 范数、L ∞范数等。
L1 范数是各个元素的绝对值之和，相当于math.abs(w1) + ... +math.abs(w2) 。
L∞范数也称为 Max 范数，相当于各个元素的绝对值中最大的那一个。
L2 范数、L1 范数、L∞范数都可以用作正则化项，它们各有各的特点，不过这里我们要实现的是比较常用的 L2 范数。

现在我们来进行实验。
对于刚刚进行的实验，应用 λ = 0.1 的权值衰减，结果如图 6-21 所示
（对应权值衰减的网络在 common/multi_layer_net.py 中，用于实验的代码在 ch06/overfit_weight_decay.py 中）。

图 6-21　使用了权值衰减的训练数据（train）和测试数据（test）的识别精度的变化

如图 6-21 所示，虽然训练数据的识别精度和测试数据的识别精度之间有差距，但是与没有使用权值衰减的图 6-20 的结果相比，差距变小了。
这说明过拟合受到了抑制。
此外，还要注意，训练数据的识别精度没有达到 100%（1.0）。

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