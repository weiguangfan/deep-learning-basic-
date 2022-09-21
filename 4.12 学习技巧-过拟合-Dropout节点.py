"""
作为抑制过拟合的方法，前面我们介绍了为损失函数加上权重的 L2 范数的权值衰减方法。
该方法可以简单地实现，在某种程度上能够抑制过拟合。
但是，如果网络的模型变得很复杂，只用权值衰减就难以应对了。
在这种情况下，我们经常会使用 Dropout [14] 方法。

Dropout 是一种在学习的过程中随机删除神经元的方法。
训练时，随机选出隐藏层的神经元，然后将其删除。
被删除的神经元不再进行信号的传递，如图 6-22所示。
训练时，每传递一次数据，就会随机选择要删除的神经元。
然后，测试时，虽然会传递所有的神经元信号，但是对于各个神经元的输出，要乘上训练时的删除比例后再输出。

图 6-22　Dropout 的概念图（引用自文献 [14]）：左边是一般的神经网络，右边是应用了 Dropout 的网络。
Dropout 通过随机选择并删除神经元，停止向前传递信号

下面我们来实现 Dropout。
这里的实现重视易理解性。
不过，因为训练时如果进行恰当的计算的话，
正向传播时单纯地传递数据就可以了（不用乘以删除比例），所以深度学习的框架中进行了这样的实现。
关于高效的实现，可以参考Chainer 中实现的 Dropout。

这里的要点是，每次正向传播时，self.mask 中都会以 False 的形式保存要删除的神经元。
self.mask 会随机生成和 x 形状相同的数组，并将值比dropout_ratio 大的元素设为 True。
反向传播时的行为和 ReLU 相同。
也就是说，正向传播时传递了信号的神经元，反向传播时按原样传递信号；
正向传播时没有传递信号的神经元，反向传播时信号将停在那里。

现在，我们使用 MNIST 数据集进行验证，以确认 Dropout 的效果。
源代码在ch06/overfit_dropout.py 中。
另外，源代码中使用了 Trainer 类来简化实现。











"""
import numpy as np


class Dropout(object):

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        # 每次正向传播时，self.mask都会以False的形式保存要删除的神经元。
        # self.mask会随机生成和x形状相同的数组，并将值比dropout_ratio大的元素设为True.
        # 反向传播时的行为和ReLu相同。
        # 也就是说，正向传播时传递了信号的神经元，反向传播时按原样传递信号；
        # 正向传播时没有传递信号的神经元，反向传播时信号将停在那里；
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.randn(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask

