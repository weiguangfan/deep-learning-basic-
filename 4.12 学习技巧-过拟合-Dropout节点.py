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

common/trainer.py 中实现了 Trainer 类。
这个类可以负责前面所进行的网络的学习。
详细内容可以参照 common/trainer.py 和ch06/overfit_dropout.py。

Dropout 的实验和前面的实验一样，
使用 7 层网络（每层有 100 个神经元，激活函数为 ReLU），
一个使用 Dropout，另一个不使用 Dropout，实验的结果如图 6-23 所示。

图 6-23　左边没有使用 Dropout，右边使用了Dropout（dropout_rate=0.15）

图 6-23 中，通过使用 Dropout，训练数据和测试数据的识别精度的差距变小了。
并且，训练数据也没有到达 100% 的识别精度。
像这样，通过使用 Dropout，即便是表现力强的网络，也可以抑制过拟合。


机器学习中经常使用集成学习。
所谓集成学习，就是让多个模型单独进行学习，推理时再取多个模型的输出的平均值。
用神经网络的语境来说，
比如，准备 5 个结构相同（或者类似）的网络，分别进行学习，测试时，以这 5个网络的输出的平均值作为答案。
实验告诉我们，通过进行集成学习，神经网络的识别精度可以提高好几个百分点。
这个集成学习与 Dropout 有密切的关系。
这是因为可以将 Dropout 理解为，
通过在学习过程中随机删除神经元，从而每一次都让不同的模型进行学习。
并且，推理时，通过对神经元的输出乘以删除比例（比如，0.5 等），可以取得模型的平均值。
也就是说，可以理解成，Dropout将集成学习的效果（模拟地）通过一个网络实现了。


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

