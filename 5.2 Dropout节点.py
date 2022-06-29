"""
如果网络的模型变得很复杂，只用权值衰减就难以应付了。
DropOut是一种在学习的过程中随机删除神经元的方法。
训练时，随机选出隐藏层的神经元，然后将其删除。被删除的神经元不再进行信号的传递。
训练时，每传递一次数据，就会随机选择要删除的神经元。
测试时，虽然会传递所有的神经元信号，但是，对于各个神经元的输出，要乘上训练时删除的比例后再输出。

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

