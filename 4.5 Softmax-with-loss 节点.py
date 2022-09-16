"""
最后介绍一下输出层的 softmax 函数。
前面我们提到过，softmax 函数会将输入值正规化之后再输出。
比如手写数字识别时，Softmax 层的输出如图 5-28 所示。

图 5-28　输入图像通过 Affine 层和 ReLU 层进行转换，10 个输入通过Softmax 层进行正规化。
在这个例子中，“0”的得分是 5.3，这个值经过Softmax 层转换为 0.008（0.8%）；“2”的得分是 10.1，被转换为0.991（99.1%）

在图 5-28 中，Softmax 层将输入值正规化（将输出值的和调整为 1）之后再输出。
另外，因为手写数字识别要进行 10 类分类，所以向Softmax 层的输入也有10 个。

神经网络中进行的处理有推理（inference）和学习两个阶段。神经网络的推理通常不使用 Softmax 层。
比如，用图 5-28 的网络进行推理时，会将最后一个 Affine 层的输出作为识别结果。
神经网络中未被正规化的输出结果（图 5-28 中 Softmax 层前面的 Affine 层的输出）有时被称为“得分”。
也就是说，当神经网络的推理只需要给出一个答案的情况下，因为此时只对得分最大值感兴趣，所以不需要 Softmax 层。
不过，神经网络的学习阶段则需要 Softmax 层。

下面来实现 Softmax 层。
考虑到这里也包含作为损失函数的交叉熵误差（cross entropy error），所以称为“Softmax-with-Loss 层”。
Softmax-with-Loss 层（Softmax 函数和交叉熵误差）的计算图如图 5-29 所示。

图 5-29　Softmax-with-Loss 层的计算图

可以看到，Softmax-with-Loss 层有些复杂。
这里只给出了最终结果，对Softmax-with-Loss 层的导出过程感兴趣的读者，请参照附录 A。

图 5-29 的计算图可以简化成图 5-30。

图 5-30 的计算图中，softmax 函数记为 Softmax 层，交叉熵误差记为Cross Entropy Error 层。
这里假设要进行 3 类分类，从前面的层接收 3 个输入（得分）。
如图 5-30 所示，Softmax 层将输入 (a1,a2,a3) 正规化，输出 (y1,y2,y3)。
Cross Entropy Error 层接收 Softmax 的输出 (y1,y2,y3) 和教师标签(t1,t2,t3)，从这些数据中输出损失 L。

图 5-30　“简易版”的 Softmax-with-Loss 层的计算图

图 5-30 中要注意的是反向传播的结果。
Softmax 层的反向传播得到了(y1 - t1,y2 - t2, y3 - t3) 这样“漂亮”的结果。
由于 (y1,y2,y3) 是 Softmax 层的输出，(t1,t2,t3) 是监督数据，所以 (y1 - t1,y2 - t2, y3 - t3) 是 Softmax 层的输出和教师标签的差分。
神经网络的反向传播会把这个差分表示的误差传递给前面的层，这是神经网络学习中的重要性质。

这里考虑一个具体的例子，比如思考教师标签是（0, 1, 0），Softmax 层的输出是 (0.3, 0.2, 0.5) 的情形。
因为正确解标签处的概率是 0.2（20%），这个时候的神经网络未能进行正确的识别。
此时，Softmax 层的反向传播传递的是(0.3, -0.8, 0.5) 这样一个大的误差。
因为这个大的误差会向前面的层传播，所以 Softmax 层前面的层会从这个大的误差中学习到“大”的内容。

使用交叉熵误差作为 softmax 函数的损失函数后，反向传播得到 (y1 - t1,y2 - t2, y3 - t3) 这样“漂亮”的结果。
实际上，这样“漂亮”的结果并不是偶然的，而是为了得到这样的结果，特意设计了交叉熵误差函数。
回归问题中输出层使用“恒等函数”，损失函数使用“平方和误差”，也是出于同样的理由（3.5 节）。
也就是说，使用“平方和误差”作为“恒等函数”的损失函数，反向传播才能得到 (y1 - t1,y2 - t2, y3 - t3) 这样“漂亮”的结果。

再举一个例子，比如思考教师标签是 (0, 1, 0)，Softmax 层的输出是(0.01, 0.99, 0) 的情形（这个神经网络识别得相当准确）。
此时 Softmax 层的反向传播传递的是 (0.01, -0.01, 0) 这样一个小的误差。这个小的误差也会向前面的层传播，因为误差很小，所以 Softmax 层前面的层学到的内容也很“小”。

现在来进行 Softmax-with-Loss 层的实现，实现过程如下所示。

"""

import numpy as np


def soft_max(a):
    """改进版的softmax"""
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def cross_entropy_error(y, t):
    """交叉熵损失函数"""
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))



class SoftmaxWithLoss(object):
    """
    softmax-with-loss层的实现；
    反向传播中，输入的导数，等于输出与标签的差值；
    """

    def __init__(self):
        self.loss = None  # 损失
        self.y = None  # softmax输出
        self.t = None  # 监督数据

    def forward(self, x, t):
        """前向传播"""
        self.t = t
        self.y = soft_max(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        """反向传播"""
        batch_size = self.t.shape[0]
        #  将要传播的值除以批的大小后，传递给前面的层的是单个数据误差
        dx = (self.y - self.t) / batch_size
        return dx

"""
这个实现利用了 3.5.2 节和 4.2.4 节中实现的 softmax() 和cross_entropy_error() 函数。
因此，这里的实现非常简单。
请注意反向传播时，将要传播的值除以批的大小（batch_size）后，传递给前面的层的是单个数据的误差。
"""