"""
首先，来看一下 CNN 的网络结构，了解 CNN 的大致框架。
CNN 和之前介绍的神经网络一样，可以像乐高积木一样通过组装层来构建。
不过，CNN 中新出现了卷积层（Convolution 层）和池化层（Pooling 层）。
卷积层和池化层将在下一节详细介绍，这里我们先看一下如何组装层以构建 CNN。

之前介绍的神经网络中，相邻层的所有神经元之间都有连接，这称为全连接（fully-connected）。
另外，我们用 Affine 层实现了全连接层。
如果使用这个Affine 层，一个 5 层的全连接的神经网络就可以通过图 7-1 所示的网络结构来实现。

图 7-1　基于全连接层（Affine 层）的网络的例子

如图 7-1 所示，全连接的神经网络中，Affine 层后面跟着激活函数 ReLU层（或者 Sigmoid 层）。
这里堆叠了 4 层“Affine-ReLU”组合，然后第 5 层是 Affine 层，最后由 Softmax 层输出最终结果（概率）。

那么，CNN 会是什么样的结构呢？图 7-2 是 CNN 的一个例子。

图 7-2　基于 CNN 的网络的例子：新增了 Convolution 层和 Pooling层（用灰色的方块表示）

如图 7-2 所示，CNN 中新增了 Convolution 层和 Pooling 层。CNN 的层的连接顺序是“Convolution - ReLU -（Pooling）”（Pooling 层有时会被省略）。
这可以理解为之前的“Affine - ReLU”连接被替换成了“Convolution -ReLU -（Pooling）”连接。


还需要注意的是，在图 7-2 的 CNN 中，靠近输出的层中使用了之前的“Affine - ReLU”组合。
此外，最后的输出层中使用了之前的“Affine -Softmax”组合。
这些都是一般的 CNN 中比较常见的结构。




"""