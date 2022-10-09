"""
CNN 中用到的卷积层在“观察”什么呢？
本节将通过卷积层的可视化，探索CNN 中到底进行了什么处理。

刚才我们对 MNIST 数据集进行了简单的 CNN 学习。
当时，第 1 层的卷积层的权重的形状是 (30, 1, 5, 5)，即 30 个大小为 5 × 5、通道为 1 的滤波器。
滤波器大小是 5 × 5、通道数是 1，意味着滤波器可以可视化为 1 通道的灰度图像。
现在，我们将卷积层（第 1 层）的滤波器显示为图像。
这里，我们来比较一下学习前和学习后的权重，结果如图 7-24 所示（源代码在ch07/visualize_filter.py 中）。

图 7-24　学习前和学习后的第 1 层的卷积层的权重：
虽然权重的元素是实数，但是在图像的显示上，统一将最小值显示为黑色（0），最大值显示为白色（255）


图 7-24 中，学习前的滤波器是随机进行初始化的，所以在黑白的浓淡上没有规律可循，但学习后的滤波器变成了有规律的图像。
我们发现，通过学习，滤波器被更新成了有规律的滤波器，比如从白到黑渐变的滤波器、含有块状区域（称为blob）的滤波器等。


如果要问图 7-24 中右边的有规律的滤波器在“观察”什么，
答案就是它在观察边缘（颜色变化的分界线）和斑块（局部的块状区域）等。
比如，左半部分为白色、右半部分为黑色的滤波器的情况下，如图 7-25 所示，会对垂直方向上的边缘有响应。

图 7-25　对水平方向上和垂直方向上的边缘有响应的滤波器：
输出图像 1中，垂直方向的边缘上出现白色像素，
输出图像 2 中，水平方向的边缘上出现很多白色像素

图 7-25 中显示了选择两个学习完的滤波器对输入图像进行卷积处理时的结果。
我们发现“滤波器 1”对垂直方向上的边缘有响应，
“滤波器 2”对水平方向上的边缘有响应。

由此可知，卷积层的滤波器会提取边缘或斑块等原始信息。
而刚才实现的 CNN会将这些原始信息传递给后面的层。

上面的结果是针对第 1 层的卷积层得出的。
第 1 层的卷积层中提取了边缘或斑块等“低级”信息，
那么在堆叠了多层的 CNN 中，各层中又会提取什么样的信息呢？

根据深度学习的可视化相关的研究 [17][18]，随着层次加深，提取的信息（正确地讲，是反映强烈的神经元）也越来越抽象。

图 7-26 中展示了进行一般物体识别（车或狗等）的 8 层 CNN。
这个网络结构的名称是下一节要介绍的 AlexNet。
AlexNet 网络结构堆叠了多层卷积层和池化层，最后经过全连接层输出结果。
图 7-26 的方块表示的是中间数据，对于这些中间数据，会连续应用卷积运算。


图 7-26　CNN 的卷积层中提取的信息。
第 1 层的神经元对边缘或斑块有响应，
第 3 层对纹理有响应，
第 5 层对物体部件有响应，
最后的全连接层对物体的类别（狗或车）有响应（图像引用自文献 [19]）

如图 7-26 所示，如果堆叠了多层卷积层，则随着层次加深，提取的信息也愈加复杂、抽象，这是深度学习中很有意思的一个地方。
最开始的层对简单的边缘有响应，接下来的层对纹理有响应，再后面的层对更加复杂的物体部件有响应。
也就是说，随着层次加深，神经元从简单的形状向“高级”信息变化。
换句话说，就像我们理解东西的“含义”一样，响应的对象在逐渐变化。

"""
# coding: utf-8
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.layers import Convolution, Relu, Pooling, Affine, SoftmaxWithLoss


class SimpleConvNet(object):

    def __init__(self, input_dim=(1,28,28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100,
                 output_size=10,
                 weight_init_std=0.01
                 ):
        """

        :param input_dim: 输入数据的维度（通道，高，长）
        :param conv_param: 卷积层的超参数
        :param hidden_size: 隐藏层的神经元数量
        :param output_size: 输出层的神经元数量
        :param weight_init_std: 初始化时权重的标准差
        """
        # 卷积层的超参数取出
        filter_num = conv_param['filter_num']  #滤波器的数量
        filter_size = conv_param['filter_size']  #滤波器的大小
        filter_pad = conv_param['pad']  #填充
        filter_stride = conv_param['stride']  #步幅
        # 计算卷积层的输出大小
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad)/filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
        # 权重参数初始化
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['w2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['w3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
        # 生成必要的层
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['w1'],
                                           self.params['b1'],
                                           conv_param['stride'],
                                           conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2,pool_w=2,stride=2)
        self.layers['Affine1'] = Affine(self.params['w2'],
                                        self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['w3'],
                                        self.params['b3'])
        # 最后一层

        self.last_layer = SoftmaxWithLoss()

    # 推理函数
    def predict(self, x):
        """
        从头开始依次调用添加的层，将结果传递给下一层
        :param x: 输入数据
        :return:
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # 求损失函数
    def loss(self, x, t):
        """
        除了使用predict()的forward()，还使用最后一层的forward()
        :param x:输入数据
        :param t:标签数据
        :return:
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    # 求参数梯度
    def gradient(self, x, t):
        """
        基于误差的反向传播
        :param x:
        :param t:
        :return: 参数的梯度
        """
        self.loss(x, t)
        dout = 1
        # 各个层均实现了反向传播
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 各个参数的梯度保存到字典grads中
        grads = {}
        grads['w1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['w2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['w3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['w' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]


def filter_show(filters, nx=8, margin=3, scale=10):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    # 读取权重维度
    FN, C, FH, FW = filters.shape
    # np.ceil: 向上取整函数
    ny = int(np.ceil(FN / nx))
    # 生成图像对象
    fig = plt.figure()
    # 调整子图参数
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


network = SimpleConvNet()
# 初始状态的滤波器
# filter_show(network.params['w1'])

# 训练完成后，加载已有权重参数
network.load_params("params.pkl")
filter_show(network.params['w1'])

