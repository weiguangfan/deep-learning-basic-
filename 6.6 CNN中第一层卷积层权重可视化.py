"""
通过卷积层的可视化，探索CNN中到底进行了什么处理；
卷积层的滤波器会提取边缘或斑块等原始信息。
实现的CNN会将这些原始信息传递给后面的层；
学习前的滤波器是随机初始化的， 在黑白的浓淡上没有规律可循；
学习后的滤波器变成了有规律的图像；
从白到黑渐变的滤波器Edge（滤波器观察边缘，颜色变化的分界线）、含有块状区域的滤波器Blob（滤波器观察斑块，局部的块状区域）；
第一层的卷积层中提取了边缘或斑块等低级信息，那么在堆叠了多层的CNN中，各层中又会提取什么样的信息呢？
随着层次加深，提取信息也越来越抽象。

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

