"""
CNN典型代表网络：
    LeNet有连续的卷积层和池化层（正确地讲，是只抽选元素的子采样层），最后是经全连接层输出结果；
        激活函数：sigmoid(CNN使用Relu)；
        使用子采样（subsampling）缩小中间数据的大小(CNN使用Max池化)；

    AlexNet网络结构堆叠了多层卷积层和池化层，最后经过全连接层输出结果；
        激活函数：Relu；
        使用局部正规化的LRN(local response normalization)层；
        使用DropOut；
    随着层次加深，提取的信息也愈加复杂、抽象，最开始的层对简单的边缘有响应，接下来的层对纹理有响应，再后面的层对更加复杂的物体部件有响应。
    随着层次加深，神经元从简单的形状向高级信息变化。

    VGG
        VGG是由卷积层和池化层构成的基础CNN；
        它的特点在于将有权重的层（卷积层或者全连接层）叠加至16层（或者19层），具备了深度（根据层的深度，有时也称VGG16或者 VGG18）；
        基于3x3的小型滤波器的卷积层的运算是连续进行的；
        重复进行卷积层重叠2次到4次，再通过池化层将大小减半的处理，最后经由全连接层输出结果；

    GoogLeNet
        网络不仅在纵向上有深度，在横向上也有深度（广度）；
        GoogleNet在横向上有宽度，这称为inception结构。
        inception结构使用了多个大小不同的滤波器（和池化），最后再合并它们的结果。
        GoogLeNet 的特征就是将这个inception 结构用作一个构件（构成元素）。
        在GoogLeNet中，很多地方都使用了大小为1x1的滤波器的卷积层。
        这个1x1的卷积运算通过在通道方向上减小大小，有助于减小参数和实现高速化处理。

    ResNet
        ResNet是微软团队开发的网络。
        它的特征是在于具有比以前的网络更深的结构。
        在深度学习中，过度加深层的话，很多情况下学习将不能顺利进行，导致最终性能不佳。
        ResNet中，为了解决这类问题，导入了快捷结构（也称为捷径或小路）。
        导入这个快捷结构后，就可以随着层的加深而不断提高性能了（当然，层的加深也是有限度的）。
        快捷结构横跨了输入数据的卷积层，将输入x合计到输出。
        在连续2层的卷积层中，将输入x跳着连接至2层后的输出。
        通过快捷结构，原来的2层卷积层的输出f(x)变成了f(x) + x。
        通过引入这种快捷结构，即使加深层，也能高效学习。
        这是因为，通过快捷结构，反向传播时信号可以无衰减地传递。
        因为快捷结构只是原封不动地传递输入数据，所以反向传播时会将来自上游的梯度原封不动的传向下游。
        这里的重点是不对来自上游的梯度进行任何处理，将其原封不动地传向下游。
        基于快捷结构，不用担心梯度会变大变小，能够向前一层传递有意义的梯度。
        之前因为加深层而导致的梯度变小的梯度消失问题就有望得到缓解。
        ResNet以VGG网络为基础，引入快捷结构以加深层，通过以2个卷积层为间隔跳跃式地连接来加深层。
        即使加深到150层以上，识别精度也会持续提高。

ImageNet数据集：
    实践中经常会灵活应用使用ImageNet这个巨大的数据集学习到的权重数据，这称为迁移学习。
    将学习完的权重（的一部分）复制到其他神经网络，进行再学习（fine tuning）.

深度学习：
    深度学习是加深了层的神经网络。
    只需要叠加层，就可以创建深度网络。
    1 基于3x3的小型滤波器的卷积层
    2 激活函数是Relu
    3 全连接层的后面使用DropOut层
    4 基于Adam的最优化
    5 使用He初始值作为权重初始值

    filter:fh=3,fw=3,c=[16,16,32,32,64,64]
    pooling层:逐渐减小中间数据的空间大小
    使用He初始值作为权重的初始值；
    使用Adam更新参数；

进一步提高识别精度的技术和线索：
    集成学习
    学习率衰减
    Data Augmentation

Data Augmentation基于算法人为地扩充输入图像（训练图像）。
通过旋转、垂直或水平方向上的移动等微小变化，增加图像的数量。
其他方法扩充图像：裁剪图像的crop处理、将图像左右翻转的flip处理、施加亮度、放大缩小；

加深层的好处：
    减少网络的参数数量；（用更少的参数达到同等水平甚至更强的表现力）
    使学习更加高效；（减少学习数据）
    分层次传递信息；
叠加小型滤波器来加深网络的好处是可以减少参数的数量，扩大感受野（receptive field,给神经元施加变化的某个局部空间区域)。
并且，通过叠加层，将Relu等激活函数夹在卷积层的中间，进一步提高网络的表现力。
这是因为向网络添加了基于激活函数的非线性表现力，通过非线性函数的叠加，可以表现更复杂的东西。


"""
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.trainer import Trainer

class DeepConvNet:
    """認識率99%以上の高精度なConvNet

    ネットワーク構成は下記の通り
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        affine - relu - dropout - affine - dropout - softmax
    """
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param_1 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_2 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_3 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_4 = {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},
                 conv_param_5 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_6 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 hidden_size=50, output_size=10):
        # 重みの初期化===========
        # 各層のニューロンひとつあたりが、前層のニューロンといくつのつながりがあるか（TODO:自動で計算する）
        pre_node_nums = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size])
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLUを使う場合に推奨される初期値

        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6]):
            self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn(conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']

        self.params['W7'] = weight_init_scales[6] * np.random.randn(64*4*4, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = weight_init_scales[7] * np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)

        # 生成层
        self.layers = []
        self.layers.append(Convolution(self.params['W1'], self.params['b1'],
                                       conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W2'], self.params['b2'],
                                       conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W3'], self.params['b3'],
                                       conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W4'], self.params['b4'],
                                       conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W5'], self.params['b5'],
                                       conv_param_5['stride'], conv_param_5['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W6'], self.params['b6'],
                                       conv_param_6['stride'], conv_param_6['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Affine(self.params['W7'], self.params['b7']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params['W8'], self.params['b8']))
        self.layers.append(Dropout(0.5))

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db

        return grads

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

        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()

network.load_params("deep_convnet_params.pkl")

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
# network.save_params("deep_convnet_params.pkl")
# print("Saved Network Parameters!")
