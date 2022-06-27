# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.trainer import Trainer
from collections import OrderedDict
from common.layers import Convolution, Relu, Pooling, Affine, SoftmaxWithLoss

class SimpleConvNet(object):

    def __init__(self, input_dim=(1, 28, 28),
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
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 処理に時間のかかる場合はデータを削減
#x_train, t_train = x_train[:5000], t_train[:5000]
#x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 20

network = SimpleConvNet(input_dim=(1,28,28),
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("params.pkl")
print("Saved Network Parameters!")

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
