"""
介绍完神经网络的结构之后，现在我们来试着解决实际问题。
这里我们来进行手写数字图像的分类。
假设学习已经全部结束，我们使用学习到的参数，先实现神经网络的“推理处理”。
这个推理处理也称为神经网络的前向传播（forwardpropagation）。

和求解机器学习问题的步骤（分成学习和推理两个阶段进行）一样，
使用神经网络解决问题时，
也需要首先使用训练数据（学习数据）进行权重参数的学习；
进行推理时，使用刚才学习到的参数，对输入数据进行分类。

这里使用的数据集是 MNIST 手写数字图像集。
MNIST 是机器学习领域最有名的数据集之一，被应用于从简单的实验到发表的论文研究等各种场合。
实际上，在阅读图像识别或机器学习的论文时，MNIST 数据集经常作为实验用的数据出现。
MNIST 数据集是由 0 到 9 的数字图像构成的（图 3-24）。
训练图像有 6万张，测试图像有 1 万张，这些图像可以用于学习和推理。
MNIST 数据集的一般使用方法是，先用训练图像进行学习，再用学习到的模型度量能在多大程度上对测试图像进行正确的分类。

图 3-24　MNIST 图像数据集的例子

MNIST 的图像数据是 28 像素 × 28 像素的灰度图像（1 通道），各个像素的取值在 0 到 255 之间。
每个图像数据都相应地标有“7”“2”“1”等标签。

本书提供了便利的 Python 脚本 mnist.py，
该脚本支持从下载 MNIST 数据集到将这些数据转换成 NumPy 数组等处理（mnist.py 在 dataset 目录下）。
使用 mnist.py 时，当前目录必须是 ch01、ch02、ch03、...、ch08 目录中的一个。
使用 mnist.py 中的 load_mnist() 函数，就可以按下述方式轻松读入MNIST 数据。

首先，为了导入父目录中的文件，进行相应的设定 。
然后，导入dataset/mnist.py 中的 load_mnist 函数。
最后，使用 load_mnist 函数，读入 MNIST 数据集。
第一次调用 load_mnist 函数时，因为要下载 MNIST 数据集，所以需要接入网络。
第 2 次及以后的调用只需读入保存在本地的文件（pickle 文件）即可，因此处理所需的时间非常短。

观察本书源代码可知，
上述代码在 mnist_show.py 文件中。
mnist_show.py 文件的当前目录是ch03，
但包含 load_mnist() 函数的 mnist.py 文件在 dataset 目录下。
因此，mnist_show.py 文件不能跨目录直接导入 mnist.py 文件。
sys.path.append(os.pardir) 语句实际上是把父目录 deep-learning-from-scratch 加入到 sys.path（Python 的搜索模块的路径集）中，
从而可以导入 deep-learning-from-scratch 下的任何目录（包括 dataset 目录）中的任何文件。

用来读入 MNIST 图像的文件在本书提供的源代码的 dataset 目录下。
并且，我们假定了这个 MNIST 数据集只能从 ch01、ch02、ch03、...、ch08 目录中使用，
因此，使用时需要从父目录（dataset 目录）中导入文件，为此需要添加 sys.path.append(os.pardir) 语句。

load_mnist 函数以“( 训练图像, 训练标签 )，( 测试图像, 测试标签)”的形式返回读入的 MNIST 数据。
此外，还可以像load_mnist(normalize=True, flatten=True, one_hot_label=False) 这样，设置 3 个参数。
第 1 个参数 normalize 设置是否将输入图像正规化为 0.0～1.0 的值。
如果将该参数设置为 False，则输入图像的像素会保持原来的 0～255。
第 2 个参数 flatten 设置是否展开输入图像（变成一维数组）。
如果将该参数设置为 False，则输入图像为 1 × 28 × 28 的三维数组；
若设置为 True，则输入图像会保存为由 784 个元素构成的一维数组。
第 3 个参数one_hot_label 设置是否将标签保存为 one-hot 表示（one-hotrepresentation）。
one-hot 表示是仅正确解标签为 1，其余皆为 0 的数组，就像 [0,0,1,0,0,0,0,0,0,0] 这样。
当 one_hot_label 为 False 时，只是像7、2 这样简单保存正确解标签；
当 one_hot_label 为 True 时，标签则保存为one-hot 表示。

Python 有 pickle 这个便利的功能。
这个功能可以将程序运行中的对象保存为文件。
如果加载保存过的 pickle 文件，可以立刻复原之前程序运行中的对象。
用于读入 MNIST 数据集的 load_mnist() 函数内部也使用了pickle 功能（在第 2 次及以后读入时）。
利用 pickle 功能，可以高效地完成 MNIST 数据的准备工作。

现在，我们试着显示 MNIST 图像，同时也确认一下数据。
图像的显示使用PIL（Python Image Library）模块。
执行下述代码后，训练图像的第一张就会显示出来，如图 3-25 所示（源代码在 ch03/mnist_show.py 中）。
图 3-25　显示 MNIST 图像

这里需要注意的是，flatten=True 时读入的图像是以一列（一维）NumPy 数组的形式保存的。
因此，显示图像时，需要把它变为原来的 28 像素 × 28 像素的形状。
可以通过 reshape() 方法的参数指定期望的形状，更改 NumPy 数组的形状。
此外，还需要把保存为 NumPy 数组的图像数据转换为 PIL 用的数据对象，这个转换处理由 Image.fromarray() 来完成。

"""

import sys, os
sys.path.append(os.pardir)  # 为导入父目录中的文件而进行的设定
from dataset.mnist import load_mnist  # 第一次调用load_mnist函数，联网下载数据；第二次调用只读存入本地的pickle文件
import numpy as np
from PIL import Image  # python image library模块
(x_train, t_train), (x_test, t_test), = load_mnist(flatten=True, normalize=False)
# 输出各个数据的形状
# print(x_train.shape)
# print(t_train.shape)
# print(x_test.shape)
# print(t_test.shape)

# 显示mnist图像


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


img = x_train[0]
print(type(img))
print(img.shape)  # (784,)
label = t_train[0]
print(label)  # 5
img = img.reshape(28, 28)  # 把图像的形状变成原来的形状
print(img.shape)  # (28, 28)
img_show(img)

