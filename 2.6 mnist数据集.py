"""
mnist的图像数据是28像素X28像素的灰度图像（1通道）；
各个像素的取值在0~255之间；
每个图像数据都相应地标有0-9标签；
"""

import sys, os
sys.path.append(os.pardir)  # 为导入父目录中的文件而进行的设定
from dataset.mnist import load_mnist  # 第一次调用load_mnist函数，联网下载数据；第二次调用只读存入本地的pickle文件
import numpy as np
from PIL import Image  # python image library模块
(x_train, t_train), (x_test, t_test), = load_mnist(flatten=True, normalize=False)
# 输出各个数据的形状
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

# 显示mnist图像
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


img = x_train[0]
print(type(img))
label = t_train[0]
print(label)  # 5
print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 把图像的形状变成原来的形状
print(img.shape)  # (28, 28)
img_show(img)

