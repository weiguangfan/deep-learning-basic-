"""
卷积层的forward处理的实现，通过im2col进行展开，基本上可以实现全连接层affine层一样的实现；
卷积层的backward的实现，和affine层的实现类似；进行im2col的逆处理；
"""

import numpy as np
from common.util import im2col, col2im


class Convolution(object):

    def __init__(self, w, b, stride=1, pad=0):
        """
        初始化接收参数
        :param w: 权重
        :param b: 偏置
        :param stride: 步幅
        :param pad: 填充
        """
        self.w = w
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        """
        滤波器4维；
        :param x: 输入
        :return: 输出
        """
        FN, C, FH, FW = self.w.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH)/self.stride)
        out_w = int(1 + (W + 2*self.pad - FW)/self.stride)
        # 调用函数im2col()，输入展开
        col1 = im2col(x, FH, FW, self.stride, self.pad)
        # 滤波器展开
        col_w = self.w.reshape(FN,-1).T
        # 展开项矩阵相乘
        out = np.dot(col1, col_w) + self.b
        # 重新整理输出的形状
        # transpose()函数：更改多维数组轴的顺序
        out = out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2)
        return out

    def backward(self, dout):
        pass



