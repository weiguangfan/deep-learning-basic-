"""
池化层forward处理的实现流程：
    1 展开输入数据；
    2 求各行的最大值；
    3 转换为合适的输出大小；
池化层的backward处理可以参考Relu层的实现中使用的max的反向传播；
池化层的实现和卷积层相同，都使用im2col展开输入数据。
池化时，在通道方向上是独立的，这一点和卷积层不同。
"""
from common.util import im2col, col2im
import numpy as np

class Pooling(object):

    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None


    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h)/self.stride)
        out_w = int(1 + (W - self.pool_w)/self.stride)
        # 输入展开
        col = im2col(x, self.pool_h, self.pool_w, stride=self.stride, pad=self.pad)
        col = col.reshpe(-1, self.pool_h*self.pool_w)
        # 最大值
        out = np.max(col, axis=1)
        # 转换
        out = out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)

        return out

    def backward(self, dout):
        pass
