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
        col = im2col(x, self.pool_h, self.pool_w, stride=self.stride, pad=self.pad)
        col = col.reshpe(-1, self.pool_h*self.pool_w)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)

        return out

    def backward(self, dout):
        pass















