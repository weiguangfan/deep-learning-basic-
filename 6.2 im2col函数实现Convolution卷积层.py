import numpy as np
from common.util import im2col, col2im


class Convolution(object):

    def __init__(self, w, b, stride=1, pad=0):
        self.w = w
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.w.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH)/self.stride)
        out_w = int(1 + (W + 2*self.pad - FW)/self.stride)
        col1 = im2col(x, FH, FW, self.stride, self.pad)
        col_w = self.w.reshape(FN,-1).T
        out = np.dot(col1, col_w) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2)
        return out

    def backward(self, dout):
        pass




















