import numpy as np
# x = np.random.rand(2)
# w = np.random.rand(2,3)
# b = np.random.rand(3)
#
# print(x.shape)
# print(w.shape)
# print(b.shape)
#
# y = np.dot(x, w) + b
# print(y)
# print(y.shape)

# x_dot_w = np.array([[0,0,0],[10,10,10]])
# print(x_dot_w)
# print(x_dot_w.shape)
# b = np.array([1,2,3])
# print(b.shape)
# print(b)
#
# y = x_dot_w + b
# print(y)
# print(y.shape)
#
# dy = np.array([[1,2,3],[4,5,6]])
# print(dy.shape)
# print(dy)
# db = np.sum(dy,axis=0)
# print(db.shape)
# print(db)

class Affine(object):
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.w) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx







