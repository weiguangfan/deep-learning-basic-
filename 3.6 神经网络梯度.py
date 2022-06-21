import sys, os
import numpy as np
sys.path.append(os.pardir)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def soft_max(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    print("grad: ", grad)
    for idx in range(x.size):
        print("idx: ", idx)
        tmp_val = x[idx]
        print('i:tmp_val: ', idx, tmp_val)
        x[idx] = tmp_val + h
        fxh1 = f(x)
        print("idx:fxh1: ", idx, fxh1)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2)/(2*h)
        print("grad: ", grad)
        x[idx] = tmp_val
    return grad


class simpleNet(object):
    def __init__(self):
        self.w = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.w)

    def loss(self, x, t):
        z = self.predict(x)
        y = soft_max(z)
        loss = cross_entropy_error(y, t)
        return loss

net = simpleNet()
print(net.w)
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))
t = np.array([0,0,1])










