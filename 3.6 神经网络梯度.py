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
    print("x: ", x)
    h = 1e-4
    grad = np.zeros_like(x)
    print("grad: ", grad)
    print("grad.shape: ", grad.shape)
    # for idx in range(x.size):
    #     tmp_val = x[idx]
    #     x[idx] = tmp_val + h
    #     fxh1 = f(x)
    #
    #     x[idx] = tmp_val - h
    #     fxh2 = f(x)
    #     grad[idx] = (fxh1 - fxh2)/(2*h)
    #     x[idx] = tmp_val

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    print("it: ", it)
    while not it.finished:
        idx = it.multi_index
        print("idx: ", idx)
        tmp_val = x[idx]
        print('tmp_val: ', tmp_val)
        x[idx] = tmp_val + h
        print('++ x[idx]: : ', x[idx])
        fxh1 = f(x)

        x[idx] = tmp_val - h
        print('-- x[idx]: : ', x[idx])
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        print('grad[idx]: ', grad[idx])
        x[idx] = tmp_val
        it.iternext()

    return grad


class simpleNet(object):
    def __init__(self):
        self.w = np.random.randn(2, 3)

    def predict(self, x):
        print("self.w: ", self.w)
        print("predict input x: ", x)
        return np.dot(x, self.w)

    def loss(self, x, t):
        print("loss_input x: ", x)
        print("loss_input t: ", t)
        z = self.predict(x)
        print("z: ", z)
        print("z.shape: ", z.shape)
        y = soft_max(z)
        print("y: ", y)
        print("y.shape: ", y.shape)
        loss = cross_entropy_error(y, t)
        print("loss: ", loss)
        return loss


net = simpleNet()
print("net.w: ", net.w)
print("net.w.shape: ", net.w.shape)

x = np.array([0.6, 0.9])
print("x: ", x)
print("x.shape: ", x.shape)

# p = net.predict(x)
# print(p)
# print(p.shape)
# print(np.argmax(p))
t = np.array([0, 0, 1])
print("t: ", t)
print("t.shape: ", t.shape)
# net.loss(x, t)


# def f(W):
#     return net.loss(x, t)

f = lambda w:net.loss(x, t)

dw = numerical_gradient(f, net.w)
print(dw)







