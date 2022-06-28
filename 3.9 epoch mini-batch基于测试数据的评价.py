"""

"""
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

def sigmoid(x):
    """sigmoid激活函数"""
    return 1/(1 + np.exp(-x))


def numerical_gradient(f, x):
    """遍历数组的各个元素，进行数值微分"""
    h = 1e-4
    grad = np.zeros_like(x)


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
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
        it.iternext()

    return grad


def soft_max(a):
    """输出层的softmax激活函数"""
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def cross_entropy_error(y, t):
    """单个数据的损失函数"""
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

# def cross_entropy_error(y, t):
#     delta = 1e-7
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
#     # 抽出正确标签对应的输出值
#     batch_size = y.shape[0]
#     print('batch_size: ', batch_size)
#     return -np.sum(np.log(y[np.arange(batch_size), t] + delta))/batch_size

class TwoLayerNet(object):
    """epoch mini-batch SGD"""
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.0):
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = soft_max(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accurary(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(y, axis=1)
        accurary = np.sum(y == t)/float(x.shape[0])
        return accurary

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['w1'] = numerical_gradient(loss_W, self.params['w1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['w2'] = numerical_gradient(loss_W, self.params['w2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

# 获取mini-batch
(x_train, t_train), (x_test, t_test), = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 超参数
iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

iter_per_epoch = max(train_size/batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
# 这个是按照mini-batch，完成一个epoch;
# for i in range(0, train_size, batch_size):
#     x_batch = x_train[i:i+batch_size]
#     t_batch = t_train[i:i+batch_size]

# 重复一千次，每次取mini-batch数据
for i in range(iters_num):
    # 随机选取的mini-batch，不是严格的epoch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)

    # 更新参数
    for key in ('w1', 'b1', 'w2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 计算损失
    loss = network.loss(x_batch, t_batch)

    # 记录学习过程
    train_loss_list.append(loss)

    # 每经历一次epoch，计算一次精确度，记录一次学习过程；
    if i % iter_per_epoch == 0:
        train_acc = network.accurary(x_train, t_train)
        test_acc = network.accurary(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc) )

print("train_loss_list: ", train_loss_list)

# 绘图观察训练集和测试集的精确度；
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

