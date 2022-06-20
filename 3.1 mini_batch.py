import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np

(x_train, t_train), (x_test, t_test), = load_mnist(one_hot_label=True, normalize=True)
print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
print(train_size)
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
print(batch_mask)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# def cross_entropy_error(y, t):
#     delta = 1e-7
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
#     batch_size = y.shape[0]
#     return -np.sum(t * np.log(y + delta)) / batch_size

def cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta))/batch_size
