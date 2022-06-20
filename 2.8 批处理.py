import pickle

import numpy as np
from dataset.mnist import load_mnist

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def soft_max(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = soft_max(a3)
    return y

# x, t = get_data()
# network = init_network()
# w1, w2, w3 = network['W1'], network['W2'], network['W3']
# print(x.shape)
# print(x[0].shape)
# print(w1.shape)
# print(w2.shape)
# print(w3.shape)

# print(list(range(0, 10)))
# print(list(range(0, 10, 3)))

# x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
# y = np.argmax(x, axis=1)
# print(y)

# y = np.array([1, 2, 1, 0])
# t = np.array([1, 2, 0, 0])
# print(y == t)
# print(np.sum(y == t))


x, t = get_data()
network = init_network()
batch_size = 100
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
















