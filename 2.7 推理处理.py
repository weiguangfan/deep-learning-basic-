"""
mnist数据集实现神经网络的推理；
输入层784个神经元，输出层10个神经元；

"""
import pickle
import numpy as np
from dataset.mnist import load_mnist


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def soft_max(a):
    """进化的防止溢出的激活函数"""
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def get_data():
    """获取数据"""
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    """读入保存的pickle文件"""
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


x, t = get_data()  # t输出真实类别标签的数组
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])  # 以数组形式输出各个标签对应的概率
    p = np.argmax(y)  # 取出概率列表最大值的索引
    if p == t[i]:
        accuracy_cnt += 1
print("accuracy_cnt: " + str(float(accuracy_cnt / len(x))))

