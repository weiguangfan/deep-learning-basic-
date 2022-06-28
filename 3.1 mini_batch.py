"""
mini-batch学习：利用一部分数据近似整体
"""


import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np

# 加载数据
(x_train, t_train), (x_test, t_test), = load_mnist(one_hot_label=True, normalize=True)
# print(x_train.shape)  # (60000, 784) 输入数据784维
# print(t_train.shape)  # (60000, 10) 监督数据10维

# 从训练数据随机抽取10笔数据
train_size = x_train.shape[0]
# print(train_size)
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
# print(batch_mask)
x_batch = x_train[batch_mask]  # [56238 14055 32411 41573 48975 37604 50257 45892 41825 49994] 索引数组
# print(x_batch.shape)  # (10, 784)
t_batch = t_train[batch_mask]
# print(t_batch.shape)  # (10, 10)

# 标签数据为onr-hot形式
# def cross_entropy_error(y, t):
#     """mini-batch 交叉熵损失：单个数据的平均交叉熵损失"""
#     delta = 1e-7
#     # 单个数据的交叉熵损失，改变形状
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
#     batch_size = y.shape[0]
#     # 单个数据的平均交叉熵损失
#     return -np.sum(t * np.log(y + delta)) / batch_size

# 标签数据为真实标签
def cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 抽出正确标签对应的输出值
    batch_size = y.shape[0]
    print('batch_size: ', batch_size)
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta))/batch_size
