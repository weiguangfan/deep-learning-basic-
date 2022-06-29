"""
超参数hyper-parameter:各层的神经元数量、batch大小、参数更新时的学习率、权值衰减等；
如果这些超参数没有设置合适的值，模型的性能就会很差；
调整超参数时，必须使用超参数专用的确认数据。
用于调整超参数的数据，一般称为验证数据validation data;
使用这个验证数据来评估超参数的好坏；
分割训练数据前，先打乱输入数据和标签数据，因为数据集的数据可能存在偏向（比如数据从0-10排列）。
进行超参数的最优化时，逐渐缩小超参数的好值的存在范围非常重要。
所谓逐渐缩小范围，是指一开始先大致设定一个范围，从这个范围中随机选出一个超参数（采样），
用这个采样到的值进行识别精度的评估；
然后，多次重复该操作，观察识别精度的结果；
根据结果缩小超参数的好值的范围。
通过重复这一操作，就可以逐渐确定超参数的合适范围。

步骤：
    1 设定超参数的范围
    2 从设定的超参数范围中随机采样
    3 使用2中采样的超参数的值进行学习，通过验证数据评估识别精度（但是要将epoch设置得很小）
    4 重复2和3（100次等），根据它们的识别精度的结果，缩小超参数的范围。

超参数的范围只要大致地指定就可以了。
所谓大致地指定，是指像0.001 到 1000 这样，以10的阶乘的尺度指定范围；
在超参数的最优化中，要注意的是深度学习需要很长时间。
因此，在超参数的搜索中，需要尽早放弃那些不符合逻辑的超参数。
减少学习的epoch，缩短一次评估所需的时间是一个不错的办法。

在超参数的最优化中，如果需要更精炼的方法，可以使用贝叶斯最优化Bayesian optimization;
这里将学习率和控制权重衰减强度的系数这两个超参数的搜索问题作为对象；
权重衰减系数的初始范围1e-8  到 1e-4
学习率的初始范围 1e-6 到 1e-2
"""
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 打乱训练数据
x_train, t_train = shuffle_dataset(x_train, t_train)

# 分割验证数据
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)

x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]

# 高速化のため訓練データの削減
x_train = x_train[:500]
t_train = t_train[:500]




def __train(lr, weight_decay, epocs=50):
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10, weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs, mini_batch_size=100,
                      optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list


# ハイパーパラメータのランダム探索======================================
optimization_trial = 100
results_val = {}
results_train = {}
for _ in range(optimization_trial):
    # 超参数的初始范围
    weight_decay = 10 ** np.random.uniform(-8, -4)
    lr = 10 ** np.random.uniform(-6, -2)
    # ================================================

    val_acc_list, train_acc_list = __train(lr, weight_decay)
    print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
    key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

# グラフの描画========================================================
print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
    print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)

    plt.subplot(row_num, col_num, i+1)
    plt.title("Best-" + str(i+1))
    plt.ylim(0.0, 1.0)
    if i % 5:
        plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    i += 1

    if i >= graph_draw_num:
        break


plt.show()
