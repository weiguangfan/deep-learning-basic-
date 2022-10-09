"""
神经网络中，除了权重和偏置等参数，超参数（hyper-parameter）也经常出现。
这里所说的超参数是指，比如各层的神经元数量、batch 大小、参数更新时的学习率或权值衰减等。
如果这些超参数没有设置合适的值，模型的性能就会很差。
虽然超参数的取值非常重要，但是在决定超参数的过程中一般会伴随很多的试错。
本节将介绍尽可能高效地寻找超参数的值的方法。

之前我们使用的数据集分成了训练数据和测试数据，训练数据用于学习，测试数据用于评估泛化能力。
由此，就可以评估是否只过度拟合了训练数据（是否发生了过拟合），以及泛化能力如何等。
下面我们要对超参数设置各种各样的值以进行验证。这里要注意的是，不能使用测试数据评估超参数的性能。
这一点非常重要，但也容易被忽视。

为什么不能用测试数据评估超参数的性能呢？
这是因为如果使用测试数据调整超参数，超参数的值会对测试数据发生过拟合。
换句话说，
用测试数据确认超参数的值的“好坏”，就会导致超参数的值被调整为只拟合测试数据。
这样的话，可能就会得到不能拟合其他数据、泛化能力低的模型。

因此，调整超参数时，必须使用超参数专用的确认数据。
用于调整超参数的数据，一般称为验证数据（validation data）。
我们使用这个验证数据来评估超参数的好坏。

训练数据用于参数（权重和偏置）的学习，
验证数据用于超参数的性能评估。
为了确认泛化能力，要在最后使用（比较理想的是只用一次）测试数据。

根据不同的数据集，
有的会事先分成训练数据、验证数据、测试数据三部分，
有的只分成训练数据和测试数据两部分，
有的则不进行分割。
在这种情况下，用户需要自行进行分割。
如果是 MNIST 数据集，获得验证数据的最简单的方法就是从训练数据中事先分割 20% 作为验证数据，代码如下所示。

这里，分割训练数据前，先打乱了输入数据和教师标签。
这是因为数据集的数据可能存在偏向（比如，数据从“0”到“10”按顺序排列等）。
这里使用的shuffle_dataset 函数利用了 np.random.shuffle，在 common/util.py 中有它的实现。

接下来，我们使用验证数据观察超参数的最优化方法。

进行超参数的最优化时，逐渐缩小超参数的“好值”的存在范围非常重要。
所谓逐渐缩小范围，
是指一开始先大致设定一个范围，
从这个范围中随机选出一个超参数（采样），
用这个采样到的值进行识别精度的评估；
然后，
多次重复该操作，
观察识别精度的结果，
根据这个结果缩小超参数的“好值”的范围。
通过重复这一操作，就可以逐渐确定超参数的合适范围。

有报告 [15] 显示，
在进行神经网络的超参数的最优化时，
与网格搜索等有规律的搜索相比，
随机采样的搜索方式效果更好。
这是因为在多个超参数中，各个超参数对最终的识别精度的影响程度不同。

超参数的范围只要“大致地指定”就可以了。
所谓“大致地指定”，
是指像0.001（10-3）到 1000（103）这样，
以“10 的阶乘”的尺度指定范围（也表述为“用对数尺度（log scale）指定”）。

在超参数的最优化中，要注意的是深度学习需要很长时间（比如，几天或几周）。
因此，在超参数的搜索中，需要尽早放弃那些不符合逻辑的超参数。
于是，在超参数的最优化中，减少学习的 epoch，缩短一次评估所需的时间是一个不错的办法。

以上就是超参数的最优化的内容，简单归纳一下，如下所示。
步骤 0
设定超参数的范围。
步骤 1
从设定的超参数范围中随机采样。
步骤 2
使用步骤 1 中采样到的超参数的值进行学习，通过验证数据评估识别精度（但是要将 epoch 设置得很小）。
步骤 3
重复步骤 1 和步骤 2（100 次等），根据它们的识别精度的结果，缩小超参数的范围。
反复进行上述操作，不断缩小超参数的范围，在缩小到一定程度时，从该范围中选出一个超参数的值。
这就是进行超参数的最优化的一种方法。

这里介绍的超参数的最优化方法是实践性的方法。
不过，这个方法与其说是科学方法，倒不如说有些实践者的经验的感觉。
在超参数的最优化中，如果需要更精炼的方法，可以使用贝叶斯最优化（Bayesianoptimization）。
贝叶斯最优化运用以贝叶斯定理为中心的数学理论，能够更加严密、高效地进行最优化。
详细内容请参考论文“Practical BayesianOptimization of Machine Learning Algorithms”[16] 等。




"""
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 用于导入父目录文件的设置
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

# 减少训练数据以加快速度
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
