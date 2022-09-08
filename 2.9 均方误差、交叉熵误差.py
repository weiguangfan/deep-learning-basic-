"""
本章的主题是神经网络的学习。
这里所说的“学习”是指从训练数据中自动获取最优权重参数的过程。
本章中，为了使神经网络能进行学习，将导入损失函数这一指标。
而学习的目的就是以该损失函数为基准，找出能使它的值达到最小的权重参数。
为了找出尽可能小的损失函数的值，本章我们将介绍利用了函数斜率的梯度法。

神经网络的特征就是可以从数据中学习。
所谓“从数据中学习”，是指可以由数据自动决定权重参数的值。
这是非常了不起的事情！因为如果所有的参数都需要人工决定的话，工作量就太大了。
在第 2 章介绍的感知机的例子中，我们对照着真值表，人工设定了参数的值，但是那时的参数只有 3 个。
而在实际的神经网络中，参数的数量成千上万，在层数更深的深度学习中，参数的数量甚至可以上亿，
想要人工决定这些参数的值是不可能的。
本章将介绍神经网络的学习，即利用数据决定参数值的方法，并用 Python 实现对 MNIST 手写数字数据集的学习。

对于线性可分问题，第 2 章的感知机是可以利用数据自动学习的。
根据“感知机收敛定理”，通过有限次数的学习，线性可分问题是可解的。
但是，非线性可分问题则无法通过（自动）学习来解决。

数据是机器学习的命根子。
从数据中寻找答案、从数据中发现模式、根据数据讲故事......这些机器学习所做的事情，如果没有数据的话，就无从谈起。
因此，数据是机器学习的核心。
这种数据驱动的方法，也可以说脱离了过往以人为中心的方法。

通常要解决某个问题，特别是需要发现某种模式时，人们一般会综合考虑各种因素后再给出回答。
“这个问题好像有这样的规律性？”“不对，可能原因在别的地方。”——类似这样，人们以自己的经验和直觉为线索，通过反复试验推进工作。
而机器学习的方法则极力避免人为介入，尝试从收集到的数据中发现答案（模式）。
神经网络或深度学习则比以往的机器学习方法更能避免人为介入。

现在我们来思考一个具体的问题，比如如何实现数字“5”的识别。
数字 5 是图 4-1 所示的手写图像，我们的目标是实现能区别是否是 5 的程序。
这个问题看起来很简单，大家能想到什么样的算法呢？

图 4-1　手写数字 5 的例子：写法因人而异，五花八门

如果让我们自己来设计一个能将 5 正确分类的程序，就会意外地发现这是一个很难的问题。
人可以简单地识别出 5，但却很难明确说出是基于何种规律而识别出了 5。
此外，从图 4-1 中也可以看到，每个人都有不同的写字习惯，要发现其中的规律是一件非常难的工作。

因此，与其绞尽脑汁，从零开始想出一个可以识别 5 的算法，不如考虑通过有效利用数据来解决这个问题。
一种方案是，先从图像中提取特征量，再用机器学习技术学习这些特征量的模式。
这里所说的“特征量”是指可以从输入数据（输入图像）中准确地提取本质数据（重要的数据）的转换器。
图像的特征量通常表示为向量的形式。
在计算机视觉领域，常用的特征量包括 SIFT、SURF 和 HOG 等。
使用这些特征量将图像数据转换为向量，然后对转换后的向量使用机器学习中的 SVM、KNN 等分类器进行学习。

机器学习的方法中，由机器从收集到的数据中找出规律性。
与从零开始想出算法相比，这种方法可以更高效地解决问题，也能减轻人的负担。
但是需要注意的是，将图像转换为向量时使用的特征量仍是由人设计的。
对于不同的问题，必须使用合适的特征量（必须设计专门的特征量），才能得到好的结果。
比如，为了区分狗的脸部，人们需要考虑与用于识别 5 的特征量不同的其他特征量。
也就是说，即使使用特征量和机器学习的方法，也需要针对不同的问题人工考虑合适的特征量。
到这里，我们介绍了两种针对机器学习任务的方法。
将这两种方法用图来表示，如图 4-2 所示。
图中还展示了神经网络（深度学习）的方法，可以看出该方法不存在人为介入。

图 4-2　从人工设计规则转变为由机器从数据中学习：没有人为介入的方块用灰色表示

如图 4-2 所示，神经网络直接学习图像本身。
在第 2 个方法，即利用特征量和机器学习的方法中，
特征量仍是由人工设计的，
而在神经网络中，连图像中包含的重要特征量也都是由机器来学习的。

深度学习有时也称为端到端机器学习（end-to-end machinelearning）。
这里所说的端到端是指从一端到另一端的意思，也就是从原始数据（输入）中获得目标结果（输出）的意思。

神经网络的优点是对所有的问题都可以用同样的流程来解决。
比如，不管要求解的问题是识别 5，还是识别狗，抑或是识别人脸，神经网络都是通过不断地学习所提供的数据，尝试发现待求解的问题的模式。
也就是说，与待处理的问题无关，神经网络可以将数据直接作为原始数据，进行“端对端”的学习。

本章主要介绍神经网络的学习，不过在这之前，我们先来介绍一下机器学习中有关数据处理的一些注意事项。
机器学习中，一般将数据分为训练数据和测试数据两部分来进行学习和实验等。
首先，使用训练数据进行学习，寻找最优的参数；然后，使用测试数据评价训练得到的模型的实际能力。
为什么需要将数据分为训练数据和测试数据呢？
因为我们追求的是模型的泛化能力。
为了正确评价模型的泛化能力，就必须划分训练数据和测试数据。
另外，训练数据也可以称为监督数据。

泛化能力是指处理未被观察过的数据（不包含在训练数据中的数据）的能力。
获得泛化能力是机器学习的最终目标。
比如，在识别手写数字的问题中，泛化能力可能会被用在自动读取明信片的邮政编码的系统上。
此时，手写数字识别就必须具备较高的识别“某个人”写的字的能力。
注意这里不是“特定的某个人写的特定的文字”，而是“任意一个人写的任意文字”。
如果系统只能正确识别已有的训练数据，那有可能是只学习到了训练数据中的个人的习惯写法。

因此，仅仅用一个数据集去学习和评价参数，是无法进行正确评价的。
这样会导致可以顺利地处理某个数据集，但无法处理其他数据集的情况。
顺便说一下，只对某个数据集过度拟合的状态称为过拟合（over fitting）。
避免过拟合也是机器学习的一个重要课题。

"""


"""
如果有人问你现在有多幸福，你会如何回答呢？
一般的人可能会给出诸如“还可以吧”或者“不是那么幸福”等笼统的回答。
如果有人回答“我现在的幸福指数是 10.23”的话，可能会把人吓一跳吧。
因为他用一个数值指标来评判自己的幸福程度。
这里的幸福指数只是打个比方，实际上神经网络的学习也在做同样的事情。
神经网络的学习通过某个指标表示现在的状态。
然后，以这个指标为基准，寻找最优权重参数。
和刚刚那位以幸福指数为指引寻找“最优人生”的人一样，神经网络以某个指标为线索寻找最优权重参数。
神经网络的学习中所用的指标称为损失函数（loss function）。
这个损失函数可以使用任意函数，但一般用均方误差和交叉熵误差等。

损失函数是表示神经网络性能的“恶劣程度”的指标，
即当前的神经网络对监督数据在多大程度上不拟合，在多大程度上不一致。
以“性能的恶劣程度”为指标可能会使人感到不太自然，
但是如果给损失函数乘上一个负值，
就可以解释为“在多大程度上不坏”，即“性能有多好”。
并且，“使性能的恶劣程度达到最小”和“使性能的优良程度达到最大”是等价的，
不管是用“恶劣程度”还是“优良程度”，做的事情本质上都是一样的。

可以用作损失函数的函数有很多，其中最有名的是均方误差（mean squared error）。
均方误差如下式所示。
E = 1/2 * sum(math.pow(yk - tk))

这里，yk 是表示神经网络的输出，tk 表示监督数据，k 表示数据的维数。
比如，在 3.6 节手写数字识别的例子中，yk、tk 是由如下 10 个元素构成的数据。

数组元素的索引从第一个开始依次对应数字“0”“1”“2”......这里，神经网络的输出 y 是 softmax 函数的输出。
由于 softmax 函数的输出可以理解为概率，因此上例表示“0”的概率是 0.1，“1”的概率是 0.05，“2”的概率是 0.6等。
t 是监督数据，将正确解标签设为 1，其他均设为 0。这里，标签“2”为1，表示正确解是“2”。
将正确解标签表示为 1，其他标签表示为 0 的表示方法称为 one-hot 表示。



损失函数：
均方误差mean squared error：输出结果和监督数据越吻合，误差越小；
交叉熵误差cross entropy error：值是正确解标签所对应的输出结果决定的；
"""


import numpy as np
# 设2为正确解
y = [0.1, 0.005, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

"""
如式（4.1）所示，均方误差会计算神经网络的输出和正确解监督数据的各个元素之差的平方，再求总和。
现在，我们用 Python 来实现这个均方误差，实现方式如下所示。

这里，参数 y 和 t 是 NumPy 数组。
代码实现完全遵照式（4.1），因此不再具体说明。
现在，我们使用这个函数，来实际地计算一下。
"""
def mean_squared_error(y, t):
    """均方误差的实现：0.5是系数"""
    return 0.5 * np.sum((y - t)**2)


# 例1：“2”的概率最高的情况（0.6）
# print(mean_squared_error(np.array(y), np.array(t)))

# 例2：“7”的概率最高的情况（0.6）
# y = [0.1, 0.005, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
# print(mean_squared_error(np.array(y), np.array(t)))
"""
这里举了两个例子。
第一个例子中，正确解是“2”，神经网络的输出的最大值是“2”；
第二个例子中，正确解是“2”，神经网络的输出的最大值是“7”。
如实验结果所示，我们发现第一个例子的损失函数的值更小，和监督数据之间的误差较小。
也就是说，均方误差显示第一个例子的输出结果与监督数据更加吻合。

"""



"""
除了均方误差之外，交叉熵误差（cross entropy error）也经常被用作损失函数。
交叉熵误差如下式所示。
E = - sum(tk * math.log(yk))

这里，log 表示以e为底数的自然对数（math.log(x,e)）。
yk 是神经网络的输出，tk 是正确解标签。并且，tk 中只有正确解标签的索引为 1，其他均为 0（one-hot 表示）。
因此，式（4.2）实际上只计算对应正确解标签的输出的自然对数。
比如，假设正确解标签的索引是“2”，与之对应的神经网络的输出是 0.6，则交叉熵误差是-log 0.6 = 0.51；
若“2”对应的输出是 0.1，则交叉熵误差为 -log 0.1 =2.30。
也就是说，交叉熵误差的值是由正确解标签所对应的输出结果决定的。

自然对数的图像如图 4-3 所示。
图 4-3　自然对数 y = log x 的图像

如图 4-3 所示，x 等于 1 时，y 为 0；随着 x 向 0 靠近，y 逐渐变小。
因此，正确解标签对应的输出越大，式（4.2）的值越接近 0；
当输出为 1 时，交叉熵误差为 0。
此外，如果正确解标签对应的输出较小，则式（4.2）的值较大。

下面，我们来用代码实现交叉熵误差。


"""
def cross_entropy_error(y, t):
    """交叉熵损失的实现：log是ln，和式前有负号；delta 防止负无限大；"""
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

"""
这里，参数 y 和 t 是 NumPy 数组。
函数内部在计算 np.log 时，加上了一个微小值 delta。
这是因为，当出现 np.log(0) 时，np.log(0) 会变为负无限大的 -inf，这样一来就会导致后续计算无法进行。
作为保护性对策，添加一个微小值可以防止负无限大的发生。
下面，我们使用 cross_entropy_error(y, t) 进行一些简单的计算。
第一个例子中，正确解标签对应的输出为 0.6，此时的交叉熵误差大约为0.51。
第二个例子中，正确解标签对应的输出为 0.1 的低值，此时的交叉熵误差大约为 2.3。
由此可以看出，这些结果与我们前面讨论的内容是一致的。
"""
print(cross_entropy_error(np.array(y), np.array(t)))
# 7的概率最高的情况
y = [0.1, 0.005, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))
