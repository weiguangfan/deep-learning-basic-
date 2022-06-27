"""
神经网络可以用在分类和回归问题撒花姑娘，需要根据情况改变输出层的激活函数；
一般，回归问题用恒等函数，分类问题用Softmax函数；
恒等函数将输入按原样输出；
一般，神经网络只把输出值最大的神经元所对应的类别作为识别的结果；
因此，神经网络在进行分类时，输出层的softmax函数可以省略；
输出层的神经元数量需要根据待解决的问题来决定；
对于分类问题，输出层的神经元数量一般设定为类别的数量；
"""
# 按照公式（3.10）实现
import numpy as np
# a = np.array([0.3, 2.9, 4.0])
# exp_a = np.exp(a)
# print(exp_a)
# sum_exp_a = np.sum(exp_a)
# print(sum_exp_a)
# y = exp_a / sum_exp_a
# print(y)

# 封装成函数（缺陷：溢出问题）
# def soft_max(a):
#     exp_a = np.exp(a)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#     return y


"""
为避免溢出问题，改写（3.10），分子分母同时加上一个常数，结果不变；
一般，常数会使用输入信号的最大值；
"""
# a = np.array([1010, 1000, 990])
# print((np.exp(a) / np.sum(np.exp(a))))  #softmax函数的运算：[nan nan nan] 输出没有被正确计算
# c = np.max(a)
# print(c)  # 1010
# print((a - c))  # [  0 -10 -20]
# print((np.exp(a - c) / np.sum(np.exp(a - c))))  # [9.99954600e-01 4.53978686e-05 2.06106005e-09] 输出被正确计算


# 因此进化为可以防止溢出的函数
def soft_max(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


# softmax函数特征：输出是0.0到1.0之间的实数，输出值总和是1，输出解释为概率；
# a中各元素的大小关系和y中各元素的大小关系并没有改变；
a = np.array([0.3, 2.9, 4.0])
y = soft_max(a)
print(y)  # [0.01821127 0.24519181 0.73659691]
# 输出值总和是1
print(np.sum(y)) # 1.0
