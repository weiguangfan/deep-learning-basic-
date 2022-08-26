"""
感知机接收多个输入信号，输出一个信号。
x1,x2 是输入信号，y 是输出信号，w1,w2是权重（w 是 weight 的首字母）。
输入信号被送往神经元时，会被分别乘以固定的权重（w1,w2）。
神经元会计算传送过来的信号的总和，只有当这个总和超过了某个界限值时，才会输出 1。这也称为“神经元被激活”。
这里将这个界限值称为阈值，用符号 θ 表示。
有两个输入的感知机,把上述内容用数学式来表示
y = 0 (w1x1 + w2x2 <= θ),
y = 1 (w1x1 + w2x2 > θ)
感知机的多个输入信号都有各自固有的权重，这些权重发挥着控制各个信号的重要性的作用。
也就是说，权重越大，对应该权重的信号的重要性就越高。


与门（AND gate）。
与门是有两个输入和一个输出的门电路
与门仅在两个输入均为 1 时输出 1，其他时候则输出 0。
真值表：
0 0 0
0 1 0
1 0 0
1 1 1
下面考虑用感知机来表示这个与门。
需要做的就是确定能满足与门的真值表的 w1,w2, 的值。
满足与门的条件的参数的选择方法有无数多个。
例如：
(w1,w2,θ) = (0.5,0.5,0.7)
(w1,w2,θ) = (0.5,0.5,0.8)
(w1,w2,θ) = (1.0,1.0,1.0)
设定这样的参数后，仅当 x1 和 x2 同时为 1 时，信号的加权总和才会超过给定的阈值θ。
"""

def and_gate(x1, x2):
    """
    用 Python 来实现与门逻辑电路。
    :param x1:
    :param x2:
    :return:
    """
    # 在函数内初始化参数 w1、w2、theta，当输入的加权总和超过阈值时返回 1，否则返回 0。
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


# 确认与门真值表
print(and_gate(0, 0))
print(and_gate(1, 0))
print(and_gate(0, 1))
print(and_gate(1, 1))

"""
numpy 改写 与门
"""
import numpy as np

x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7
# print(w*x)
# print(np.sum(w * x))
# print((np.sum(w * x) + b))

"""
用权重和偏置的实现
"""


# def and_gate(x1, x2):
#     x = np.array([x1, x2])
#     w = np.array([0.5, 0.5])
#     b = -0.7
#     tmp = np.sum(w * x) + b
#     if tmp <= 0:
#         return 0
#     else:
#         return 1

# 确认与门真值表
# print(and_gate(0, 0))
# print(and_gate(1, 0))
# print(and_gate(0, 1))
# print(and_gate(1, 1))

"""
与非门（NAND gate）。
NAND 是 Not AND 的意思，与非门就是颠倒了与门的输出。
仅当 x1 和 x2 同时为 1 时输出 0，其他时候则输出 1。
真值表：
0 0 1
0 1 1
1 0 1
1 1 0
例如：(w1,w2,θ) = (-0.5,-0.5,-0.7)
实际上，只要把实现与门的参数值的符号取反，就可以实现与非门。
"""
def not_and_gate(x1, x2):
    """
    用 Python 来实现与非门逻辑电路。
    :param x1:
    :param x2:
    :return:
    """
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5]) # 仅权重和偏置与AND不同！
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1
# 与非门真值表
# print(not_and_gate(0, 0))
# print(not_and_gate(1, 0))
# print(not_and_gate(0, 1))
# print(not_and_gate(1, 1))



"""
或门是“只要有一个输入信号是 1，输出就为 1”的逻辑电路。
真值表：
0 0 0
0 1 1
1 0 1
1 1 1
这里决定感知机参数的并不是计算机，而是我们人。
我们看着真值表这种“训练数据”，人工考虑（想到）了参数的值。
而机器学习的课题就是将这个决定参数值的工作交由计算机自动进行。
学习是确定合适的参数的过程，而人要做的是思考感知机的构造（模型），并把训练数据交给计算机。
如上所示，我们已经知道使用感知机可以表示与门、与非门、或门的逻辑电路。
这里重要的一点是：与门、与非门、或门的感知机构造是一样的。
实际上，3个门电路只有参数的值（权重和阈值）不同。
也就是说，相同构造的感知机，只需通过适当地调整参数的值，就可以像“变色龙演员”表演不同的角色一样，变身为与门、与非门、或门。
"""


def or_gate(x1, x2):
    """
    用 Python 来实现或门逻辑电路。
    :param x1: 输入1
    :param x2: 输入2
    :return: 输出1
    """
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5]) # 仅权重和偏置与AND不同！
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# 或门真值表
# print(or_gate(0, 0))
# print(or_gate(1, 0))
# print(or_gate(0, 1))
# print(or_gate(1, 1))

"""
用与非门、或门、与门组合异或门
"""


def xor_gate(x1, x2):
    s1 = not_and_gate(x1, x2)
    print(s1)
    s2 = or_gate(x1, x2)
    print(s2)
    y = and_gate(s1, s2)
    return y


# print(xor_gate(0, 0))
# print(xor_gate(0, 1))
# print(xor_gate(1, 0))
# print(xor_gate(1, 1))
