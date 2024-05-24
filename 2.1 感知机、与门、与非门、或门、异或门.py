"""
感知机接收多个输入信号，输出一个信号。
x1,x2 是输入信号，y 是输出信号，w1,w2是权重（w 是 weight 的首字母）。
输入信号被送往神经元时，会被分别乘以固定的权重（w1,w2）。
神经元会计算传送过来的信号的总和，只有当这个总和超过了某个界限值时，才会输出 1。这也称为“神经元被激活”。
这里将这个界限值称为阈值，用符号 θ 表示。
有两个输入的感知机,把上述内容用数学式来表示
式子2.1
y = 0 (w1x1 + w2x2 <= θ),
y = 1 (w1x1 + w2x2 > θ)

改为另外一种形式，θ = -b:
此处，b 称为偏置，w1 和 w2 称为权重。
如下所示，感知机会计算输入信号和权重的乘积，然后加上偏置，如果这个值大于 0 则输出 1，否则输出 0。
式子2.2
y = 0 (b + w1x1 + w2x2 <= 0),
y = 1 (b + w1x1 + w2x2 > 0)


感知机的多个输入信号都有各自固有的权重，这些权重发挥着控制各个信号的重要性的作用。
也就是说，权重越大，对应该权重的信号的重要性就越高。

偏置是调整神经元被激活的容易程度（输出信号为 1 的程度）的参数。
比如，若 b 为 -0.1，则只要输入信号的加权总和超过 0.1，神经元就会被激活。
但是如果 b 为 -20.0，则输入信号的加权总和必须超过 20.0，神经元才会被激活。
像这样，偏置的值决定了神经元被激活的容易程度。
另外，这里我们将 w1 和 w2 称为权重，将 b 称为偏置，但是根据上下文，有时也会将 b、w1、w2 这些参数统称为权重。
实际上，在式 b + w1x1 + w2x2 的计算中，当输入 x1 和 x2 为 0 时，只输出偏置的值。

这里决定感知机参数的并不是计算机，而是我们人。
我们看着真值表这种“训练数据”，人工考虑（想到）了参数的值。
而机器学习的课题就是将这个决定参数值的工作交由计算机自动进行。
学习是确定合适的参数的过程，而人要做的是思考感知机的构造（模型），并把训练数据交给计算机。
如上所示，我们已经知道使用感知机可以表示与门、与非门、或门的逻辑电路。
这里重要的一点是：与门、与非门、或门的感知机构造是一样的。
实际上，3个门电路只有参数的值（权重和阈值）不同。
也就是说，相同构造的感知机，只需通过适当地调整参数的值，就可以像“变色龙演员”表演不同的角色一样，变身为与门、与非门、或门。

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

# def and_gate(x1, x2):
#     """
#     式子2.1
#     用 Python 来实现与门逻辑电路。
#     :param x1: 输入参数1
#     :param x2: 输入参数2
#     :return: 返回0或1
#     """
#     # 在函数内初始化参数 w1、w2、theta，当输入的加权总和超过阈值时返回 1，否则返回 0。
#     w1, w2, theta = 0.5, 0.5, 0.7
#     tmp = x1*w1 + x2*w2
#     if tmp <= theta:
#         return 0
#     elif tmp > theta:
#         return 1


# 确认与门真值表
# print(and_gate(0, 0))
# print(and_gate(1, 0))
# print(and_gate(0, 1))
# print(and_gate(1, 1))

"""
使用 NumPy，实现感知机。
numpy 改写 与门
如下所示，在 NumPy 数组的乘法运算中，
当两个数组的元素个数相同时，各个元素分别相乘，因此 w*x 的结果就是它们的各个元素分别相乘（[0, 1] *[0.5, 0.5] => [0, 0.5]）。
之后，np.sum(w*x) 再计算相乘后的各个元素的总和。
最后再把偏置加到这个加权总和上，就完成了计算。
"""
import numpy as np


# x = np.array([0, 1])  # 输入
# w = np.array([0.5, 0.5])  # 权重
# b = -0.7  # 偏置
# print(w * x)
# print(np.sum(w * x))
# print((np.sum(w * x) + b))


def and_gate(x1, x2):
    """
    式子2.2
    使用权重和偏置，可以像下面这样实现与门。
    :param x1: 输入1
    :param x2: 输入2
    :return: 返回0或1
    """
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


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
    :param x1: 输入1
    :param x2: 输入2
    :return: 返回0或1
    """
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])  # 仅权重和偏置与AND不同！
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1


# 确认与非门真值表
print(not_and_gate(0, 0))
print(not_and_gate(1, 0))
print(not_and_gate(0, 1))
print(not_and_gate(1, 1))

"""
或门是“只要有一个输入信号是 1，输出就为 1”的逻辑电路。
真值表：
0 0 0
0 1 1
1 0 1
1 1 1
例如：
(w1,w2,θ) = (0.5,0.5,0.2)
(w1,w2,θ) = (1.0,1.0,0.5)
"""

# def or_gate(x1, x2):
#     """
#     用 Python 来实现或门逻辑电路。
#     :param x1: 输入1
#     :param x2: 输入2
#     :return: 输出1
#     """
#     x = np.array([x1, x2])
#     w = np.array([0.5, 0.5])  # 仅权重和偏置与AND不同！
#     b = -0.2
#     tmp = np.sum(w*x) + b
#     if tmp <= 0:
#         return 0
#     else:
#         return 1

# 确认或门真值表
# print(or_gate(0, 0))
# print(or_gate(1, 0))
# print(or_gate(0, 1))
# print(or_gate(1, 1))


"""
使用感知机可以实现与门、与非门、或门三种逻辑电路。
现在我们来考虑一下异或门（XOR gate）。
异或门也被称为逻辑异或电路。
仅当 x1 或 x2 中的一方为 1时，才会输出 1（“异或”是拒绝其他的意思）。
真值表：
0 0 0
0 1 1
1 0 1
1 1 0
用前面介绍的感知机是无法实现这个异或门的。
为什么用感知机可以实现与门、或门，却无法实现异或门呢？
下面我们尝试通过画图来思考其中的原因。

试着将或门的动作形象化。
或门的情况下，当权重参数:
(w1,w2,θ) = (1.0,1.0,0.5)
(b,w1,w2) = (-0.5,1.0,1.0)
感知机可用下面的式子表示:
y = 0 (-0.5 + x1 + x2 <= 0),
y = 1 (-0.5 + x1 + x2 > 0)
感知机会生成由直线 -0.5 + x1 + x2 分割开的两个空间。
其中一个空间输出 1，另一个空间输出 0，
○ 表示 0，△ 表示 1。
如果想制作或门，需要用直线将空间中的○和△分开。

有时候，空间中的○和△无法用一条直线分开，但是如果将“直线”这个限制条件去掉，就可以实现了。
使用曲线可以分开○和△。
这样弯曲的曲线无法用感知机表示。
感知机的局限性就在于它只能表示由一条直线分割的空间。
感知机的局限性，严格地讲，应该是“单层感知机无法表示异或门”或者“单层感知机无法分离非线性空间”。
另外，由这样的曲线分割而成的空间称为非线性空间，由直线分割而成的空间称为线性空间。
线性、非线性这两个术语在机器学习领域很常见。

感知机不能表示异或门让人深感遗憾，但也无需悲观。
实际上，感知机的绝妙之处在于它可以“叠加层”（通过叠加层来表示异或门是本节的要点）。
异或门的制作方法有很多，其中之一就是组合我们前面做好的与门、与非门、或门进行配置。
与门、与非门、或门用符号表示。
与非门前端的○表示反转输出的意思。
通过组合感知机（叠加层）就可以实现异或门。
异或门可以通过已下的配置来实现。
这里，x1 和 x2 表示输入信号，y 表示输出信号。 
x1 和 x2 是与非门和或门的输入，而与非门和或门的输出则是与门的输入。
我们来确认一下上面的配置是否真正实现了异或门。
这里，把 s1 作为与非门的输出，把 s2 作为或门的输出，填入真值表中。
观察 x1、x2、y，可以发现确实符合异或门的输出。
真值表：
x1  x2  s1   s2  y
0   0   1    0   0
0   1   1    1   1
1   0   1    1   1
1   1   0    1   0

用与非门、或门、与门组合异或门
"""

# def xor_gate(x1, x2):
#     """
#     用 Python 来实现异或门。
#     :param x1:
#     :param x2:
#     :return:
#     """
#     s1 = not_and_gate(x1, x2)
#     s2 = or_gate(x1, x2)
#     y = and_gate(s1, s2)
#     return y


# 确认异或门真值表
# print(xor_gate(0, 0))
# print(xor_gate(0, 1))
# print(xor_gate(1, 0))
# print(xor_gate(1, 1))


"""
试着用感知机的表示方法（明确地显示神经元）来表示这个异或门，
异或门是一种多层结构的神经网络。
这里，将最左边的一列称为第 0 层，中间的一列称为第 1 层，最右边的一列称为第 2 层。
x1,x2构成第0层，s1,s2构成第1层，y构成第2层。

实际上，与门、或门是单层感知机，而异或门是 2 层感知机。
叠加了多层的感知机也称为多层感知机（multi-layered perceptron）。

异或门感知机总共由 3 层构成，但是因为拥有权重的层实质上只有 2 层（第 0 层和第 1 层之间，第 1 层和第 2 层之间），所以称为“2 层感知机”。
不过，有的文献认为的异或门感知机是由 3 层构成的，因而将其称为“3 层感知机”。

在异或门的 2 层感知机中，
先在第 0 层和第 1 层的神经元之间进行信号的传送和接收，
然后在第 1 层和第 2 层之间进行信号的传送和接收。
第 0 层的两个神经元接收输入信号，并将信号发送至第 1 层的神经元。
第 1 层的神经元将信号发送至第 2 层的神经元，第 2 层的神经元输出y。

通过这样的结构（2 层结构），感知机得以实现异或门。
这可以解释为“单层感知机无法表示的东西，通过增加一层就可以解决”。
也就是说，通过叠加层（加深层），感知机能进行更加灵活的表示。

如果通过组合与非门可以实现计算机的话，那么通过组合感知机也可以表示计算机（感知机的组合可以通过叠加了多层的单层感知机来表示）。
《计算机系统要素：从零开始构建现代计算机》

理论上可以说 2 层感知机就能构建计算机。
这是因为，已有研究证明，2 层感知机（严格地说是激活函数使用了非线性的 sigmoid 函数的感知机，具体请参照下一章）可以表示任意函数。
但是，使用 2 层感知机的构造，通过设定合适的权重来构建计算机是一件非常累人的事情。
实际上，在用与非门等低层的元件构建计算机的情况下，
分阶段地制作所需的零件（模块）会比较自然，即先实现与门和或门，然后实现半加器和全加器，
接着实现算数逻辑单元（ALU），然后实现 CPU。

感知机通过叠加层能够进行非线性的表示，理论上还可以表示计算机进行的处理。


本章所学的内容
感知机是具有输入和输出的算法。
给定一个输入后，将输出一个既定的值。
感知机将权重和偏置设定为参数。
使用感知机可以表示与门和或门等逻辑电路。
异或门无法通过单层感知机来表示。
使用2层感知机可以表示异或门。
单层感知机只能表示线性空间，而多层感知机可以表示非线性空间。
多层感知机（在理论上）可以表示计算机。

"""
