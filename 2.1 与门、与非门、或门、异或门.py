"""
与门;
与非门；
或门；
"""


# def and_gate(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.7
#     tmp = x1*w1 + x2*w2
#     if tmp <= theta:
#         return 0
#     elif tmp > theta:
#         return 1


# print(and_gate(0, 0))
# print(and_gate(1, 0))
# print(and_gate(0, 1))
# print(and_gate(1, 1))

"""
numpy 改写 与门
"""
import numpy as np

# x = np.array([0, 1])
# w = np.array([0.5, 0.5])
# b = -0.7
# print(w*x)
# print(np.sum(w * x))
# print((np.sum(w * x) + b))


def and_gate(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def not_and_gate(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1


def or_gate(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


"""
用与非门、非门、与门组合异或门
"""

def xor_gate(x1, x2):
    s1 = not_and_gate(x1, x2)
    s2 = or_gate(x1, x2)
    y = and_gate(s1, s2)
    return y


print(xor_gate(0, 0))
print(xor_gate(0, 1))
print(xor_gate(1, 0))
print(xor_gate(0, 0))

