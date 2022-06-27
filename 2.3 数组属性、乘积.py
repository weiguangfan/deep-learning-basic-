"""
多维数组就是“数字的集合”，
数字排成一列的集合、排成长方形的集合、排成三维状或者（更加一般化的）N 维状的集合都称为多维数组。
数组的维数可以通过 np.dim() 函数获得。
此外，数组的形状可以通过实例变量 shape 获得。
一维数组的情况下也要返回和多维数组的情况下一致的结果，结果是个元组（tuple）。
二维数组第一个维度对应第 0 维，第二个维度对应第1 维
二维数组也称为矩阵（matrix）。
数组的横向排列称为行（row），纵向排列称为列（column）。
矩阵1和矩阵2的对应维度的元素个数必须保持一致。
此外，还有一点很重要，就是运算结果的矩阵3的形状是由矩阵1的行数和矩阵2的列数构成的。
"""

import numpy as np
a = np.array([1, 2, 3, 4])
# print(a)
# print(np.ndim(a))
# print(a.shape)
# print(a.shape[0])

b = np.array([[1, 2], [3, 4], [5, 6]])
# print(b)
# print(b.shape)
# print(np.ndim(b))

c = np.array([[1, 2], [3, 4]])
# print(c.shape)
d = np.array([[5, 6], [7, 8]])
# print(d.shape)
# print(np.dot(c, d))

e = np.array([[1, 2, 3], [4, 5, 6]])
# print(e.shape)
f = np.array([[1, 2], [3, 4], [5, 6]])
# print(f.shape)
# print(np.dot(e, f))

g = np.array([[1, 2], [3, 4], [5, 6]])
# print(g.shape)
h = np.array([7, 8])
# print(h.shape)
# print(np.dot(g, h))

i = np.array([1, 2])
print(i.shape)
j = np.array([[1, 3, 5], [2, 4, 6]])
print(j.shape)
print(np.dot(i, j))
