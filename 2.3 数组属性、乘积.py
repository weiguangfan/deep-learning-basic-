"""
如果掌握了 NumPy 多维数组的运算，就可以高效地实现神经网络。

多维数组就是“数字的集合”，
数字排成一列的集合、排成长方形的集合、排成三维状或者（更加一般化的）N 维状的集合都称为多维数组。
数组的维数可以通过 np.dim() 函数获得。
此外，数组的形状可以通过实例变量 shape 获得。
在上面的例子中，A 是一维数组，由 4 个元素构成。
注意，这里的 A.shape 的结果是个元组（tuple）。
这是因为一维数组的情况下也要返回和多维数组的情况下一致的结果。
例如，二维数组时返回的是元组 (4,3)，三维数组时返回的是元组 (4,3,2)，因此一维数组时也同样以元组的形式返回结果。
下面我们就用 NumPy 来生成多维数组，先从前面介绍过的一维数组开始。
"""

import numpy as np
a = np.array([1, 2, 3, 4])
# print(a)
# print(np.ndim(a))
# print(a.shape)
# print(a.shape[0])

"""
下面我们来生成一个二维数组。
这里生成了一个 3 × 2 的数组 B。
3 × 2 的数组表示第一个维度有 3 个元素，第二个维度有 2 个元素。
另外，第一个维度对应第 0 维，第二个维度对应第1 维（Python 的索引从 0 开始）。
二维数组也称为矩阵（matrix）。
数组的横向排列称为行（row），纵向排列称为列（column）。
"""
b = np.array([[1, 2], [3, 4], [5, 6]])
# print(b)
# print(b.shape)
# print(np.ndim(b))
"""
下面，我们来介绍矩阵（二维数组）的乘积
如本例所示，矩阵的乘积是通过左边矩阵的行（横向）和右边矩阵的列（纵向）以对应元素的方式相乘后再求和而得到的。
并且，运算的结果保存为新的多维数组的元素。
比如，A 的第 1 行和 B 的第 1 列的乘积结果是新数组的第 1 行第 1 列的元素，A 的第 2 行和 B 的第 1 列的结果是新数组的第 2 行第 1 列的元素。
另外，在本书的数学标记中，矩阵将用黑斜体表示（比如，矩阵A ），以区别于单个元素的标量（比如，a 或 b）。

这里，A 和 B 都是 2 × 2 的矩阵，它们的乘积可以通过 NumPy 的np.dot() 函数计算（乘积也称为点积）。
np.dot() 接收两个 NumPy 数组作为参数，并返回数组的乘积。
这里要注意的是，np.dot(A, B) 和 np.dot(B, A) 的值可能不一样。
和一般的运算（+ 或 * 等）不同，矩阵的乘积运算中，操作数



"""
c = np.array([[1, 2], [3, 4]])
# print(c.shape)
d = np.array([[5, 6], [7, 8]])
# print(d.shape)
# print(np.dot(c, d))
"""
这里介绍的是计算 2 × 2 形状的矩阵的乘积的例子，其他形状的矩阵的乘积也可以用相同的方法来计算。
比如，2 × 3 的矩阵和 3 × 2 的矩阵的乘积可按如下形式用 Python 来实现。
2 × 3 的矩阵  和 3 × 2 的矩阵  的乘积可按以上方式实现。这里需要注意的是矩阵的形状（shape）。具体地讲，矩阵  的第 1 维的元素个数（列数）必须和矩阵  的第 0 维的元素个数（行数）相等。
在上面的例子中，矩阵 A 的形状是 2 × 3，矩阵 B 的形状是 3 × 2，矩阵 A 的第 1 维的元素个数（3）和矩阵B  的第 0 维的元素个数（3）相等。
如果这两个值不相等，则无法计算矩阵的乘积
在矩阵的乘积运算中，对应维度的元素个数要保持一致
在多维数组的乘积运算中，必须使两个矩阵中的对应维度的元素个数一致，这一点很重要。

3 × 2 的矩阵 A 和 2 × 4 的矩阵 B 的乘积运算生成了 3× 4 的矩阵 。
如图所示，矩阵 A 和矩阵 B 的对应维度的元素个数必须保持一致。
此外，还有一点很重要，就是运算结果的矩阵 C 的形状是由矩阵 A 的行数和矩阵 B 的列数构成的。
"""
e = np.array([[1, 2, 3], [4, 5, 6]])
# print(e.shape)
f = np.array([[1, 2], [3, 4], [5, 6]])
# print(f.shape)
# print(np.dot(e, f))
"""
当 A 是二维矩阵、B 是一维数组时，如图 3-13 所示，对应维度的元素个数要保持一致的原则依然成立。
A 是二维矩阵、B 是一维数组时，也要保持对应维度的元素个数一致
"""
g = np.array([[1, 2], [3, 4], [5, 6]])
# print(g.shape)
h = np.array([7, 8])
# print(h.shape)
# print(np.dot(g, h))
"""
下面我们使用 NumPy 矩阵来实现神经网络。
这里我们以图 3-14 中的简单神经网络为对象。
这个神经网络省略了偏置和激活函数，只有权重。
通过矩阵的乘积进行神经网络的运算
实现该神经网络时，要注意 X、W、Y 的形状，特别是 X 和 W 的对应维度的元素个数是否一致，这一点很重要。
如上所示，使用 np.dot（多维数组的点积），可以一次性计算出  的结果。
这意味着，即便 Y 的元素个数为 100 或 1000，也可以通过一次运算就计算出结果！
如果不使用 np.dot，就必须单独计算 Y 的每一个元素（或者说必须使用for 语句），非常麻烦。
因此，通过矩阵的乘积一次性完成计算的技巧，在实现的层面上可以说是非常重要的。
"""
i = np.array([1, 2])
print(i.shape)
j = np.array([[1, 3, 5], [2, 4, 6]])
print(j.shape)
print(np.dot(i, j))
