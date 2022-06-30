"""
input:c,h,w
filter:c,fh,fw
output:1,oh,ow
input横向展开为一列：c * fh * fw 是一列的元素数；oh * ow 是行数；形状（oh * ow, c*fh*fw）
filter纵向展开为一列：c * fh * fw 是一列的元素数；fn 是列数；形状（oh * ow, fn）
展开项矩阵相乘，输出形状（oh*ow,fn）
展开项的第二维的元素个数是filter的元素个数的总和
"""
import sys, os
sys.path.append(os.pardir)
from common.util import im2col
import numpy as np

x1 = np.random.randn(1,3,7,7)
col1 = im2col(x1,5,5,stride=1,pad=0)
print(col1.shape)

x2 = np.random.randn(10,3,7,7)
col2 = im2col(x2,5,5,stride=1,pad=0)
print(col2.shape)

