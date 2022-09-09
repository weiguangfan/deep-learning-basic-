# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x 


def tangent_line(f, x):
    # 计算固定点的斜率
    d = numerical_diff(f, x)
    print(d)
    # 点斜式方程：计算该点的y值
    y = f(x) - d*x
    # 返回该点的直线方程
    return lambda t: d*t + y
     
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

# 返回在点x=5处的直线方程
tf1 = tangent_line(function_1, 5)
# 返回在点x=10处的直线方程
tf2 = tangent_line(function_1, 10)
# 计算直线方程的每一点
y2 = tf1(x)
y3 = tf2(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.plot(x, y3)
plt.show()
