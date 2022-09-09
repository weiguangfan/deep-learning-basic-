# coding: utf-8
# cf.http://d.hatena.ne.jp/white_wheels/20100327/p3
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def _numerical_gradient_no_batch(f, x):
    """
    数值微分法
    :param f:
    :param x:
    :return:
    """
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val  # 値を元に戻す
        
    return grad


def numerical_gradient(f, X):
    """
    计算梯度
    :param f:
    :param X:
    :return:
    """
    # 如果是1维
    if X.ndim == 1:
        # 直接调用数值微分法，返回梯度值
        return _numerical_gradient_no_batch(f, X)
    # 如果是多维
    else:
        # 生成与x形状相同的数组
        grad = np.zeros_like(X)
        # 遍历元素
        for idx, x in enumerate(X):
            # 直接调用数值微分法，返回梯度值
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad


def function_2(x):
    """
    原函数
    :param x:
    :return:
    """
    # 如果为1维
    if x.ndim == 1:
        return np.sum(x**2)
    # 如果为多维
    else:
        return np.sum(x**2, axis=1)


def tangent_line(f, x):
    """
    返回点斜式方程
    :param f:
    :param x:
    :return:
    """
    # 计算固定点的斜率
    d = numerical_gradient(f, x)
    print(d)
    # 点斜式方程：计算该点的y值
    y = f(x) - d*x
    # 返回该点的直线方程
    return lambda t: d*t + y


if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)
    
    X = X.flatten()
    Y = Y.flatten()

    grad = numerical_gradient(function_2, np.array([X, Y]).T).T

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.draw()
    plt.show()
