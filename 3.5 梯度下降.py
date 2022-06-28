"""
梯度指示的方向是各点处的函数值减小最多的方向；
最优参数是指损失函数取最小值时的参数；
通过反转损失函数的符号，求最小值和求最大值的问题会变成相同的问题；
函数的极小值、最小值以及被称为鞍点的地方，梯度为0；
鞍点是从某个方向看是极大值，从另一个方向看则是极小值的点；
虽然梯度的方向并不一定指向最小值，但沿着它的方向，能够最大限度地减少函数的值；
梯度法gradient method:通过不断地沿梯度方向前进，逐渐减小函数值的过程；
寻找最小值的梯度法称为梯度下降法（gradient descent method);
寻找最大值的梯度法称为梯度上升法（gradient ascent method);
"""
import numpy as np


def function_2(x):
    """原函数"""
    print("x: ", x)
    return x[0]**2 + x[1]**2


def numerical_gradient(f, x):
    """遍历数组的每个元素，进行数值微分"""
    h = 1e-4
    grad = np.zeros_like(x)
    print("grad: ", grad)
    for idx in range(x.size):
        print("idx: ", idx)
        tmp_val = x[idx]
        print('i:tmp_val: ', idx, tmp_val)
        x[idx] = tmp_val + h
        fxh1 = f(x)
        print("idx:fxh1: ", idx, fxh1)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2)/(2*h)
        print("grad: ", grad)
        x[idx] = tmp_val
    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """梯度下降法："""
    x = init_x
    # 反复执行更新的式子
    for i in range(step_num):  # step_num梯度法的重复次数
        print("###" * 10)
        print("i: ", i)
        # 返回偏导向量组成的数组
        grad = numerical_gradient(f, x)
        # 更新一次的式子，lr学习率过大过小，都无法抵达一个好的位置
        x -= lr * grad
    return x


# 初始值
init_x = np.array([-3.0, 4.0])

# 使用梯度法寻找最小值，没怎么更新就结束了
# print(gradient_descent(function_2, init_x, lr=0.1, step_num=100))

# 学习率过大，lr=10.0，会发散成一个很大的值
# print(gradient_descent(function_2, init_x, lr=10, step_num=100))


# 学习率过小，lr=1e-10，
print(gradient_descent(function_2, init_x, lr=1e-10, step_num=100))

