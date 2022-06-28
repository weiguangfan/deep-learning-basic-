"""
梯度gradient:由全部变量的偏导数汇总而成的向量；
"""
import numpy as np

def function_2(x):
    """原函数"""
    print("x: ", x)
    return x[0]**2 + x[1]**2


def numerical_gradient(f, x):
    """梯度实现：对每个元素进行数值微分"""
    h = 1e-4
    grad = np.zeros_like(x)
    print("grad: ", grad)
    # 遍历数组的每个元素
    for idx in range(x.size):
        print("idx: ", idx)
        # 取当前下标对应的值
        tmp_val = x[idx]
        print('i:tmp_val: ', idx, tmp_val)
        x[idx] = tmp_val + h
        # 固定其他下标的值，当前下标的值 + h，并求函数值
        fxh1 = f(x)
        print("idx:fxh1: ", idx, fxh1)
        x[idx] = tmp_val - h
        # 固定其他下标的值，当前下标的值 - h，并求函数值
        fxh2 = f(x)
        print("idx:fxh2: ", idx, fxh2)
        # 进行数值微分，并保存为字典
        grad[idx] = (fxh1 - fxh2)/(2*h)
        print("idx:grad: ", idx, grad)
        # 回复当前下标对应的值
        x[idx] = tmp_val
    return grad


# 执行函数，求点（3.0,4.0）梯度
print(numerical_gradient(function_2, np.array([3.0, 4.0])))  # [6. 8.]

# 执行函数，求点（0.0,2.0）梯度
print(numerical_gradient(function_2, np.array([0.0, 2.0])))  # [0. 4.]

# 执行函数，求点（3.0,0.0）梯度
print(numerical_gradient(function_2, np.array([3.0, 0.0])))  # [6. 0.]
