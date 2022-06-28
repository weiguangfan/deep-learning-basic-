"""
偏导数：有多个变量的函数的导数
"""

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h))/(2 * h)


# x为numpy数组，等效于np.sum(x**2)
def function_2(x):
    """含有两个变量的函数"""
    return x[0]**2 + x[1]**2


# 求x0=3,x1=4时，关于x0的偏导
def function_tmp1(x0):
    """原函数中设x1=4.0，定义一个只有一个变量的函数"""
    return x0*x0 + 4.0**2.0


# 对x0求导时
print(numerical_diff(function_tmp1, 3.0))  # 6.00000000000378


# 求x0=3,x1=4时，关于x1的偏导
def function_tmp2(x1):
    """原函数中设x0=3.0，定义一个只有一个变量的函数"""
    return 3.0**2.0 + x1*x1


# 对x1求导时
print(numerical_diff(function_tmp2, 4.0))  # 7.999999999999119

