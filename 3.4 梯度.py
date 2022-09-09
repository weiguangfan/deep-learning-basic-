"""
在刚才的例子中，我们按变量分别计算了 x0 和 x1 的偏导数。
现在，我们希望一起计算 x0 和 x1 的偏导数。
比如，我们来考虑求x0=3,x1=4  时 (x0,x1) 的偏导数 (df/dx0,df/dx1) 。
另外，像 (df/dx0,df/dx1) 这样的由全部变量的偏导数汇总而成的向量称为梯度（gradient）。
梯度可以像下面这样来实现。

"""
import numpy as np

def function_2(x):
    """原函数"""
    print("x: ", x)
    return x[0]**2 + x[1]**2


def numerical_gradient(f, x):
    """梯度实现：对每个元素进行数值微分"""
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # 生成和x形状相同的数组
    print("grad: ", grad)
    # 遍历数组的每个元素
    for idx in range(x.size):
        print("idx: ", idx)
        # 取当前下标对应的值
        tmp_val = x[idx]
        print('i:tmp_val: ', idx, tmp_val)
        # f(x + h) 的计算
        x[idx] = tmp_val + h
        # 固定其他下标的值，当前下标的值 + h，并求函数值
        fxh1 = f(x)
        print("idx:fxh1: ", idx, fxh1)
        # f(x - h) 的计算
        x[idx] = tmp_val - h
        # 固定其他下标的值，当前下标的值 - h，并求函数值
        fxh2 = f(x)
        print("idx:fxh2: ", idx, fxh2)
        # 进行数值微分，并保存为字典
        grad[idx] = (fxh1 - fxh2)/(2*h)
        print("idx:grad: ", idx, grad)
        # 回复当前下标对应的值
        x[idx] = tmp_val  # 还原值
    return grad
"""
函数 numerical_gradient(f, x) 的实现看上去有些复杂，但它执行的处理和求单变量的数值微分基本没有区别。
需要补充说明一下的是，np.zeros_like(x)会生成一个形状和 x 相同、所有元素都为 0 的数组。

函数 numerical_gradient(f, x) 中，参数 f 为函数，x 为 NumPy 数组，该函数对 NumPy 数组 x 的各个元素求数值微分。
现在，我们用这个函数实际计算一下梯度。
这里我们求点 (3, 4)、(0, 2)、(3, 0) 处的梯度。

"""

# 执行函数，求点（3.0,4.0）梯度
print(numerical_gradient(function_2, np.array([3.0, 4.0])))  # [6. 8.]
"""
实际上，虽然求到的值是 [6.0000000000037801,7.9999999999991189]，
但实际输出的是 [6., 8.]。
这是因为在输出 NumPy数组时，数值会被改成“易读”的形式。

"""
# 执行函数，求点（0.0,2.0）梯度
print(numerical_gradient(function_2, np.array([0.0, 2.0])))  # [0. 4.]

# 执行函数，求点（3.0,0.0）梯度
print(numerical_gradient(function_2, np.array([3.0, 0.0])))  # [6. 0.]
"""
像这样，我们可以计算 (x0,x1) 在各点处的梯度。
上例中，点 (3, 4) 处的梯度是 (6, 8)、点 (0, 2) 处的梯度是 (0, 4)、点 (3, 0) 处的梯度是 (6,0)。
这个梯度意味着什么呢？
为了更好地理解，我们把 f(x0 + x1) = math.pow(x0) + math.pow(x1) 的梯度画在图上。
不过，这里我们画的是元素值为负梯度的向量（源代码在ch04/gradient_2d.py 中）。

后面我们将会看到，负梯度方向是梯度法中变量的更新方向。

图 4-9　f(x0 + x1) = math.pow(x0) + math.pow(x1) 的梯度

如图 4-9 所示，f(x0 + x1) = math.pow(x0) + math.pow(x1) 的梯度呈现为有向向量（箭头）。
观察图4-9，我们发现梯度指向函数 f(x0,x1) 的“最低处”（最小值），就像指南针一样，所有的箭头都指向同一点。
其次，我们发现离“最低处”越远，箭头越大。

虽然图 4-9 中的梯度指向了最低处，但并非任何时候都这样。
实际上，梯度会指向各点处的函数值降低的方向。
更严格地讲，梯度指示的方向是各点处的函数值减小最多的方向 。
这是一个非常重要的性质，请一定牢记！

高等数学告诉我们，方向导数 = cos(θ) × 梯度（θ 是方向导数的方向与梯度方向的夹角）。
因此，所有的下降方向中，梯度方向下降最多。


"""