"""
本节将用 Python 实现前面的购买苹果的例子。
这里，我们把要实现的计算图的乘法节点称为“乘法层”（MulLayer），加法节点称为“加法层”（AddLayer）。

下一节，我们将把构建神经网络的“层”实现为一个类。
这里所说的“层”是神经网络中功能的单位。
比如，负责 sigmoid 函数的 Sigmoid、负责矩阵乘积的 Affine 等，都以层为单位进行实现。
因此，这里也以层为单位来实现乘法节点和加法节点。

层的实现中有两个共通的方法（接口）forward() 和backward()。
forward() 对应正向传播，backward() 对应反向传播。

现在来实现乘法层。
乘法层作为 MulLayer 类，其实现过程如下所示（源代码在 ch05/layer_naive.py 中）。

__init__() 中会初始化实例变量 x 和 y，它们用于保存正向传播时的输入值。
forward() 接收 x 和 y 两个参数，将它们相乘后输出。
backward() 将从上游传来的导数（dout）乘以正向传播的翻转值，然后传给下游。

上面就是 MulLayer 的实现。
现在我们使用 MulLayer 实现前面的购买苹果的例子（2 个苹果和消费税）。
上一节中我们使用计算图的正向传播和反向传播，像图 5-16 这样进行了计算。

图 5-16　购买 2 个苹果
"""
class MulLayer(object):
    """实现乘法层"""
    def __init__(self):
        """初始化实例变量，用于保存正向传播时的输入值"""
        self.x = None
        self.y = None

    def forward(self, x, y):
        """"接收x，y两个参数，相乘后输出"""
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        """将从上游传来的导数dout乘以正向传播的翻转值，然后传给下游"""
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy
"""
使用这个乘法层的话，图 5-16 的正向传播可以像下面这样实现（源代码在ch05/buy_apple.py 中）。

"""

apple = 100
apple_num = 2
tax = 1.1
# 实例化两个乘法层
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()
# 前向传播
apple_price = mul_apple_layer.forward(apple, apple_num)
print(apple_price)

price = mul_tax_layer.forward(apple_price, tax)
print(price)
"""
此外，关于各个变量的导数可由 backward() 求出。
这里，调用 backward() 的顺序与调用 forward() 的顺序相反。
此外，要注意 backward() 的参数中需要输入“关于正向传播时的输出变量的导数”。
比如，mul_apple_layer 乘法层在正向传播时会输出 apple_price，在反向传播时，则会将 apple_price 的导数 dapple_price 设为参数。
另外，这个程序的运行结果和图 5-16 是一致的。
"""
# 反向传播
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax)