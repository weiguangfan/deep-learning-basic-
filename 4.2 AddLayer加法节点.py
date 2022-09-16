"""
接下来，我们实现加法节点的加法层，如下所示。
加法层不需要特意进行初始化，所以 __init__() 中什么也不运行（pass 语句表示“什么也不运行”）。
加法层的 forward() 接收 x 和 y 两个参数，将它们相加后输出。
backward() 将上游传来的导数（dout）原封不动地传递给下游。

现在，我们使用加法层和乘法层，实现图 5-17 所示的购买 2 个苹果和 3 个橘子的例子。

图 5-17　购买 2 个苹果和 3 个橘子

用 Python 实现图 5-17 的计算图的过程如下所示（源代码在ch05/buy_apple_orange.py 中）。



"""

class AddLayer(object):
    """加法层的实现"""
    def __init__(self):
        pass

    def forward(self,x,y):
        """接收x，y两个参数，相加后输出"""
        out = x + y
        return out

    def backward(self,dout):
        """将上游传来的导数dout原封不动地传递给下游"""
        dx = dout * 1
        dy = dout * 1
        return dx, dy


class MulLayer(object):
    """乘法层的实现"""
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1
# 实例化加法层和乘法层
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()
# 前向传播
apple_price = mul_apple_layer.forward(apple,apple_num)
orange_price = mul_orange_layer.forward(orange,orange_num)
all_price = add_apple_orange_layer.forward(apple_price,orange_price)
price = mul_tax_layer.forward(all_price, tax)
print(price)
# 后向传播
dprice = 1
dall_price,dtax = mul_tax_layer.backward(dprice)
dapple_price,dorange_price = add_apple_orange_layer.backward(dall_price)
dapple,dapple_num = mul_apple_layer.backward(dapple_price)
dorange,dorange_num = mul_orange_layer.backward(dorange_price)
print(dapple,dapple_num,dorange,dorange_num,dtax)
"""
这个实现稍微有一点长，但是每一条命令都很简单。
首先，生成必要的层，以合适的顺序调用正向传播的 forward() 方法。
然后，用与正向传播相反的顺序调用反向传播的 backward() 方法，就可以求出想要的导数。
综上，计算图中层的实现（这里是加法层和乘法层）非常简单，使用这些层可以进行复杂的导数计算。
下面，我们来实现神经网络中使用的层。

"""