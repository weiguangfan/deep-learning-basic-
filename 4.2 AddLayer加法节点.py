"""
调用backward()和forward()的顺序相反；
backward()的参数中需要输入关于正向传播时的输出变量的导数；
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
