"""
计算图的乘法节点，称为乘法层MulLayer;
加法节点称为加法层AddLayer;
层的实现中有两个共通的方法forward()正向传播和backward()反向传播;
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
# 反向传播
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax)