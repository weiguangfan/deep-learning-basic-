"""
神经网络学习中，学习率的值很重要。
学习率过小，会导致学习花费过多的时间；
学习率过大，则会导致学习发散而不能正确进行；
与训练相关的技巧：
1 寻找最优权重参数的最优化方法optimization:找到使损失函数的值尽可能小的参数；
    SGD:stochastic gradient descent 使用参数的梯度，沿梯度方向更新参数，并重复这个步骤多次，从而逐渐靠近最优参数。
        朝着梯度方向只前进一定距离的简单方法。
2 权重参数的初始值；
3 超参数的设定方法；
4 应对过拟合：权重衰减、dropout等正则化方法；
5 batch normalization;
"""

import sys, os
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
from dataset.mnist import load_mnist


class SGD(object):
    """
    sgd呈之字型移动，这是一个相当低效的路径。
    sgd的缺点是，如果函数的形状非均向（anisotropic），比如呈延伸状，搜索的路径就会非常低效。
    sgd低效的根本原因是，梯度的方向并没有指向最小值的方向。
    """
    def __init__(self, lr=0.01):
        """进行初始化的参数保存为实例变量"""
        self.lr = lr

    def update(self, params, grad):
        """该方法会被反复调用"""
        for key in params.keys():
            params[key] -= self.lr * grad[key]


class Momentum(object):
    """
    momentum是动量的意思，和物理有关。
    更新路径就像小球在碗中滚动一样。
    和sgd相比，之字形的程度减轻了；
    因为虽然x轴方向上受到的力非常小，但是一直在同一个方向上受力，所以朝同一个方向会有一定的加速。
    虽然y轴方向上受到的力很大，但是因为交互地受到正方向和反方向的力，它们会相互抵消，所以y轴方向上的速度不稳定。
    和sgd相比，可以更快地朝x轴方向靠近，减弱之字形的变动程度。
    """
    def __init__(self, lr=0.01, momentum=0.9):
        """
        self.v:保存物体的速度；
        :param lr:
        :param momentum:
        """
        self.lr = lr
        # 对应物理上的地面摩擦或空气阻力
        self.momentum = momentum
        # 初始化时，self.v中什么也不保存；
        self.v = None

    def update(self, params, grads):
        # 首次调用update()时，v会以字典型变量的形式保存与参数结构相同的数据。
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad(object):
    """
    在关于学习率的有效技巧中，有一种被称为学习率衰减learning rate decay的方法，即随着学习的进行，使学习率逐渐减小。
    实际上，一开始多学，然后逐渐少学的方法。
    逐渐减小学习率的想法，相当于将全体参数的学习率值一起降低。
    adagrad进一步发展了这个想法，针对一个一个的参数，赋予其定制的值。
    adagrad会为参数的每个元素适当地调整学习率，与此同时进行学习。
    adaGrad会记录过去所有梯度的平方和。
    因此，学习越深入，更新的幅度就越小。
    如果无止境的学习，更新量就会变为0。
    函数的取值高效地向着最小值移动。
    为了改善这个问题，RMSProp方法并不是将过去所有的梯度一视同仁地相加，而是逐渐地遗忘过去的梯度，在做加法运算时将新梯度的信息更多地反应出来。
    这种操作，称为指数移动平均，呈指数函数式地减小过去的梯度的尺度。

    由于y轴方向上的梯度较大，因此当开始变动较大，但是后面会根据这个较大的变动按比例进行调整，减小更新的步伐。
    因此y轴方向上的更新程度被减弱，之字形的变动程度有所衰减。
    """
    def __init__(self, lr=0.01):
        self.lr = lr
        # self.h保存了以前的所有梯度值的平方和
        # 通过乘以1 / np.sqrt(h)就可以调整学习的尺度。
        # 参数的元素中变动较大的元素的学习率将变小。
        # 可以按参数的元素进行学习率衰减，使变动大的参数的学习率逐渐减小。
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key,val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key] + 1e-7))


class Adam:
    """
    momentum参照小球在碗中滚动的物理规则进行移动；
    adagrad为参数的每个元素适当地调整跟新步伐；
    将这两个方法融合在一起，就是adam方法的基本思路。
    基于adam的更新过程就像小球在碗中滚动一样。
    虽然momentum也有类似的移动，但是相比之下，adam的小球左右摇晃的程度有所减轻。
    这得益于学习的更新程度被适当地调整了。
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)


class RMSprop:

    """
    RMSProp方法并不是将过去所有的梯度一视同仁地相加，而是逐渐地遗忘过去的梯度，在做加法运算时将新梯度的信息更多地反应出来。
    这种操作，称为指数移动平均，呈指数函数式地减小过去的梯度的尺度。
    """

    def __init__(self, lr=0.01, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


