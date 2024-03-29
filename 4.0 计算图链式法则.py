"""
上一章中，我们介绍了神经网络的学习，并通过数值微分计算了神经网络的权重参数的梯度（严格来说，是损失函数关于权重参数的梯度）。
数值微分虽然简单，也容易实现，但缺点是计算上比较费时间。
本章我们将学习一个能够高效计算权重参数的梯度的方法——误差反向传播法。

要正确理解误差反向传播法，我个人认为有两种方法：
一种是基于数学式；
另一种是基于计算图（computational graph）。
前者是比较常见的方法，机器学习相关的图书中多数都是以数学式为中心展开论述的。
因为这种方法严密且简洁，所以确实非常合理，但如果一上来就围绕数学式进行探讨，会忽略一些根本的东西，止步于式子的罗列。
因此，本章希望大家通过计算图，直观地理解误差反向传播法。
然后，再结合实际的代码加深理解，相信大家一定会有种“原来如此！”的感觉。

此外，通过计算图来理解误差反向传播法这个想法，参考了 Andrej Karpathy的博客 [4] 和他与 Fei-Fei Li 教授负责的斯坦福大学的深度学习课程 CS231n[5]。

计算图将计算过程用图形表示出来。
这里说的图形是数据结构图，通过多个节点和边表示（连接节点的直线称为“边”）。
为了让大家熟悉计算图，本节先用计算图解一些简单的问题。
从这些简单的问题开始，逐步深入，最终抵达误差反向传播法。

现在，我们尝试用计算图解简单的问题。
下面我们要看的几个问题都是用心算就能解开的简单问题，这里的目的只是通过它们让大家熟悉计算图。
掌握了计算图的使用方法之后，在后面即将看到的复杂计算中它将发挥巨大威力，所以本节请一定学会计算图的使用方法。

问题 1：
太郎在超市买了 2 个 100 日元一个的苹果，消费税是 10%，请计算支付金额。
计算图通过节点和箭头表示计算过程。
节点用○表示，○中是计算的内容。
将计算的中间结果写在箭头的上方，表示各个节点的计算结果从左向右传递。
用计算图解问题 1，求解过程如图 5-1 所示。

图 5-1　基于计算图求解的问题 1 的答案

如图 5-1 所示，开始时，苹果的 100 日元流到“× 2”节点，变成 200 日元，然后被传递给下一个节点。
接着，这个 200 日元流向“× 1.1”节点，变成220 日元。
因此，从这个计算图的结果可知，答案为 220 日元。

虽然图 5-1 中把“× 2”“× 1.1”等作为一个运算整体用○括起来了，不过只用○表示乘法运算“×”也是可行的。
此时，如图 5-2 所示，可以将“2”和“1.1”分别作为变量“苹果的个数”和“消费税”标在○外面。

图 5-2　基于计算图求解的问题 1 的答案：“苹果的个数”和“消费税”作为变量标在○外面

问题 2：
太郎在超市买了 2 个苹果、3 个橘子。
其中，苹果每个 100 日元，橘子每个 150 日元。消费税是 10%，请计算支付金额。
同问题 1，我们用计算图来解问题 2，求解过程如图 5-3 所示。

图 5-3　基于计算图求解的问题 2 的答案


这个问题中新增了加法节点“+”，用来合计苹果和橘子的金额。
构建了计算图后，从左向右进行计算。就像电路中的电流流动一样，计算结果从左向右传递。
到达最右边的计算结果后，计算过程就结束了。
从图 5-3 中可知，问题 2 的答案为715 日元。

综上，用计算图解题的情况下，需要按如下流程进行。
01. 构建计算图。
02. 在计算图上，从左向右进行计算。
这里的第 2 歩“从左向右进行计算”是一种正方向上的传播，简称为正向传播（forward propagation）。
正向传播是从计算图出发点到结束点的传播。
既然有正向传播这个名称，当然也可以考虑反向（从图上看的话，就是从右向左）的传播。
实际上，这种传播称为反向传播（backward propagation）。
反向传播将在接下来的导数计算中发挥重要作用。

计算图的特征是可以通过传递“局部计算”获得最终结果。
“局部”这个词的意思是“与自己相关的某个小范围”。
局部计算是指，无论全局发生了什么，都能只根据与自己相关的信息输出接下来的结果。
我们用一个具体的例子来说明局部计算。
比如，在超市买了 2 个苹果和其他很多东西。
此时，可以画出如图 5-4 所示的计算图。

图 5-4　买了 2 个苹果和其他很多东西的例子

如图 5-4 所示，假设（经过复杂的计算）购买的其他很多东西总共花费 4000日元。
这里的重点是，各个节点处的计算都是局部计算。
这意味着，例如苹果和其他很多东西的求和运算（4000 + 200 → 4200）并不关心 4000 这个数字是如何计算而来的，只要把两个数字相加就可以了。
换言之，各个节点处只需进行与自己有关的计算（在这个例子中是对输入的两个数字进行加法运算），不用考虑全局。
综上，计算图可以集中精力于局部计算。
无论全局的计算有多么复杂，各个步骤所要做的就是对象节点的局部计算。
虽然局部计算非常简单，但是通过传递它的计算结果，可以获得全局的复杂计算的结果。

比如，组装汽车是一个复杂的工作，通常需要进行“流水线”作业。每个工人（机器）所承担的都是被简化了的工作，这个工作的成果会传递给下一个工人，直至汽车组装完成。
计算图将复杂的计算分割成简单的局部计算，和流水线作业一样，将局部计算的结果传递给下一个节点。
在将复杂的计算分解成简单的计算这一点上与汽车的组装有相似之处。

前面我们用计算图解答了两个问题，那么计算图到底有什么优点呢？
一个优点就在于前面所说的局部计算。
无论全局是多么复杂的计算，都可以通过局部计算使各个节点致力于简单的计算，从而简化问题。
另一个优点是，利用计算图可以将中间的计算结果全部保存起来（比如，计算进行到 2 个苹果时的金额是 200 日元、加上消费税之前的金额 650 日元等）。
但是只有这些理由可能还无法令人信服。
实际上，使用计算图最大的原因是，可以通过反向传播高效计算导数。

在介绍计算图的反向传播时，我们再来思考一下问题 1。
问题 1 中，我们计算了购买 2 个苹果时加上消费税最终需要支付的金额。
这里，假设我们想知道苹果价格的上涨会在多大程度上影响最终的支付金额，即求“支付金额关于苹果的价格的导数”。
设苹果的价格为 x，支付金额为 L，则相当于求 e_L/e_x。
这个导数的值表示当苹果的价格稍微上涨时，支付金额会增加多少。

如前所述，“支付金额关于苹果的价格的导数”的值可以通过计算图的反向传播求出来。
先来看一下结果，如图 5-5 所示，可以通过计算图的反向传播求导数（关于如何进行反向传播，接下来马上会介绍）。

图 5-5　基于反向传播的导数的传递

如图 5-5 所示，反向传播使用与正方向相反的箭头（粗线）表示。
反向传播传递“局部导数”，将导数的值写在箭头的下方。
在这个例子中，反向传播从右向左传递导数的值（1 → 1.1 → 2.2）。
从这个结果中可知，“支付金额关于苹果的价格的导数”的值是 2.2。
这意味着，如果苹果的价格上涨 1 日元，
最终的支付金额会增加 2.2 日元（严格地讲，如果苹果的价格增加某个微小值，则最终的支付金额将增加那个微小值的 2.2 倍）。

这里只求了关于苹果的价格的导数，不过“支付金额关于消费税的导数”“支付金额关于苹果的个数的导数”等也都可以用同样的方式算出来。
并且，计算中途求得的导数的结果（中间传递的导数）可以被共享，从而可以高效地计算多个导数。
综上，计算图的优点是，可以通过正向传播和反向传播高效地计算各个变量的导数值。

前面介绍的计算图的正向传播将计算结果正向（从左到右）传递，其计算过程是我们日常接触的计算过程，所以感觉上可能比较自然。
而反向传播将局部导数向正方向的反方向（从右到左）传递，一开始可能会让人感到困惑。
传递这个局部导数的原理，是基于链式法则（chain rule）的。
本节将介绍链式法则，并阐明它是如何对应计算图上的反向传播的。

话不多说，让我们先来看一个使用计算图的反向传播的例子。
假设存在 y = f(x) 的计算，这个计算的反向传播如图 5-6 所示。

图 5-6　计算图的反向传播：沿着与正方向相反的方向，乘上局部导数

如图所示，反向传播的计算顺序是，将信号 E 乘以节点的局部导数 ，然后将结果传递给下一个节点。
这里所说的局部导数是指正向传播中 y = f (x) 的导数，也就是 y 关于 x 的导数 e_y/e_x。
比如，假设 y = f(x**2)，则局部导数为 2x。
把这个局部导数乘以上游传过来的值（本例中为 E），然后传递给前面的节点。

这就是反向传播的计算顺序。
通过这样的计算，可以高效地求出导数的值，这是反向传播的要点。
那么这是如何实现的呢？我们可以从链式法则的原理进行解释。
下面我们就来介绍链式法则。

介绍链式法则时，我们需要先从复合函数说起。
复合函数是由多个函数构成的函数。
比如，z = (x + y)**2 是由式（5.1）所示的两个式子构成的。

z = t **2
t = x + y

链式法则是关于复合函数的导数的性质，定义如下。
如果某个函数由复合函数表示，则该复合函数的导数可以用构成复合函数的各个函数的导数的乘积表示。

这就是链式法则的原理，乍一看可能比较难理解，但实际上它是一个非常简单的性质。
以式（5.1）为例，e_Z/e_x（z 关于 x 的导数）可以用 e_Z/e_t（z 关于 t 的导数）和 e_t/e_x（t 关于 x 的导数）的乘积表示。
用数学式表示的话，可以写成式（5.2）。

e_Z/e_x = e_Z/e_t * e_t/e_x

式（5.2）中的 e_t 正好可以像下面这样“互相抵消”，所以记起来很简单。

现在我们使用链式法则，试着求式（5.2）的导数 e_Z/e_x。
为此，我们要先求式（5.1）中的局部导数（偏导数）。

e_Z/e_t = 2t
e_t/e_x = 1

如式（5.3）所示，e_Z/e_t 等于 2t，e_t/e_x 等于 1。
这是基于导数公式的解析解。
然后，最后要计算的 e_Z/e_x 可由式（5.3）求得的导数的乘积计算出来。

e_Z/e_x = e_Z/e_t * e_t/e_x = 2t * 1 = 2t

现在我们尝试将式（5.4）的链式法则的计算用计算图表示出来。
如果用“**2”节点表示平方运算的话，则计算图如图 5-7 所示。

图 5-7　式（5.4）的计算图：沿着与正方向相反的方向，乘上局部导数后传递

如图所示，计算图的反向传播从右到左传播信号。
反向传播的计算顺序是，先将节点的输入信号乘以节点的局部导数（偏导数），然后再传递给下一个节点。
比如，
反向传播时，“**2”节点的输入是 e_Z/e_Z，
将其乘以局部导数 e_Z/e_t（因为正向传播时输入是 t、输出是 z，所以这个节点的局部导数是e_Z/e_t ），然后传递给下一个节点。
另外，图 5-7 中反向传播最开始的信号 e_Z/e_Z 在前面的数学式中没有出现，这是因为 e_Z/e_Z=1 ，所以在刚才的式子中被省略了。

图 5-7 中需要注意的是最左边的反向传播的结果。
根据链式法则，e_Z/e_Z * e_Z/e_t * e_t/e_x 成立，对应“z 关于 x 的导数”。
也就是说，反向传播是基于链式法则的。

把式（5.3）的结果代入到图 5-7 中，结果如图 5-8 所示，e_Z/e_x  的结果为 2(x+ y)。

图 5-8　根据计算图的反向传播的结果，e_Z/e_x 等于 2(x + y)

上一节介绍了计算图的反向传播是基于链式法则成立的。
本节将以“+”和“×”等运算为例，介绍反向传播的结构。

首先来考虑加法节点的反向传播。
这里以 z = x + y 为对象，观察它的反向传播。
z = x + y 的导数可由下式（解析性地）计算出来。

e_Z/e_x = 1
e_Z/e_y = 1

如式（5.5）所示，e_Z/e_x 和 e_Z/e_y 同时都等于 1。
因此，用计算图表示的话，如图5-9 所示。

图 5-9　加法节点的反向传播：左图是正向传播，右图是反向传播。
如右图的反向传播所示，加法节点的反向传播将上游的值原封不动地输出到下游

在图 5-9 中，反向传播将从上游传过来的导数（本例中是 e_L/e_Z）乘以 1，然后传向下游。
也就是说，因为加法节点的反向传播只乘以 1，所以输入的值会原封不动地流向下一个节点。

另外，本例中把从上游传过来的导数的值设为 e_L/e_Z。
这是因为，如图 5-10 所示，我们假定了一个最终输出值为 L 的大型计算图。
z = x + y 的计算位于这个大型计算图的某个地方，从上游会传来 e_L/e_Z 的值，并向下游传递  和 。

图 5-10　加法节点存在于某个最后输出的计算的一部分中。
反向传播时，从最右边的输出出发，局部导数从节点向节点反方向传播

现在来看一个加法的反向传播的具体例子。
假设有“10 + 5 = 15”这一计算，反向传播时，从上游会传来值 1.3。
用计算图表示的话，如图 5-11 所示。

图 5-11　加法节点的反向传播的具体例子

因为加法节点的反向传播只是将输入信号输出到下一个节点，所以如图 5-11所示，反向传播将 1.3 向下一个节点传递。

接下来，我们看一下乘法节点的反向传播。
这里我们考虑 z = xy。
这个式子的导数用式（5.6）表示。

e_Z/e_x = y
e_Z/e_y = x

根据式（5.6），可以像图 5-12 那样画计算图。

图 5-12　乘法的反向传播：左图是正向传播，右图是反向传播

乘法的反向传播会将上游的值乘以正向传播时的输入信号的“翻转值”后传递给下游。
翻转值表示一种翻转关系，
如图 5-12 所示，
正向传播时信号是 x 的话，反向传播时则是 y；
正向传播时信号是 y 的话，反向传播时则是 x。

现在我们来看一个具体的例子。
比如，假设有“10 × 5 = 50”这一计算，反向传播时，从上游会传来值 1.3。
用计算图表示的话，如图 5-13 所示。

图 5-13　乘法节点的反向传播的具体例子

因为乘法的反向传播会乘以输入信号的翻转值，所以各自可按 1.3 × 5 =6.5、1.3 × 10 = 13 计算。
另外，加法的反向传播只是将上游的值传给下游，并不需要正向传播的输入信号。
但是，乘法的反向传播需要正向传播时的输入信号值。
因此，实现乘法节点的反向传播时，要保存正向传播的输入信号。

再来思考一下本章最开始举的购买苹果的例子（2 个苹果和消费税）。
这里要解的问题是苹果的价格、苹果的个数、消费税这 3 个变量各自如何影响最终支付的金额。
这个问题相当于求“支付金额关于苹果的价格的导数”“支付金额关于苹果的个数的导数”“支付金额关于消费税的导数”。
用计算图的反向传播来解的话，求解过程如图 5-14 所示。

图 5-14　购买苹果的反向传播的例子

如前所述，乘法节点的反向传播会将输入信号翻转后传给下游。
从图 5-14 的结果可知，苹果的价格的导数是 2.2，苹果的个数的导数是 110，消费税的导数是200。
这可以解释为，如果消费税和苹果的价格增加相同的值，
则消费税将对最终价格产生 200 倍大小的影响，苹果的价格将产生 2.2 倍大小的影响。
不过，因为这个例子中消费税和苹果的价格的量纲不同，所以才形成了这样的结果（消费税的 1是 100%，苹果的价格的 1 是 1 日元）。

最后作为练习，请大家来试着解一下“购买苹果和橘子”的反向传播。
在图 5-15 中的方块中填入数字，求各个变量的导数（答案在若干页后）。

图 5-15　购买苹果和橘子的反向传播的例子：在方块中填入数字，完成反向传播

"""