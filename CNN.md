# 卷积

## 实现方式

虽然理论上卷积计算是按照卷积核滑动，元素点乘计算的，但是实际上是按照im2col的方式计算的。这是为了优化速度，因为张量是按照行的元素存在内存上的，如果按原始方式点乘，则需要跨内存取值。

### im2col

输入特征图：$(1,C,H_{in},W_{in})$

卷积核：$(C_{out},C,K,K)$

输出特征图：$(1,C_{out},H_{out},W_{out})$

![image-20200414151124365](image\im2col.png)

1. **前向传播**

1. 先根据pad填充0，从左到右从上到下在每个通道上滑动写入右边的行，所有通道的叠在一起。

$H_{out}=[\frac{H_{in}-K+2p}{S}]+1$，向下取整

$W_{out}=[\frac{W_{in}-K+2p}{S}]+1$

输入->$(C_{in} \times K \times K,H_{out},W_{out})$

卷积核->$(C_{out},C_{in} \times K \times K)$

这样就可以按照矩阵相乘的方式进行卷积运算，并且取连续的内存，提高效率

![image-20200414151124365](image\im2col_times.png)

2. **反向传播**

输出特征图的梯度：$(1,C_{out},H_{out},W_{out})$，与输出特征图尺寸相同

则反算出的输入特征图的梯度：$(C_{in} \times K \times K,H_{out},W_{out})$，与输入特征图尺寸相同，再通过相反的col2im转换为$(1,C,H_{in},W_{in})$

注意：反传时卷积核需要转置再进行计算

![image-20200414151124365](image\im2col_back.png)

## 反卷积

**实现方式1：卷积的反向运算（主流实现方式)**

反卷积的前向传播实现与卷积的反向传播实现方式是一样的，只不过把梯度变成了特征图。

![image-20200414151124365](image\deconv_forward.png)

$H_{out}=(H_{in}-1) \times S +K -2p$

其实就是卷积的输出公式的输出和输入互换。

**实现方式2：插空补0再加一个卷积**

输入之间插$s-1$个0，边缘填充0的个数是$k-p-1$，将卷积核旋转180度，再进行点乘。

**反卷积缺点：棋盘效应**



**从另一个角度理解卷积和反卷积**

**卷积**

输入特征图：$x=(1,C_{in},H_{in},W_{in})$->$(1,C_{in},H_{in} \cdot W_{in} \times 1)$

卷积核：$(C_{out},C_{in},K,K)$->$C=(C_{out},C_{in},H_{out} \cdot W_{out} \times H_{in} \cdot W_{in})$，权值矩阵C是稀疏矩阵，很多元素都是0

![image-20200414151124365](image\C.png)

输出特征图：$y=(1,C_{out},H_{out},W_{out})$

则正向传播时，输入特征图与权重矩阵矩阵相乘后得$(1,C_{out},H_{out} \cdot W_{out})$->$$(1,C_{out},H_{out},W_{out})$$

正向传播时，$y=Cx$

反向传播时，$\frac{\partial L o s s}{\partial x}=C^{T}\frac{\partial L o s s}{\partial y}$

**反卷积**

正向传播时，$y=C^{T}x$

反向传播时，$\frac{\partial L o s s}{\partial x}=C\frac{\partial L o s s}{\partial y}$







