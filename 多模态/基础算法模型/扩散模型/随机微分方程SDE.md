## 0. 预备知识点
### 0.1 分数生成模型
在DDPM 诞生之前还有一种生成范式：基于数据分布相关的梯度的**分数生成模型（Score-based Generative Model）**。其主要目标是估计与数据分布相关的梯度，即“stein分数” $\nabla _{\mathbf{x}}\log p(\mathbf{x})$ 。并使用朗之万动力学（Langevin dynamics）的方法从估计的数据分布中进行采样来生成新的样本。
这里的

传统生成模型的目标就是要得到数据的分布。例如一个数据集${x_1, x_2, ..., x_N}$ 的数据的概率密度分布（注意，这里是概率密度分布，PDF）为p(x)，我们会按照正态分布的格式记为：
$$
p_{\theta}(\mathbf{x}) = \frac{e^{-f_{\theta}(\mathbf{x})}}{C_{\theta}},f_{\theta}(\mathbf{x})\in \mathbb{R}\\
$$
$\theta$ 是他们的参数，用于建模， $f_{\theta}$ 是核心energy-based model。这个型就很像高斯函数 $f(x)=\frac{1}{σ \sqrt{2π}} \cdot e^{\frac{-(x - μ)^2} {2σ^2}}$
我们一般可以通过最大化log-likelihood的方式 $\max_{\theta}\sum\limits_{i=1}^{N}\log_{\theta}(\mathbf{x}_{i})$ 训练参数$\theta$ 。我们进一步计算基于x的导数，去除部分变量为 
$$
\nabla _{\mathbf{x}}\log(p_{\theta}(\mathbf{x})) = -\nabla _{\mathbf{x}}f _{\theta}(\mathbf{x}) - \nabla _{\mathbf{x}}\log C_{\theta} = -\nabla _{\mathbf{x}}f _{\theta}(\mathbf{x})\\
$$
### 0.2 “分数”是什么？

数据往往是多维的。由分数的定义以及从数学的角度出发来看，它应当是一个“矢(向)量场”(vector field) 。向量的方向就是：对于输入数据(样本)来说，其对数概率密度增长最快的方向。（下图仅仅是示意，不代表真实场景）如果在采样过程中沿着分数的方向走，就能够走到数据分布的高概率密度区域（即为中心，方向近乎垂直区），最终生成的样本就会符合原数据分布。

![](images/分数.gif)
如果回到DDPM，分数与噪声有很大的关系
### 0.3  郎之万动力学采样方法

朗之万动力学（Langevin dynamics）原是描述物理学中布朗运动（悬浮在液体或气体中的微小颗粒所做的无规则运动）的微分方程，借鉴到这里作为一种生成样本的方法。
你说这个干什么？有用吗？当然！从一个分布中采样时，我们都要用。概括地来说，该方法首先从先验分布随机采样一个初始样本，然后利用模型估计出来的分数逐渐将样本向数据分布的高概率密度区域靠近。为保证生成结果的多样性，我们需要采样过程带有随机性。当经过中所述的分数匹配方法训练深度生成模型后，可以使用具有迭代过程的朗之万动力学采样方法从分布中来生成新的样本。

朗之万动力学采样过程描述：假设初始数据满足先验分布 $x_0 \sim \pi(x)$ ，然后使用迭代过程
$$
\begin{align*}
x_{t+1}&=x_t+\frac{\epsilon}{2}\nabla _{x}\log  p\left( x \right) +\sqrt{\epsilon}\boldsymbol{z}_t, t=0,1,2,\cdots ,T\\
&=x_t+\frac{\epsilon}{2}s_{\theta}\left( x \right) +\sqrt{\epsilon}\boldsymbol{z}_t, t=0,1,2,\cdots ,T
\end{align*}
$$
其中， $z_i \sim \mathcal{N}(0,I)$ 为标准高斯分布，可以看成是噪声的概念。理论上，当 $k \rightarrow \infty, \epsilon \rightarrow 0$ ，最终生成的样本 $x_T$ 将会服从原数据分布 $p_{data}(x)$，趋于某一个特定值。


## 1. 分数生成模型具体实现

现在我们想要训练一个网络来估计出真实的分布。 $s_\theta(\mathbf{x})$ 就是网络估计的分数，同理 $\theta$ 还是代表网络参数。我们可以最小化真实的score function，用这种形似来优化即可。
$$\mathcal{L} = \mathbb{E}_{p(\mathbf{x})}[||\nabla _{\mathbf{x}}\log p(\mathbf{x}) - \mathbf{s} _{\theta}(\mathbf{x})||^{2}]$$
但是这样的一个loss我们是算不出来的，因为我们并不知道真实的$p(\mathbf{x})$是什么。我们把上面loss的期望写开，二次项打开，可以得到
$$
\begin{align*}\mathcal{L} =& \mathbb{E}_{p(\mathbf{x})}[||\nabla _{\mathbf{x}}\log p(\mathbf{x}) - \mathbf{s} _{\theta}(\mathbf{x})||^{2}]\\
=& \int p(\mathbf{x}) [||\nabla _{\mathbf{x}}\log p(\mathbf{x})||^{2} + ||\mathbf{s} _{\theta}(\mathbf{x})||^{2} - 2(\nabla _{\mathbf{x}}\log p(\mathbf{x}))^{T}\mathbf{s} _{\theta}(\mathbf{x})] d \mathbf{x}\end{align*}\\
$$
第一项对于 $\theta$ 来说是常数可以忽略。因为我们要算网络 $\theta$ 的参数，与x无关，我们可以把第一项去掉，不予考虑。
第二项为：
$$\int p(\mathbf{x}) ||\mathbf{s} _{\theta}(\mathbf{x})||^{2} d \mathbf{x}$$
对于第三项，若x的维度为N则有：
$$
\begin{align*}
& -2\int p(\mathbf{x}) (\nabla _{\mathbf{x}}\log p(\mathbf{x}))^{T}\mathbf{s} _{\theta}(\mathbf{x}) d \mathbf{x}\\ 
=& -2 \int p(\mathbf{x}) \sum\limits_{i=1}^{N}(\frac{\partial \log p(\mathbf{x})}{\partial \mathbf{x}_{i}}\mathbf{s}_{\theta_i}(\mathbf{x})) d \mathbf{x}\\ 
=& -2 \sum\limits_{i=1}^{N} \int p(\mathbf{x}) \frac{1}{p(\mathbf{x})} \frac{\partial p(\mathbf{x})}{\partial \mathbf{x}_{i}}\mathbf{s}_{\theta_i}(\mathbf{x}) d \mathbf{x}\\ 
=& -2 \sum\limits_{i=1}^{N} \int \frac{\partial p(\mathbf{x})}{\partial \mathbf{x}_{i}}\mathbf{s}_{\theta_i}(\mathbf{x}) d \mathbf{x}\\ 
=& 2 \sum\limits_{i=1}^{N} - \int( \frac{\partial (p(\mathbf{x})\mathbf{s}_{\theta_i}(\mathbf{x}))}{\partial \mathbf{x}_{i}} d \mathbf{x} + \int p(\mathbf{x}) \frac{\partial \mathbf{s}_{\theta_i}(\mathbf{x})}{\partial \mathbf{x}_{i}}) d \mathbf{x}
\end{align*}
$$
上面的最后一步是分段求积分公式，因为$P(x)$取极限时都是0，这是PDF的特性。可得下面左边那部分为0。
$$
\begin{align*}
=& 2 - (p(\mathbf{x})\mathbf{s}_{\theta_i}(\mathbf{x})\bigg\rvert^{\infty}_{-\infty}) + \sum\limits_{i=1}^{N}\int p(\mathbf{x}) \frac{\partial \mathbf{s}_{\theta i}(\mathbf{x})}{\partial \mathbf{x}_{i}} d \mathbf{x}\\ 
=& 2 \sum\limits_{i=1}^{N} \int p(\mathbf{x}) \frac{\partial \mathbf{s}_{\theta i}(\mathbf{x})}{\partial \mathbf{x}_{i}} d \mathbf{x}\\ 
=& 2\int p(\mathbf{x}) \sum\limits_{i=1}^{N} \frac{\partial \mathbf{s}_{\theta i}(\mathbf{x})}{\partial \mathbf{x}_{i}} d \mathbf{x}\\ 
=& 2\int p(\mathbf{x}) \text{tr}(\nabla _{\mathbf{x}}\mathbf{s}_{\theta}(\mathbf{x})) d \mathbf{x}
\end{align*}
$$
$tr(.)$ 函数表示矩阵的迹(trace)，即矩阵主对角线元素的总和。因为 $\nabla _{\mathbf{x}}\mathbf{s}_{\theta}(\mathbf{x})$ 是个二阶海森矩阵求导，这里只用主对角线就行。
所以最后的loss是第二和第三项的和：
$$\begin{align*} \mathcal{L} &= \int p(\mathbf{x}) ||\mathbf{s} _{\theta}(\mathbf{x})||^{2} d \mathbf{x} + 2\int p(\mathbf{x}) \text{tr}(\nabla _{\mathbf{x}}\mathbf{s}_{\theta}(\mathbf{x})) d \mathbf{x}\\\\ &= \mathbb{E}_{p(\mathbf{x})}[||\mathbf{s} _{\theta}(\mathbf{x})||^{2} + 2\text{tr}(\nabla _{\mathbf{x}}\mathbf{s}_{\theta}(\mathbf{x}))].\end{align*}$$
这里再给大家说下，$tr(.)$ 到底是怎么求？
### 1.1  Denoising Score Matching(去噪分数匹配)
这种方法是作者在噪声条件分数网络 #NCSN 采取的方法，在这个思想里我们先不算迹 $tr(.)$ 。具体做法就是对原始数据加噪，使得其结果满足我们预先定义好的分布，比如高斯分布。这样，我们就知道了现在的概率密度了，于是就可以进行训练。
#### 1.1.1 噪声条件分数网络（NCSN）
![](分数在高低区域的作用.png)
我们再回到向量场的图，这个图仅仅演示用。真实情况是图像大部位是的区域是低概率密度区域，而损失函数简单的二阶计算会出现大密度区间“妨碍”小密度区间的问题。这种情况我们会加大噪声，填充整个区域。较大的噪声显然可以覆盖低密度区域，在更多区域获得更好的评分估计，但是过度破坏了数据。如图我们可以看到边界模糊，这就是强度过大的噪声反之会干扰到原始数据的分布的表现。这种情况还会造成估计分数的误差增大，从而基于加噪后的分数使用朗之万动力学采样生成的结果也就不太符合原数据分布。相反地，噪声强度小能够获得与原数据分布较为近似的效果，但是却不能够很好地“填充”低概率密度区域。为了达到两者最佳。我们使用了多尺度噪声扰动。
定义一组几何级数序列 $\left\{ \sigma _i \right\} _{i=1}^{L}$ ，其中 $\sigma _i>0$ 且 $\frac{\sigma _1}{\sigma _2}=\frac{\sigma _2}{\sigma _3}=\cdots =\frac{\sigma _{L-1}}{\sigma _L}>1$ ，用分布 $q_{\sigma}\left( \mathbf{x} \right) =\int{p_{data}\left( \mathbf{x} \right) \mathcal{N} \left( \mathbf{x}|\mathbf{t},\sigma ^2\mathbf{I} \right) \mathrm{d}\mathbf{t}}$ （暂时不用理解为什么）表示扰动噪声数据分布。NCSN的目标是训练一个条件分数网络 $s_{\theta}\left( \mathbf{x};t \right)$ 来估计扰动数据的分布，即 $\forall {{\mathbf{\sigma }}_{i}}\in \left\{ \sigma _i \right\} _{i=1}^{L}$ 都有 $s_{\theta}\left( \mathbf{x},\sigma \right) \approx \nabla _x\log q_{\sigma}\left( \mathbf{x} \right)$ ，其中当 $\mathbf{x}\in \mathbb{R} ^D$ 时 $s_{\theta}\left( \mathbf{x},\sigma \right) \in \mathbb{R} ^D$ 。一般把 $s_{\theta}\left( \mathbf{x};t \right)$ 称为噪声条件分数网络。
