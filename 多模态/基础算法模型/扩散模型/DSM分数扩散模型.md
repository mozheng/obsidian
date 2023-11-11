## 分数生成模型
在DDPM 诞生之前还有一种生成范式：基于数据分布相关的梯度的**分数生成模型（Score-based Generative Model）**。其主要目标是估计与数据分布相关的梯度，即“stein分数” $\nabla _{\mathbf{x}}\log p(\mathbf{x})$ 。并使用朗之万动力学（Langevin dynamics）的方法从估计的数据分布中进行采样来生成新的样本。
这里的

传统生成模型的目标就是要得到数据的分布。例如一个数据集${x_1, x_2, ..., x_N}$ 的数据的概率密度分布（注意，这里是概率密度分布，PDF）为p(x)，我们会按照正态分布的格式记为：
$$
p_{\theta}(\mathbf{x}) = \frac{e^{-f_{\theta}(\mathbf{x})}}{C_{\theta}},f_{\theta}(\mathbf{x})\in \mathbb{R}\\
$$
$\theta$ 是他们的参数，用于建模， $f_{\theta}$ 是核心energy-based model。这个型就很像高斯函数$f(x)=\frac{1}{σ \sqrt{2π}} \cdot e^{\frac{-(x - μ)^2} {2σ^2}}$
我们一般可以通过最大化log-likelihood的方式 $\max_{\theta}\sum\limits_{i=1}^{N}\log_{\theta}(\mathbf{x}_{i})$ 训练参数$\theta$ 。我们进一步计算基于x的导数，去除部分变量为 
$$
\nabla _{\mathbf{x}}\log(p_{\theta}(\mathbf{x})) = -\nabla _{\mathbf{x}}f _{\theta}(\mathbf{x}) - \nabla _{\mathbf{x}}\log C_{\theta} = -\nabla _{\mathbf{x}}f _{\theta}(\mathbf{x})\\
$$
## Score matching

现在我们想要训练一个网络来估计出真实的score function来拟合c， $s_\theta(\mathbf{x})$ 是网络估计的分数。我们可以最小化真实的score function，用这种形似来优化即可。
$$\mathcal{L} = \mathbb{E}_{p(\mathbf{x})}[||\nabla _{\mathbf{x}}\log p(\mathbf{x}) - \mathbf{s} _{\theta}(\mathbf{x})||^{2}]$$
但是这样的一个loss我们是算不出来的，因为我们并不知道真实的$p(\mathbf{x})$是什么。我们把上面loss的期望写开，二次项打开，可以得到
$$
\begin{align*}\mathcal{L} =& \mathbb{E}_{p(\mathbf{x})}[||\nabla _{\mathbf{x}}\log p(\mathbf{x}) - \mathbf{s} _{\theta}(\mathbf{x})||^{2}]\\=& \int p(\mathbf{x}) [||\nabla _{\mathbf{x}}\log p(\mathbf{x})||^{2} + ||\mathbf{s} _{\theta}(\mathbf{x})||^{2} - 2(\nabla _{\mathbf{x}}\log p(\mathbf{x}))^{T}\mathbf{s} _{\theta}(\mathbf{x})] d \mathbf{x}\end{align*}\\
$$
第一项对于$\theta$来说是常数可以忽略。  
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
$tr(.)$函数表示矩阵的迹(trace)，即矩阵主对角线元素的总和。因为$\nabla _{\mathbf{x}}\mathbf{s}_{\theta}(\mathbf{x})$ 是个二阶海森矩阵求导，这里只用主对角线就行。
所以最后的loss是第二和第三项的和：
$$\begin{align*} \mathcal{L} &= \int p(\mathbf{x}) ||\mathbf{s} _{\theta}(\mathbf{x})||^{2} d \mathbf{x} + 2\int p(\mathbf{x}) \text{tr}(\nabla _{\mathbf{x}}\mathbf{s}_{\theta}(\mathbf{x})) d \mathbf{x}\\\\ &= \mathbb{E}_{p(\mathbf{x})}[||\mathbf{s} _{\theta}(\mathbf{x})||^{2} + 2\text{tr}(\nabla _{\mathbf{x}}\mathbf{s}_{\theta}(\mathbf{x}))].\end{align*}$$
