#DDPM
Denoising Diffusion Probabilistic Models
论文：https://arxiv.org/abs/2006.11239
代码：https://github.com/lucidrains/denoising-diffusion-pytorch
```text
注意：同为图像生成，本章节不需要知道GAN或VAE的前置基础知识。
```
## 1. 背景介绍

扩散模型算法是一种受到热力学中非平衡热力学分支启发的算法，其思想来源是扩散过程，通过神经网络学习从纯噪声数据逐渐对数据进行去噪的过程，从单个图像样来看这个过程，扩散过程q就是不断往图像上加噪声直到图像变成一个纯噪声，扩散过程从 到最后的 就是一个马尔可夫链，表示状态空间中经过从一个状态到另一个状态的转换的随机过程。逆扩散过程p就是从纯噪声生成一张图像的过程。

## 2. 理论推理
### 2.1 变量声明
![](images/ddpm图示.png)
1. $x_0$ 为原始图像，$x_t$ 为加t步噪声后的图像，噪声加到最后（第T步）为 $x_T$ 
2. 图像加噪声为 $q(x_t|x_{t-1})$ 操作，意为从 $x_{t-1}$ 加噪声成 $x_t$ ；图像降噪为 $p_\theta(x_{t-1}|x_t)$ 操作，意为从 $x_t$ 降噪声成 $x_{t-1}$ 
3. 每一步加噪声（或去噪声）都有参数对 $\alpha_t,\beta_t$ ，其中有 $0<\beta_t<<\alpha_t<1$ ，并且 $\alpha_t+\beta_t=1$

### 2.2 前向过程-混入噪声
1. 从原图 $x_0$ 开始一点一点加噪声
$$
\begin{equation*} 
\begin{split}
x_1 &= \sqrt{\alpha_1}x_0+\sqrt{1-\alpha_1}Z_1 \\
x_2 &= \sqrt{\alpha_2}x_1+\sqrt{1-\alpha_2}Z_2 \\
&=\sqrt{\alpha_2 \alpha_1}x_0+\sqrt{\alpha_2(1-\alpha_1)}Z_1+\sqrt{1-\alpha_2}Z_2
\end{split}
\end{equation*} 
$$
2. 因为噪声是基于标准正态分布随机采样的，后面两个噪声 $Z$ 可以合并到一起（即两个正态分布相加）。方法如下
$$
\begin{equation*} 
\begin{split}
\sqrt{\alpha_2(1-\alpha_1)}Z_1 & \sim N(0,\sqrt{\alpha_2-\alpha_2 \alpha_1}) \\
\sqrt{1-\alpha_2}Z_2 & \sim N(0,\sqrt{1-\alpha_2 }) \\
\sqrt{\alpha_2(1-\alpha_1)}Z_1+\sqrt{1-\alpha_2}Z_2 & \sim N(0,\sqrt{1-\alpha_2 \alpha_1})  \\
\end{split}
\end{equation*} 
$$
可知
$$ x_2=\sqrt{\alpha_2 \alpha_1}x_0+\sqrt{1-\alpha_2 \alpha_1}Z \\ $$
这里的 $Z$ 没有下标，因为已经与 $Z_1$ 含义不同了，但还保持正态分布的特性。最终递归可得
$$
x_t=\sqrt{\bar{\alpha_t}}x_0+\sqrt{1-\bar{\alpha_t}}Z , \bar{\alpha_t}=\Pi_{s=0}^{t}\alpha_s
$$
这里的 $\bar{\alpha_t}$ 就是前面 $\alpha$ 序列连续累乘的表示记法，与我们传统的平均数标识记法无关。用条件概率的写法就是
$$ q(x_t|x_0)=N(x_t;\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})$$
注意，这里括号内的记法多了个 $x_t;$ ，代表这是 $x_t$ 的分布。其实本没有必要这么写，唯一的目的是让大家注意这是基于谁的分布。这一段对应的python的代码也很直接。其中 (sqrt_alphas_cumprod, t) 就是 $\sqrt{\bar{\alpha_t}}$ ，(sqrt_one_minus_alphas_cumprod, t) 就是 $\sqrt{1-\bar{\alpha_t}}$  
```python
def q_sample(self, x_start, t, noise=None):
	noise = default(noise, lambda: torch.randn_like(x_start))
	return (
		extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
		extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
	)
```

### 2.3 后向过程
反向过程都是基于 $q ( x_{t − 1} ∣ x_t, x_0 )$ 推理的。总体思想是，在知道原始图像 $x_0$ 、噪声 $x_t$ 的两个条件下给噪声 $x_t$ 降噪，得到  $x_{t − 1}$ 的过程。这时，许多人会有两个疑问点。

>**疑问点1：你不就是要一步一步求原始图像 $x_0$ 吗？你这里将原始图像 $x_0$ 当成条件是什么意思？**
> 对！因为单纯思考 $q ( x_{t − 1} ∣ x_t)$ 没有任何条件是肯定推不出来的，但是 $q ( x_{t − 1} ∣ x_0)$ 是可以的 。我这里借 $x_0$ 推理我的条件概率。而且由于上文推理出 $x_t=\sqrt{\bar{\alpha_t}}x_0+\sqrt{1-\bar{\alpha_t}}Z$ 这一公式，可得 $x_0=\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t-\sqrt{1-\bar{\alpha}_t}z_t)$ 。（注意：此时的 $z_t$ 与原始声明变量中的 $Z_t$ 并不一致，他是正态重定义后的标识，但是由于是随机噪声，本质一致，所以才这么记。）又因为 $x_0$ 也可以用 $x_t$ 进行表示，所以在把噪声参数当成常量时 ，$q ( x_{t − 1} ∣ x_t, x_0 )$ 与 $q ( x_{t − 1} ∣ x_t)$ 是相等。

> **疑问点1 plus：那也不对啊，前向推理 $z_t$ 是提前生成，是已知的。后向过程怎么知道？**
> 对！能思考到这一步就说明你很棒，到现在前后向概念还没有混淆！在这里我们会用一个带参数 $\theta$  的噪声 $z_\theta(x_t,t)$ 来表示 $z_t$ 。这种公式的妥协，带给我们一个预感：损失函数与噪声是直接相关的。具体的证明，在优化过程中得以完整描述

> **疑问点2：后向过程不是说好了用  $p_\theta(x_{t-1}|x_t)$ 表示吗？你怎么还用 q？**
> 对！我说这是我个人爱好行吗？☺，你翻翻其他人的博客文献，你真会发现他们的确用p来表示。但是我这里用q是有两个原因：其一，这里仅仅表示简单条件概率；其二单看在 $x_0$ 的条件下推理 $x_{t-1}$ ，这就是前向过程。

接下来就是推理过程：
$$
\begin{align}
q({x}_{t-1} \vert {x}_t, {x}_0) 
&= q({x}_t \vert {x}_{t-1}, {x}_0) \frac{ q({x}_{t-1} \vert {x}_0) }{ q({x}_t \vert {x}_0) } \\
&= \frac{\sqrt{1-\bar{\alpha}_t}}{ \sqrt{2\pi}\sqrt{\beta_t}\sqrt{1-\bar{\alpha}_{t-1}}} \cdot e^ { (-\frac{1}{2} (\frac{({x}_t - \sqrt{\alpha_t} {x}_{t-1})^2}{\beta_t} + \frac{({x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} {x}_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{({x}_t - \sqrt{\bar{\alpha}_t} {x}_0)^2}{1-\bar{\alpha}_t} ) )}
\end{align}
$$
此时有小伙伴会喊“停停！第一个等号是贝叶斯公式我知道，第二个等号是怎么回事？”
其实这里都是高斯分布 $f(x)=\frac{1}{σ \sqrt{2π}} \cdot e^{\frac{-(x - μ)^2} {2σ^2}}$ ，相乘是指数相加，相除是指数相减。根据上一步的前向公式我们知道 $q(x_t|x_0)=N(x_t;\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})$  与 $x_t = \sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}Z_t$ ，因此我们可以推出以下：
$$
\begin{align}
q({x}_{t} \vert {x}_{t-1}, {x}_0)  &= N(x_{t};\sqrt{\alpha_{t}}x_{t-1}, (1-{\alpha}_{t})\mathbf{I}) \\
q(x_{t-1}|x_0) & = N(x_{t-1};\sqrt{\bar{\alpha}_{t-1}}x_0, (1-\bar{\alpha}_{t-1})\mathbf{I}) \\
q(x_t|x_0) & = N(x_t;\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})
\end{align}
$$
前面设定了 $\alpha_t+\beta_t=1$ ，因此这里 $q({x}_{t} \vert {x}_{t-1}, {x}_0)= N(x_{t};\sqrt{\alpha_{t}}x_{t-1}, {\beta}_{t}\mathbf{I})$ ，继续推理得到以下的结果：
$$
\begin{align}
q({x}_{t-1} \vert {x}_t, {x}_0) 
&= q({x}_t \vert {x}_{t-1}, {x}_0) \frac{ q({x}_{t-1} \vert {x}_0) }{ q({x}_t \vert {x}_0) } \\
&= \frac{\sqrt{1-\bar{\alpha}_t}}{\sqrt{2\pi}\sqrt{\beta_t}\sqrt{1-\bar{\alpha}_{t-1}}} \cdot e^ {(-\frac{1}{2} (\frac{({x}_t - \sqrt{\alpha_t} {x}_{t-1})^2}{\beta_t} + \frac{({x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} {x}_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{({x}_t - \sqrt{\bar{\alpha}_t} {x}_0)^2}{1-\bar{\alpha}_t} ) ) }\\
&= C(\alpha,\beta,t)\cdot e ^ {-\frac{1}{2} (\frac{{x}_t^2 - 2\sqrt{\alpha_t} {x}_t {{x}_{t-1}} {+ \alpha_t} {{x}_{t-1}^2} }{\beta_t} + \frac{ {{x}_{t-1}^2} {- 2 \sqrt{\bar{\alpha}_{t-1}} {x}_0} {{x}_{t-1}} {+ \bar{\alpha}_{t-1} {x}_0^2}  }{1-\bar{\alpha}_{t-1}} - \frac{({x}_t - \sqrt{\bar{\alpha}_t} {x}_0)^2}{1-\bar{\alpha}_t} ) } \\
&= C(\alpha,\beta,t)\cdot e^{-\frac{1}{2} ( {(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})} {x}_{t-1}^2 - {(\frac{2\sqrt{\alpha_t}}{\beta_t} {x}_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} {x}_0)} {x}_{t-1} + C({x}_t, {x}_0) ) )}
\end{align}
$$
最后一步是基于 $x_{t-1}$ 的二次方程的配方，因为我们算的就是  $x_{t-1}$ 分布嘛！开头项 $C(\alpha,\beta,t)$ 与最后项 $C(x_t,x_0)$  不是基于 $x_{t-1}$ 的函数可以这里将其分出来。这么做的目的其实是将本应是正态分布的 $q({x}_{t-1} \vert {x}_t, {x}_0)$ 化为正态分布格式，方便我们获取其均值与方差。又因为正态分布的 $(\sigma, \mu)$ 是基于二次方程配方结果。
$$
\begin{align}
\tilde{\sigma}_t 
&= 1/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) 
= 1/(\frac{\alpha_t - \bar{\alpha}_t + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})})
= {\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
\tilde{\boldsymbol{\mu}}_t ({x}_t, {x}_0)
&= (\frac{\sqrt{\alpha_t}}{\beta_t} {x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} {x}_0)/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) \\
&= (\frac{\sqrt{\alpha_t}}{\beta_t} {x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} {x}_0) {\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} {x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} {x}_0\\
\end{align}
$$
又因为我们是借 $q ( x_{t − 1} ∣ x_t)$ 算 $q ( x_{t − 1} ∣ x_t, x_0 )$ ，所以要通过 $x_0=\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t-\sqrt{1-\bar{\alpha}_t} \cdot z_\theta(x_t,t))$ 消去 $x_0$ 项。带入可得 
$$
\tilde{\boldsymbol{\mu}}_t ({x}_t, {x}_0) = \mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} z_\theta(x_t,t))
$$
最终总结可得：
$$
q({x}_{t-1} \vert {x}_t) = N(\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} z_\theta(x_t,t)),\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t)
$$
其中带参数 $\theta$  的噪声 $z_\theta(x_t,t)$ 可以用神经网络拟合。反向过程结束。
> **注意**：后面的推理中会单纯用到 $q({x}_{t-1} \vert {x}_t, {x}_0)$ ，此时的 $z_t$ 就不用带参数 $\theta$  的噪声 $z_\theta(x_t,t)$ 来表示了。

### 2.4 优化过程
优化过程就是损失函数的计算，该推理有很多种方法，但大体一样。这里我选择一个我喜欢的方法。这是基于KL散度非负的特性做的，如下：
$$
 \begin{align} 
-\log p_\theta(x_0)&\leq-\log p_\theta(x_0)+D_{KL}(q(x_{1:T}|x_0)||p_\theta(x_{1:T}|x_0))\\ &=-\log p_\theta(x_0)+\mathbb{E}_{q(x_{1:T}|x_0)}\left[\log\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})/p_\theta(x_0)}\right];\text{where}\quad p_\theta(x_{1:T}|x_0)=\frac{p_\theta(x_{0:T})}{p_\theta(x_0)}\\
&=-\log p_\theta(x_0)+\mathbb{E}_{q(x_{1:T}|x_0)}\left[\log\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}+\underbrace{\log p_\theta(x_0)}_{与q无关，可拉出来}\right]\\ &=\mathbb{E}_{q(x_{1:T}|x_0)}\left[\log\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}\right]
 \end{align}
$$
这里正式使用了带参数 $\theta$ 的后向去噪 $p_\theta(x_{1:T}|x_0)$ 函数。上式左右两边取期望，并利用到重积分中的Fubini定理可得
$$\small\mathcal{L}_{VLB}=\underbrace{\mathbb{E}_{q(x_0)}\left(\mathbb{E}_{q(x_{1:T}|x_0)}\left[\log\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}\right]\right)=\mathbb{E}_{q(x_{0:T})}\left[\log\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}\right]}_{Fubini定理}\geq\mathbb{E}_{q(x_0)}[-\log p_\theta(x_0)]$$
> **说明**：Fubini提出：在一定可积的条件下，不仅能够用逐次积分计算双重积分，而且交换逐次积分的顺序时，积分结果不变。这个可积的条件在Fubini那里比较苛刻，但我们不用担心，因为这里的期望函数都比较简单。有兴趣的可以找资料详细研究，这里不做陈述。（就是懒）

这里，我们最小化左边的 $\small\mathcal{L}_{VLB}$ ，即可最小化 $p_\theta(x_0)$ 的信息熵的最大上界，从而最小化信息熵。下面过程很复杂知道即可，进一步推导VLB，得到组合的KL散度和熵。我们还可以将结果拆开写成右边这种格式
$$
\begin{aligned}
\mathcal{L}_\text{VLB} 
&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
&= \mathbb{E}_q \Big[ \log\frac{\prod_{t=1}^T q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{ p_\theta(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t) } \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=1}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \Big( \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)}\cdot \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1}\vert\mathbf{x}_0)} \Big) + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \vert \mathbf{x}_0)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{q(\mathbf{x}_1 \vert \mathbf{x}_0)} + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big]\\
&= \mathbb{E}_q \Big[ \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1) \Big] \\
&= \mathbb{E}_q [\underbrace{D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t))}_{L_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)}_{L_0} ]
\end{aligned}
$$
看最后的公式推理，可以将结果简化为下面这种形式
$$
\begin{align}
 &\mathcal{L}_{VLB}=\mathbb{E}_{q}(L_T+L_{T-1}+...+L_0) \\ 
&L_T=D_{KL}(q(x_T|x_0)||p_\theta(x_T)) \\
&L_t=D_{KL}(q(x_t|x_{t+1},x_0)||p_\theta(x_t|x_{t+1}));\qquad 1\leq t \leq T-1 \\
&L_0=-\log p_\theta(x_0|x_1)
\end{align}
$$
这里的 $L_T,L_0$ 都是固定的，没有可优化价值我们关注中间的 $L_t$ ，它本质是下面两个正态分布求KL散度
$$
\begin{align}
q(x_{t-1}|x_t, x_0)=N(x_{t};\tilde{\mu}(x_t,x_0),\Sigma_t)&= N(\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} z_t,t)),\Sigma)  \\
p_\theta(x_{t-1}|x_{t})=N(x_{t-1};\mu_\theta(x_t,t),\Sigma_t)&=N(\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} z_\theta(x_t,t)),\Sigma)
\end{align}
$$
其中  $\Sigma_t= \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t$ ，因为不包含目标x与噪声z，这里不展开说。根据多元高斯分布求KL散度的公式如下：
$$
\begin{align} 
L_{t-1}&=\mathbb{E}_{x_0,z_t}\left[\frac{1}{2||\Sigma(x_t,t)||_2^2}||\tilde{\mu}_t(x_t,x_0)-\mu_\theta(x_t,t)||^2\right] \\
&=\mathbb{E}_{x_0,z_t}\left[\frac{1}{2||\Sigma(x_t,t)||_2^2}||\frac{1}{\sqrt{a_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar{a}_t}}z_t)-\frac{1}{\sqrt{a_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar{a}_t}}z_\theta(x_t,t))||^2\right] \\ 
&=\mathbb{E}_{x_0,z_t}\left[\frac{\beta_t^2}{2\alpha_t(1-\bar{\alpha}_t)||\Sigma(x_t,t)||_2^2}||z_t-z_\theta(x_t,t)||^2\right] \\ 
&=\mathbb{E}_{x_0,z_t}\left[\frac{\beta_t^2}{2\alpha_t(1-\bar{\alpha}_t)||\Sigma(x_t,t)||_2^2}||z_t-z_\theta(\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}z_t,t)||^2\right]
\end{align}
$$
此时我们发现只要通过拉进 $z_t$ 与 $z_\theta(x_t,t)$ 的距离的方式就可以训练参数 $\theta$。所以，我们利用噪声间MSE Loss 即可完成优化损失函数的设计：
$$ \mathcal{Loss} =  ||z_t-z_\theta(\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}z_t,t)||^2 
$$

## 3. 理论实现
### 3.1 训练过程
![](images/ddpm训练.png)
上图为原论文的训练过程，该过程比较简单，在一个训练循环内大致分为5步：
1. 【变量准备】从数据集中选取 $x_0$ ，这就是原始图片
2. 【变量准备】随机选取时间戳 t，它代表扩散模型需要扩散的轮数。
3. 【变量准备】生成t个高斯噪声，每个都是 $\epsilon_t\in\mathcal{N}(0, \mathbf{I})$
4. 【模型设计】调用模型 $ϵ_θ$（这里是UNet网络）预估 $\epsilon_\theta\left(\sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} \epsilon_t, t\right)$
5. 【损失函数】计算噪声之间的 MSE Loss与反向传播: $\mathcal{Loss} =  \left\|\epsilon_t-\epsilon_\theta\left(\sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} \epsilon_t, t\right)\right\|^2$

### 3.2 推理过程
![](images/ddpm采样.png)
上图为原论文的推理采样过程伪代码。因为此时已经训练出来了 $\epsilon_θ$ （这里是UNet网络），所以在下面的推理过程中 $ϵ_θ(x_t,t)$ 是已知的。假设我用推理的过程中扩散T步，那么从T步开始逆向回推，每一步有如下操作：

1. 如果 $t = 1$ ，噪声 $z = 0$ 。因为在算 $\sigma_t$ 时需要 $\alpha_0$ ，但我们没有，所以在这里直接 $z = 0$ 不计算即可。如果 $t > 1$ 时，即可取随机噪声 $z\in\mathcal{N}(0, \mathbf{I})$ 。
2. 因为前面我们已经得出 $q({x}_{t-1} \vert {x}_t) = N(\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} z_\theta(x_t,t)),\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t)$ ， 所以可得如下推理
$$
\begin{align}
& \frac{x_{t-1} - \mu}{\sigma} \sim N(0,I)=z \\
& x_{t-1} = \mu + \sigma z \\
& x_{t-1} =\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} z_\theta(x_t,t)) + \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t z
\end{align}
$$
3. 最后一步公式对应的就是 $x_{t-1} =\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{1- \alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t,t)) + \sigma_t z$
