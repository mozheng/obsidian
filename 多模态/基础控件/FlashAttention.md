论文：[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
代码：https://github.com/HazyResearch/flash-attention

## 前置知识点
GPU中存储单元主要有HBM和SRAM：HBM容量大但是访问速度慢，SRAM容量小却有着较高的访问速度。例如：A100 GPU有40-80GB的HBM，带宽为1.5-2.0TB/s；每108个流式多核处理器各有192KB的片上SRAM，带宽估计约为19TB/s。可以看出，片上的SRAM比HBM快一个数量级，但尺寸要小许多数量级。
## 传统Attention
当输入序列（sequence length）较长时，Transformer的计算过程缓慢且耗费内存，这是因为self-attention的时间和会随着内存复杂度会随sequence length的增加成二次增长的趋势。
标准attention 输入为 $Q, K, V \in \mathbb{R}^{N*d}$ ，输出为$O \in \mathbb{R}^{N*d}$，计算如下：
$$
S=QK^T \in \mathbb{R}^{N*N}, P=\text{softmax}(S) \in \mathbb{R}^{N*N}, O=PV \in \mathbb{R}^{N*d}
$$

标准Attention操作的中间结果S,P通常需要通过高带宽内存（HBM）进行存取。其中P中的softmax操作是row-wise的，即每行都算一次softmax，一共计算N行。
1. 从 HBM 按块加载 Q, K，计算 S = QK ，将 S 写入 HBM。
2. 从 HBM读取 S，计算 P = softmax(S)，将 P 写入 HBM。
3. 从 HBM 中按块加载 P和 V，计算 O = PV，将 O 写入 HBM。
4. 返回O。
![[Pasted image 20230821144328.png]]

## Flash Attention

使用tiling技术将注意力计算过程分块
使用recomputation技术在每个块内重新计算注意力输出
以此来优化Transformer模型在HBM和SRAM混合内存架构下的注意力计算过程。它避免了直接在HBM上计算整个注意力所需的大量读写,通过在SRAM上分块计算和最后的聚合,实现了较低的内存和计算复杂度。
![[Pasted image 20230821153108.png]]
## 结论
两者所需内存空间复杂度为 $O(N^2)$。本文分析：
- 计算对HBM的访问次数为 $\Omega (Nd+N^2)$
- FlashAttention：对HBM访问的次数为$O(N^{2}d^{2}M^{−1})$

往往N≫d（例如GPT2中N=1024，d=64），因此FlashAttention会快很多。