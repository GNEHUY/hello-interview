# LLM 八股
1.  请详细解释一下 Transformer 模型中的自注意力机制是如何工作的？它为什么比 RNN 更适合处理长序列？
2.  什么是位置编码？在 Transformer 中，为什么它是必需的？请列举至少两种实现方式。
3.  请你详细介绍ROPE，对比绝对位置编码它的优劣势分别是什么？
4.  你知道MHA，MQA，GQA的区别吗？详细解释一下。
5.  请比较一下几种常见的 LLM 架构，例如 Encoder-Only, Decoder-Only, 和 Encoder-Decoder，并说明它们各自最擅长的任务类型。
6.  什么是Scaling Laws？它揭示了模型性能、计算量和数据量之间的什么关系？这对LLM的研发有什么指导意义？
7.  在LLM的推理阶段，有哪些常见的解码策略？请解释 Greedy Search, Beam Search, Top-K Sampling 和 Nucleus Sampling (Top-P) 的原理和优缺点。
8.  什么是词元化？请比较一下 BPE 和 WordPiece 这两种主流的子词切分算法。
9.  你觉得NLP和LLM最大的区别是什么？两者有何共同和不同之处？
10. L1和L2正则化分别是什么，什么场景适合使用呢？
11. “涌现能力”是大型模型中一个备受关注的现象，请问你如何理解这个概念？它通常在模型规模达到什么程度时出现？
12. 激活函数有了解吗，你知道哪些LLM常用的激活函数？为什么选用它？
13. 混合专家模型（MoE）是如何在不显著增加推理成本的情况下，有效扩大模型参数规模的？请简述其工作原理。
14. 在训练一个百或千亿参数级别的 LLM 时，你会面临哪些主要的工程和算法挑战？（例如：显存、通信、训练不稳定性等）
15. 开源框架了解过哪些？Qwen，Deepseek的论文是否有研读过，说一下其中的创新点主要体现在哪？
16. 最近读过哪些LLM比较前沿的论文，聊一下它的相关方法，针对什么问题，提出了什么方法，对比实验有哪些？

--参考自[hello-agents](https://github.com/datawhalechina/hello-agents/blob/main/Extra-Chapter/Extra01-%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93.md)

## 参考回答
1. 
    <strong>Transformer 模型中的自注意力机制是如何工作的？</strong>
    
    简单来说：自注意力机制使得模型能够动态地衡量输入序列中不同单词之间的重要性，并据此生成每个单词的上下文感知表示。
    1. Q, K, V向量生成：对于输入序列中的每个token的嵌入向量，通过乘以三个可学习的权重矩阵，分别生成三个向量，查询Q，键K，值V
        * <strong>Query (Q):</strong> 代表当前词元为了更好地理解自己，需要去“查询”序列中其他词元的信息。
        * <strong>Key (K):</strong> 代表序列中每个词元所“携带”的，可以被查询的信息标签。
        * <strong>Value (V):</strong> 代表序列中每个词元实际包含的深层含义。

    2. 计算注意力分数（查找的过程）：为了确定当前token（由Q代表）应该对其他所有token（由K代表）投入多少关注，我们要计算当前token的Q和其他所有token的K的点积。
    <div align="center">
    $$\text{Score}(Q_i, K_j) = Q_i \cdot K_j$$
    </div>

    3. 缩放：将计算出的分数除以一个缩放因子 $\sqrt{d_k}$（ $d_k$ 是K向量的维度）。这一步是为了在反向传播时获得更稳定的梯度，防止点积结果过大导致Softmax函数进入饱和区。
    <div align="center">
    $$\frac{Q \cdot K^T}{\sqrt{d_k}}$$
    </div>

    4. Softmax归一化：通过一个Softmax函数，使其转换为一组总和为1的概率分布。这些概率就是“注意力权重”，表示在当前位置，每个输入词元所占的重要性。
    <div align="center">
    $$\text{AttentionWeights} = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)$$
    </div>

    4. 加权求和：将得到的注意力权重与每个词元对应的V向量相乘并求和，得到最终的自注意力层输出。这个输出向量融合了整个序列的上下文信息，且权重由模型动态学习得到。
    <div align="center">
    $$\text{Output} = \text{AttentionWeights} \cdot V$$
    </div>

    <strong>为什么比RNN更适合处理长序列？</strong>
    1. <strong>并行计算能力RNN较差：</strong>自注意力机制相比RNN可以一次性处理整个序列，计算所有位置之间的关联，是高度并行的。但是RNN是按时间顺序处理每个词，无法并行处理，在处理长序列时很慢
    2. <strong>解决RNN中的长程依赖：</strong>自注意力机制在处理任意两个位置时，复杂度为O(1)，可以直接计算注意力分数。但是RNN在处理首尾时必须要经过整个序列的信息传递，复杂度为O(N)，从CNN的残差连接的经验，这里RNN也会存在梯度消失或梯度爆炸的情况，导致模型很难捕捉长距离的依赖关系

---
