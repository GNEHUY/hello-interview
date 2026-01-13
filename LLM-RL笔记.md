# LLM RL 学习笔记

*本笔记是为了记录关于大语言模型（LLM）强化学习（RL）的一些核心算法与理论。*

---

## 1. RL 算法的分类

| 分类 | 核心特点 | 代表算法 | 优势 |
| :--- | :--- | :--- | :--- |
| **Online RL** | 需要与环境实时交互，采样最新轨迹进行训练。 | PPO, GRPO, DAPO, GSPO, Scaling GRPO | 更好激发模型推理能力，探索未知解空间。 |
| **Offline RL** | 避免在线采样，直接利用预先收集的数据集优化。 | DPO, Rejection Sampling | 在对齐阶段效率极高，计算资源消耗更低。 |

---

## 2. PPO —— 经典“双塔”之 Actor-Critic

PPO (Proximal Policy Optimization) 的核心思想是在优化策略的同时，利用 `clip` 机制限制新旧策略的差异，防止更新步长过大导致训练崩溃。

### 训练目标函数



$$
\mathcal{J}_{PPO}(\theta) = \mathbb{E}_{(q,o) \sim \pi_{old}} \left[ \sum_{t=1}^{T} \min \left( \rho_t(\theta) \hat{A}_t, \text{clip}(\rho_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

**核心项说明：**
* **$\mathcal{J}_{PPO}(\theta)$**: PPO 的目标函数，通过最大化它来更新参数 $\theta$。
* **$\rho_t(\theta)$**: 重要性采样比率（Importance Sampling Ratio），即 $\frac{\pi_{\theta}(o_t|q,o_{<t})}{\pi_{{old}}(o_t|q,o_{<t})}$，衡量新旧策略的差异。
* **$\hat{A}_t$**: 优势函数（Advantage Function）的估计值，衡量当前动作比平均表现好多少
* **$\text{clip}(\dots)$**: 裁剪函数，将比率限制在 $[1-\epsilon, 1+\epsilon]$ 之间，保证训练稳定性。

---

### 2.1 优势函数的计算：TD Error 与 GAE



### TD Error ($\delta_t$)
全称 **Temporal Difference Error**（时间差分误差）。根据优势函数的广义定义 $\hat{A}_t = Q(s_t, a_t) - V(s_t)$，定义 $\delta_t$ 为：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

其中 $\gamma$ 为 **Discount Factor**（折扣因子），范围 $[0, 1]$。它决定了模型看多远：
* $\gamma = 0.99$: 表示模型非常在意未来的收益。
* $\gamma = 0$: 表示模型只在乎眼前的奖励。

### GAE (Generalized Advantage Estimation)
为了平衡 **Bias (偏差)** 和 **Variance (方差)**，PPO 不直接使用单步的 TD Error，而是使用它的指数加权平均：

$$
\hat{A}_t = \sum_{k=0}^{\infty} (\gamma \lambda)^k \delta_{t+k} = \delta_t + (\gamma \lambda)\delta_{t+1} + (\gamma \lambda)^2\delta_{t+2} + \dots
$$

通过推导，可以得到 GAE 的**递归形式**（代码实现常用）：

$$
\hat{A}_t = \delta_t + (\gamma \lambda) \hat{A}_{t+1}
$$

---

### 2.2 深度思考：关键参数理解

### Q1: 为什么要用 GAE？
如果只用 $\delta_t$（即 $\lambda = 0$），优势计算完全依赖于下一时刻的 $V$ 值估计。在训练初期 Critic 极其不准时，会导致巨大的 **Bias**。GAE 通过引入 $\lambda$ 融合了多步的真实奖励，减少了对 Critic 单步预测的过度依赖。

### Q2: 参数 $\lambda$ (Lambda) 的含义？
$\lambda \in [0, 1]$ 用于调节对 Critic 和 真实回报的信任程度：
* **$\lambda = 0$ (高偏差, 低方差)**: $\hat{A}_t = \delta_t$。方差小，但高度依赖 Critic 的准确性。
* **$\lambda = 1$ (无偏差, 高方差)**: $\hat{A}_t = \sum \gamma^k r - V_t$。无偏差，但受随机采样和环境噪声影响严重。
* **$\lambda \approx 0.95$ (折中)**: 既利用了多步奖励校正偏差，又平滑了随机性带来的方差。

---

### 2.3 四个追问

### Q1: 为什么要用重要性采样 **$\rho_t(\theta)$**
- 可以从数学本质、工程困境的角度来思考

### Q2: “未来”和“未来得分的预期”怎么理解？
- PPO 的训练过程，就是不断用“真实发生的未来”去打脸 Critic 的“预期”，迫使 Critic 修正预测，同时迫使 Actor 选择那些让 Critic “惊喜”（结果优于预期）的动作。

### Q3: 为什么一定要建模 $V$ ？为什么不能只用 Reward Model 对每个 token 打分直接作为优势？
- Critic 是去噪的关键。它过滤掉了环境本身的难易度波动，让模型专注于学习策略本身的优劣。

### Q4: 为什么公式里要有 `clip` 和 `min`？
- 直观理解： PPO (Proximal Policy Optimization) 的精髓在于**近端约束**。我们允许策略更新，但必须把新旧策略的差异限制在“安全范围”内，防止策略震荡。

详细解答参考[[1]](https://zhuanlan.zhihu.com/p/1980367577969616247)

---

## 3. GRPO —— 抛弃Critic
Critic 虽然好，但它通常与 Policy 网络一样大。在 70B 参数 BF16 格式的模型上，为了一个标量预测值，要多维护 140GB 的显存，这在 Scaling 阶段是不可接受的。

DeepSeek 提出的 GRPO (Group Relative Policy Optimization)，其核心贡献在于证明了：对于推理任务，Baseline 不需要网络预测，用统计学估计就够了。

### 3.1 第一阶段：DeepSeek-Math 时代的 GRPO
GRPO 不再训练 **Value Network (Critic)**，而是通过 **组内采样** 来构建 Baseline，极大地节省了显存开销。

**算法形式化：**
对于同一个 Prompt $q$，从旧策略 $\pi_{old}$ 中采样 $G$ 个输出 $\{o_1, o_2, \dots, o_G\}$。其目标函数如下：

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q\sim P(Q), \{o_i\}^{G}_{i=1}\sim \pi_{old}} \left[\frac{1}{G} \sum_{i=1}^{G} \left( \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min \left( \rho_{i,t}(\theta) \hat{A}_{i,t}, \text{clip}(\rho_{i,t}(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_{i,t} \right) - \beta D_{KL}(\pi_\theta \| \pi_{ref}) \right)\right]
$$



#### 3.1.1 三个核心技术细节

1. **重要性采样比率 ($\rho_{i,t}$)**
   与 PPO 类似，针对组内每个样本 $o_i$ 的每个 token $t$ 计算新旧策略概率比：
   $$\rho_{i,t}(\theta) = \frac{\pi_{\theta}(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})}$$

2. **优势函数 ($\hat{A}_{i,t}$)：组内相对奖励**
   使用组内 $G$ 个样本奖励的均值和标准差作为 Baseline。对于同一个样本 $o_i$ 中的所有 token，其优势值是共享的：
   $$\hat{A}_{i,t} = \frac{r_i - \text{mean}(r_1, r_2, \dots, r_G)}{\text{std}(r_1, r_2, \dots, r_G)+\epsilon}$$
   *注：这种方式让模型通过“自我博弈”来学习，表现优于组内平均水平的样本获得正向激励。*

3. **KL 散度：无偏估计器**
   DeepSeek-Math 采用了一种基于重要性采样的无偏估计器（K3 估计的修正版），防止策略偏移过大：
   $$D_{KL}(\pi_\theta \| \pi_{ref}) = \frac{\pi_{ref}(o_{i,t} | q, o_{i,<t})}{\pi_\theta(o_{i,t} | q, o_{i,<t})} - \log \frac{\pi_{ref}(o_{i,t} | q, o_{i,<t})}{\pi_\theta(o_{i,t} | q, o_{i,<t})} - 1$$
   *注：这个 KL 项是作为正则项直接减在 Loss 里的，而不是像 PPO 那样扣在 Reward 里。*

---

#### 3.1.2 关键结论：The Nature of RL
DeepSeek-Math 论文通过实验揭示了一个极其重要的现象：

> **RL 显著提升了 Maj@K (多路投票准确率)，但并未显著提升 Pass@K (单次通过率)。**

**深度理解：**
这时候的 DeepSeek 认为，强化学习的主要作用是让模型的输出分布更加 **Robust (鲁棒)**。它更像是一个“重排器”，将正确答案从 Top-K 的候选池中“捞”到了 Top-1 (即更大概率被采样到)，而**并没有本质上增强模型原本不具备的基础能力**。这也为后来 DeepSeek-R1 转向大规模推理强化学习埋下了伏笔。

---

### 3.2 第二阶段：DeepSeek-R1 时代的 GRPO
但是 DeepSeek 在 DeepSeek-R1 中迈出了更大的一步。既然 RL 能提升鲁棒性，那么如果奖励信号足够客观且解空间足够大，RL 能否激发模型涌现出新的能力？

在 DeepSeek-R1-Zero 的实验中，算法依然是 GRPO，但结论被刷新了：

- 不仅是鲁棒性，更是能力的涌现：在没有 SFT 数据冷启动的情况下，纯 RL 训练（DeepSeek-R1-Zero）不仅提升了准确率，还让模型自发学会了 Self-Verification 和 Backtracking。
- Aha Moment：模型在输出中出现了拟人化的顿悟时刻（”Wait, I made a mistake…“）。这证明了在 RLVR (RL with Verifiable Rewards) 场景下，GRPO 配合 Group Exploration，足以支撑模型在庞大的搜索空间中，自行演化出复杂的思维链 (Chain of Thought)。

### 3.3 两个追问

### Q1: GRPO 的优势函数 $\hat{A}_{i,t}$ 为什么是对所有 token 共享的？有什么好处和缺陷？
- 优势来源于整个序列最终Reward，所以是对同一个回答的所有token；好处是降低了计算复杂度，缺陷是信用分配模糊

### Q2: KL 散度采用了 K3 估计形式，为什么不使用 K1 和 K2？
- 从偏差和方差的角度去思考

详细解答参考[[1]](https://zhuanlan.zhihu.com/p/1980367577969616247)

## 4. DAPO —— 业界改进
GRPO 虽然在 DeepSeek 手中大放异彩，但字节在复现时（特别是在 Qwen-32B 上）常常遭遇 “熵坍塌 (Entropy Collapse)” 和 “训练不稳定”。字节提出的 DAPO (Decoupled Clip and Dynamic sampling Policy Optimization) 论文，针对 GRPO 提出了五大改进。

### 4.1 五大改进
DAPO 的目标函数形式化如下：

$$
\mathcal{J}_{DAPO}(\theta) = \mathbb{E}_{q\sim P(Q), \{o_i\}^{G}_{i=1}\sim \pi_{old}} \left[\frac{1}{\sum_{i=1}^{G}|o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} \min \left( \rho_{i,t}(\theta) \hat{A}_{i,t}, \text{clip}(\rho_{i,t}(\theta), 1 - \epsilon_{low}, 1 + \epsilon_{high}) \hat{A}_{i,t} \right)\right]
$$

**改进一：Remove KL Divergence (移除 KL 散度项)**

- 原因：KL 惩罚的初衷是让模型不要偏离参考模型（通常是 SFT 模型）。但在 Long-CoT 的推理任务中，我们的目标是让模型探索出 SFT 模型不具备的复杂推理路径。此时，如果强行用 KL 把模型拉回 SFT 的分布，反而限制了模型的“进化”和探索能力。因此 DAPO 大胆移除了这一项。

**改进二：Clip-Higher (提升截断上限)**

- 问题：传统 PPO/GRPO 使用对称截断 $\epsilon=0.2$。对于低概率的“探索性”Token，其概率提升空间被锁死。
- 改进：解耦截断范围，调高上限 $\epsilon_{high} \gt \epsilon_{low}$ , 例如 $\epsilon_{high}=0.28, \epsilon_{low}=0.2$ 。这允许那些虽然初始概率低、但被证明有效的 Token 获得更大的更新幅度，强行撑开探索空间，防止熵坍塌。

**改进三：Dynamic Sampling (动态采样)**

- 问题：随着模型变强，很多 Batch 中所有样本可能全部做对（Reward 全为 1）或全部做错。此时组内标准差为 0，优势为 0，导致梯度消失。
- 改进：在训练前过滤掉那些全对或全错的 Prompt。如果 Batch 不满，则动态继续采样，直到填满包含有效梯度（组内有对有错）的样本。

**改进四：Token-level Policy Gradient Loss (Token 级 Loss)**

- 问题：GRPO 的公式中包含 $\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|}\sum L_{i,t}$ ，本质上是对 **样本级 (Sample-level)** 求平均。这意味着无论思维链多长，其权重都被 $\frac{1}{|o_i|}$ 稀释了。
- 改进：改为 **Token-level** 求和（见公式最前方的 $\frac{1}{\sum_{i=1}^{G}|o_i|}$ ）。让每个 Token 在梯度中拥有平等的地位，赋予长推理过程更大的权重，同时能更严厉地惩罚长篇胡言乱语。

**改进五：Overlong Reward Shaping (超长奖励重塑)**

- 问题：RL 训练通常有最大长度限制。被截断的样本如果直接给负分，会引入噪声（因为截断前逻辑可能是对的）。
- 改进：引入**软惩罚 (Soft Punishment)**。在达到硬截断之前（如 Max Len - Cache Len 区间），随着长度增加逐渐施加惩罚，引导模型学会自我控制长度。

## 5. GSPO —— 回归序列本质

在 DeepSeek 引领 GRPO 浪潮的同时，Qwen 团队在训练大规模 MoE（Mixture-of-Experts）模型时发现，GRPO 存在严重的训练不稳定性。这促使他们重新审视 GRPO 的数学根基，并提出了 **GSPO (Group Sequence Policy Optimization)**。GSPO 的核心论点非常犀利：**GRPO 在 Token 级别应用重要性采样是理论上的误用**。

### 5.1 GRPO 的理论缺陷：Token 级采样的误用
GRPO 的梯度估计依赖于 Token 级的重要性比率：

$$
\rho_{i,t}(\theta) = \frac{\pi_{\theta}(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})}
$$

Qwen 团队指出，重要性采样的基本原理是通过**对分布进行多次采样并加权平均**来修正分布偏差。然而，在 GRPO 中，对于每一个具体的 $o_{i,<t}$ 历史，我们只采样了一个 next-token $o_{i,t}$ 。

这意味着重要性比率是基于单次采样计算的，它无法起到“分布修正”的作用，反而引入了极大的方差。这种高方差噪声会随着序列长度的增加而累积，并在 `clip` 截断机制下被进一步放大，最终导致训练崩溃。

### 5.2 GSPO 核心算法：序列级重要性采样
既然 Reward 是针对整个序列 (Sequence) 给予的，那么优化目标和重要性比率也应当回归到序列级别。GSPO 提出了基于序列似然的 Importance Ratio：

$$
s_{i}(\theta) = \left( \frac{\pi_{\theta}(o_{i} | q)}{\pi_{\theta_{old}}(o_{i} | q)} \right)^{\frac{1}{|o_{i}|}} = \exp \left( \frac{1}{|o_{i}|} \sum_{t=1}^{|o_{i}|} \log \frac{\pi_{\theta}(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})} \right) 
$$

注意这里引入了 $\frac{1}{|o_i|}$ 进行长度归一化，以防止长序列的概率数值过小导致下溢或方差失控。

GSPO 目标函数：

$$
\mathcal{J}_{GSPO}(\theta) = \mathbb{E} \left[\frac{1}{G} \sum_{i=1}^{G} \min \left( s_{i}(\theta) \hat{A}_{i}, \text{clip}(s_{i}(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_{i} \right)\right]
$$

根本区别：

- GRPO：截断 (Clip) 发生在 Token 级别。每个 Token 根据自己的概率变化决定是否被 Clip。
- GSPO：截断 (Clip) 发生在 Sequence 级别。如果整个序列的生成概率相对于旧策略变化过大，则整个序列的更新被截断。

### 5.3 解决 MoE 训练的“专家抖动”难题
GSPO 的最大实战价值体现在 MoE 模型的训练上。

- 问题：在 MoE 模型中，相同的输入在不同的训练步数下可能会路由（Route）到不同的专家（Experts）。这导致 $\pi_{\theta}$ 和 $\pi_{old}$ 激活的专家网络可能不同。对于 GRPO 这种 Token 级算法，专家路由的微小变化会导致单个 Token 的概率剧烈波动，引发梯度爆炸。为此，GRPO 甚至需要引入 “Routing Replay”（路由重放）这种高成本的工程补丁来强制锁定专家。
- GSPO 的解法：序列的总概率对具体的专家路由路径不敏感。只要模型作为一个整体的语言建模能力保持稳定，序列概率 $s_{i}(\theta)$ 就是平滑的。因此，GSPO 无需 Routing Replay 即可稳定训练超大规模 MoE 模型（如 Qwen3）。

### 5.4 关于 GSPO 的权衡与反思

### Q1: GSPO 将整个序列视为一个整体，是否牺牲了 Token 级的细粒度控制？

- 权衡 (Trade-off)：是的。在 GSPO 中，一个序列内的所有 Token 共享相同的重要性权重。这意味着我们失去了一部分“微操”能力（即区分序列中哪一步推理更偏离旧策略）。
- 收益 (Gain)：换来的是极致的稳定性。对于推理类任务（CoT），逻辑是连贯的，将序列视为最小原子单位往往比纠结于单个连词的概率变化更符合任务本质。对于多轮对话等需要细粒度控制的场景，论文也提出了 GSPO-token 变体作为补充。

### Q2: 为什么 GSPO 的截断比例 (Clipping Fraction) 高达 15%，远超 GRPO 的 0.13%？

- 数据澄清：在 Qwen3 的实验中，GSPO 的 Token 级截断比例约为 15% (0.15)，而 GRPO 仅为 0.13% (0.0013)。两者相差两个数量级。
- 现象解读：这是一个反直觉的现象。通常认为 Clip 比例高意味着很多数据被“浪费”了。但 Qwen 团队认为，GRPO 极低的 Clip 比例恰恰说明其 Token 级 Importance Ratio 充满了高频随机噪声。这些噪声看似没有触发 Clip，但实际上并没有提供有效的梯度方向。
- GSPO 的优势：高达 15% 的截断率表明，算法真正捕捉到了那些显著偏离旧策略的样本，并有效地通过 Clip 限制了它们的影响。这证明了信号的有效性，而非浪费。GSPO 过滤的是“真正的偏离”，而 GRPO 往往被噪声淹没。

## 6. Scaling GRPO —— DeepSeek-V3.2
面对 Scaling Up 过程中的数值不稳定性，DeepSeek-V3.2 技术报告给出了一份“Scaling GRPO”的最终答卷。它没有推翻 GRPO，而是针对观察到的问题，进行了数学修正。

### 6.1 修正一：无偏 KL 估计 (Unbiased KL Estimate)
传统的 KL 散度近似计算（如 K3 估计器）在当前策略 $\pi_\theta$ 与参考策略 $\pi_{ref}$ 差异巨大时（即 $\pi_\theta \ll \pi_{ref}$），其梯度会产生偏差且方差极大，容易导致训练发散。

V3.2 引入了基于重要性采样的无偏估计公式。对于第 $i$ 个样本的第 $t$ 个 token，无偏 KL 计算定义如下：

$$
\mathbb{D}_{KL}(\pi_{\theta}(o_{i,t}) \| \pi_{ref}(o_{i,t})) = \frac{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}{\pi_{old}(o_{i,t}|q,o_{i,<t})} \left( \frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})} - \log \frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})} - 1 \right)
$$

**核心优势：**
- **消除偏差**：利用重要性采样比率对梯度进行校正，确保梯度方向的准确性。与K3相比是无偏的
- **保持稳定**：即使在模型分布发生剧烈变化时，该估计器依然能保持梯度平稳，防止训练崩溃。

### 6.2 修正二：Off-Policy 序列掩码 (Masking)

为了提高训练效率，RL 训练通常采用“一次采样，多次更新”（Multi-step Updates）的策略。然而，随着更新次数增加，当前策略 $\pi_\theta$ 会逐渐偏离采样时的策略 $\pi_{old}$，导致数据变得严重的 **Off-Policy**。

DeepSeek-V3.2 引入了一个二值掩码 $M_{i,t}$，直接作用于 Scaling GRPO 的最终目标函数中，用以过滤掉过时的、偏差过大的样本：

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} M_{i,t} \left( \min \left( \rho_{i,t} \hat{A}_{i,t}, \text{clip}(\rho_{i,t}, 1 - \epsilon, 1 + \epsilon) \hat{A}_{i,t} \right) - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref})_t \right) \right]
$$

其中，重要性采样比率为：

$$
\rho_{i,t} = \frac{\pi_{\theta}(o_{i,t} | \dots)}{\pi_{old}(o_{i,t} | \dots)}
$$

**掩码 $M_{i,t}$ 的逻辑是丢弃那些“离谱的负样本”：**

$$
M_{i,t} = 
\begin{cases} 
0 & \text{if } \mathbb{D}_{KL}(\pi_{old} \| \pi_{\theta}) > \delta \text{ and } \hat{A}_{i,t} < 0 \\
1 & \text{otherwise}
\end{cases}
$$

* **核心逻辑**：如果一个样本已经严重偏离了当前策略（KL 很大），并且它还是个负样本（$\hat{A} < 0$），那就把它 Mask 掉。因为从过度偏离的错误中学习只会引入噪声，模型应该更多地从“成功的探索”或“在策略分布内 (On-Policy) 的错误”中学习。

### 6.3 修正三：保持采样掩码 (Keep Sampling Mask)

这是 V3.2 报告中最晦涩但也最精彩的一个点，解决的是 **Top-p 采样与 RL 训练的数学冲突**。

**冲突点 (Action Space Mismatch)：**
* **推理时 (Rollout)**：开启 Top-p 后，词表尾部的低概率词被 Mask 掉了（概率归零）。 $\pi_{old}$ 是在截断后的子空间里归一化的。
* **训练时 (Train)**： $\pi_{\theta}$ 通常是在全词表空间计算 Softmax 的。

这导致重要性采样比率 $\rho$ 的分母和分子基准不同，梯度估计是歪的。

**解决方案：“存下当时的 Mask！”**
在训练计算 $\pi_\theta$ 时，强制应用推理时一模一样的 Mask。把那些被 Top-p 截断的词的 Logits 设为 $-\infty$，然后再做 Softmax。这确保了 $\pi_{\theta}$ 和 $\pi_{old}$ 在完全一致的动作子空间中竞争，修复了数学上的不一致性。

### 6.4 修正四：保持路由 (Keep Routing)
这与上文 GSPO 提到的 MoE 训练难题（专家抖动）相呼应。DeepSeek 团队同样发现了：由于推理框架和训练框架的实现差异，或者是 RL 训练过程中参数的微小更新，会导致对于同一个 Input，MoE 模型在 Rollout 和 Train 阶段激活的专家（Experts）不一致。

这种“路由不一致”会导致 $\pi_{\theta}$ 和 $\pi_{old}$ 激活的参数子空间发生剧烈漂移，进而引发 Off-Policy 灾难。

DeepSeek 的解法 (Keep Routing)： 简单而暴力。直接记录推理（Rollout）时走的专家路径，在训练时强制锁定走完全一样的路径，强迫模型去优化那些真正被使用过的专家参数。

对比总结： 面对 MoE 训练的不稳定性，Qwen (GSPO) 选择了“避其锋芒”，通过转战序列级优化来规避 Token 级路由抖动的影响；而 DeepSeek (Scaling GRPO) 选择了“直面问题”，通过工程手段强制锁定路由路径。两种思路殊途同归，都保证了大规模 MoE 模型 RL 训练的稳定性。

## 结语：LLM Online RL的演进逻辑

从 **PPO** 到 **Scaling GRPO**，我们可以看到一条清晰的算法演进路线，这本质上是对算力效率、数学严谨性与探索能力的平衡过程：

### 🚀 演进路线全景

* **PPO $\rightarrow$ GRPO：做减法**
    * **核心变化**：去掉笨重的 Critic 网络，利用群组统计（Group Statistics）解决计算瓶颈。
    * **关键发现**：RL 现阶段主要提升了模型的鲁棒性（ $Maj@K$ ），将正确答案从候选池中“捞”到首位，而非本质增强基础能力。

* **GRPO $\rightarrow$ DAPO：做反思**
    * **核心变化**：质疑样本级损失（Sample-level Loss）的合理性。
    * **关键方案**：通过 Token-level Loss 和 **Clip-Higher** 策略解决长链推理中的权重分配与探索问题，并移除了束缚探索的 KL 项。

* **GRPO $\rightarrow$ GSPO：做回归**
    * **核心变化**：指出 Token 级重要性采样的理论缺陷。
    * **关键方案**：回归序列级优化的本质，彻底解决了 MoE（混合专家模型）训练过程中的稳定性问题。

* **GRPO $\rightarrow$ Scaling GRPO：做修正**
    * **核心变化**：全方位修补大规模训练中的数值稳定性漏洞。
    * **关键技术**：引入 **无偏 KL 估计**、**Off-Policy Mask**、**Keep Sampling Mask** 以及 **Keep Routing**，确保数学假设与大规模工程实现的高度一致。

---

### 💡 核心启示

> **Scaling RL 不仅仅是堆算力，更是对算法细节极致严谨的追求。**

无论是 DeepSeek 的 **Scaling GRPO** 还是 Qwen 的 **GSPO**，都证明了：只有修补好这些微小的数学漏洞（如采样空间不一致、梯度偏差等），模型才能在数万步的强化学习中，真正实现由量变引起的质变。

---


## 参考文章
[[1]: LLM的Online RL：从PPO到Scaling GRPO](https://zhuanlan.zhihu.com/p/1980367577969616247)
