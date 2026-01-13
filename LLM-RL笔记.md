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



### 3.2 三个核心技术细节

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

### 3.3 关键结论：The Nature of RL
DeepSeek-Math 论文通过实验揭示了一个极其重要的现象：

> **RL 显著提升了 Maj@K (多路投票准确率)，但并未显著提升 Pass@K (单次通过率)。**

**深度理解：**
这时候的 DeepSeek 认为，强化学习的主要作用是让模型的输出分布更加 **Robust (鲁棒)**。它更像是一个“重排器”，将正确答案从 Top-K 的候选池中“捞”到了 Top-1 (即更大概率被采样到)，而**并没有本质上增强模型原本不具备的基础能力**。这也为后来 DeepSeek-R1 转向大规模推理强化学习埋下了伏笔。

---

## 参考文章
[[1]: LLM的Online RL：从PPO到Scaling GRPO](https://zhuanlan.zhihu.com/p/1980367577969616247)
