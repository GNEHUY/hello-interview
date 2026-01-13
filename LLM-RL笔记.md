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

### 2.3 四个追问[[1]](@ref)

### Q1: 为什么要用重要性采样 **$\rho_t(\theta)$**
- 可以从数学本质、工程困境的角度来思考

### Q2: “未来”和“未来得分的预期”怎么理解？


## 参考文章
[[1]](@ref): [LLM的Online RL：从PPO到Scaling GRPO](https://zhuanlan.zhihu.com/p/1980367577969616247)
