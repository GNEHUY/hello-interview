*本笔记是为了记录关于LLM RL的一些学习笔记*

**RL算法的分类**
- Online RL（需要与在线环境进行交互采样轨迹）：PPO，GRPO，DAPO，GSPO，Scaling GRPO——更好激发模型推理能力和探索未知解空间
- Offline RL（避免在线采样，直接根据数据集优化偏好）：DPO，Rejection Sampling——在对齐阶段效率很高

**PPO——经典“双塔” 之 Actor-Critic**

PPO的核心思想是在优化策略的同时，利用`clip`去限制新旧策略的差异，防止更新步长过大，导致训练崩溃

训练目标：
$$ \mathcal{J}_{PPO}(\theta) = \mathbb{E}_{(q,o) \sim \pi_{old}} \left[ \sum_{t=1}^{T} \min \left( \rho_t(\theta) \hat{A}_t, \text{clip}(\rho_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right] $$

公式说明：
- $\mathcal{J}_{PPO}(\theta)$: PPO 的目标函数，通过最大化它来更新参数 $\theta$。
- $\mathbb{E}_{(q,o) \sim \pi_{old}}$: 表示对旧策略 $\pi_{old}$ 采样得到的轨迹（这里 $q, o$ 通常代表 Query 和 Output）求期望。
- $\rho_t(\theta)$: 重要性采样比率（Importance Sampling Ratio），即 $\frac{\pi_{\theta}(o_t|q,o_{<t})}{\pi_{\theta_{old}}(o_t|q,o_{<t})}$，衡量新旧策略的差异。
- $\hat{A}_t$: 优势函数（Advantage Function）的估计值，衡量当前动作比平均表现好多少。$\hat{A}_t = Q(s_t, o_t) - V(s_t)$
- $\text{clip}(\dots)$: 裁剪函数，将比率限制在 $[1-\epsilon, 1+\epsilon]$ 之间，防止策略更新幅度过大，从而保证训练的稳定性。

TD Error $\delta_t$

全称 Temporal Difference Error（时间差分误差）。按道理有了 $r_t$ 和 $V(s_t)$，我们就可以计算 Advantage 了。根据优势函数的广义定义 $\hat{A}_t = Q(s_t, a_t) - V(s_t)$，定义 TD Error $\delta_t$ 为：$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$其中 $\gamma$ 为 Discount Factor（折扣因子），范围 $[0, 1]$。它决定了模型看多远。$\gamma = 0.99$ 表示模型非常在意未来的收益；$\gamma = 0$ 表示模型只在乎眼前的奖励（目光短浅）。

GAE (Generalized Advantage Estimation)

为了平衡 Bias (偏差) 和 Variance (方差)，PPO 不直接使用单步的 TD Error ($\delta_t$) 作为优势，而是使用它的指数加权平均。GAE 本质上是多步 TD Error 的指数加权求和：$$\hat{A}_t = \sum_{k=0}^{\infty} (\gamma \lambda)^k \delta_{t+k} = \delta_t + (\gamma \lambda)\delta_{t+1} + (\gamma \lambda)^2\delta_{t+2} + \dots$$其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。观察 $\hat{A}_t$ 的展开式，如果我们提取出第二项后的公因子 $(\gamma \lambda)$，剩下的部分恰好就是 $\hat{A}_{t+1}$：$$\hat{A}_t = \delta_t + (\gamma \lambda) \underbrace{\left[ \delta_{t+1} + (\gamma \lambda)\delta_{t+2} + (\gamma \lambda)^2\delta_{t+3} + \dots \right]}_{\text{这正是下一时刻的优势 } \hat{A}_{t+1}}$$由此可以得到 GAE 的递归形式：$$\hat{A}_t = \delta_t + (\gamma \lambda) \hat{A}_{t+1}$$

Q1: 为什么要用 GAE？
- 如果只用 $\delta_t$（即 $\lambda = 0$），优势值的计算完全依赖于下一时刻的 $V$ 值估计。在训练初期 Critic（评论员）很不准时，会导致极大的 Bias。GAE 通过引入 $\lambda$ 融合了多步的真实奖励，减少了对 Critic 单步预测的过度依赖。

Q2: 参数 $\lambda$ (Lambda) 的含义？
- 范围 $[0, 1]$，用于调节对 Critic 和 真实回报 的信任程度： 
- $\lambda = 0$ (High Bias, Low Variance):$\hat{A}_t = \delta_t$。只看这一步。方差极小，但 偏差大（高度依赖 Critic 的准确性）。
- $\lambda = 1$ (No Bias, High Variance):$\hat{A}_t = \sum \gamma^k r - V_t$。看直到结束的所有奖励。无偏差，但 方差大（受随机采样和环境噪声影响严重）。
- $\lambda \approx 0.95$ (折中):既利用了多步奖励校正偏差，又平滑了随机性带来的方差。

参考文章：
1. [LLM的Online RL：从PPO到Scaling GRPO](https://zhuanlan.zhihu.com/p/1980367577969616247)