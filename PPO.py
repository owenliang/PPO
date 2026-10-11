# LLM PPO (RLHF) 训练伪代码（agentic 版：整条 agent 轨迹一个总分）
# 流程：跑完整条轨迹（动作与 obs/工具结果交错）→ 轨迹总分 + 逐 token KL 罚 → GAE → 同批数据几轮 PPO 更新
# 两个 mask：traj_mask=真实轨迹 token（定末位、圈 GAE 递推范围）；action_mask=策略生成的动作 token（各项 loss 只消费它）

# ============ 0. 三个模型 + 规则奖励 ============
actor   = LLM.from_pretrained(SFT_CKPT)                # 策略模型（要训练的）
ref     = LLM.from_pretrained(SFT_CKPT); ref.eval(); freeze(ref)     # 参考模型（冻结 SFT），算 KL 用
critic  = LLMWithValueHead.from_pretrained(SFT_CKPT)   # 价值模型：每个 token 估一个 V(s)

# 奖励不训 RM，规则给每条完整轨迹打一个总分
def rule_reward(traj_text) -> float:
    r  = 1.0 * format_ok(traj_text)                    # 格式合规
    r += 1.0 * keyword_hits(traj_text)                 # 关键点命中
    r -= 0.5 * repetition_ratio(traj_text)             # 复读扣分
    r -= 1.0 * over_length(traj_text)                  # 超长扣分
    return r

train_params = list(actor.parameters()) + list(critic.parameters())
opt = AdamW(train_params, lr=1e-6, eps=1e-5)
kl_coef = 0.02        # KL 惩罚强度：别让模型为了刷分跑得离 SFT 太远
clip_eps, ppo_epochs = 0.2, 4                         # PPO 裁剪幅度、每批数据更新几轮
ent_coef = 0.01       # 熵奖励系数：奖励“不确定”，防止策略过早坍缩成单一输出模式


# ============ 工具：逐 token 的 log π(a_t|s_t) 和分布熵 ============
def token_logprobs(model, seq):                  # seq [B,T] → (lp, ent) 各 [B,T]
    # 机关：logits[:, t] 预测 seq[:, t+1]。“token t 的 log 概率”要查位置 t-1 的 logits——整体左移一位
    logits = model(seq).logits[:, :-1]           # [B, T-1, V]
    lp_all = log_softmax(logits, -1)             # [B, T-1, V] log 概率表
    lp  = lp_all.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)   # 查出真实 token 的 log π [B, T-1]
    ent = -(lp_all.exp() * lp_all).sum(-1)       # 位置 t-1 分布的熵，越平（越爱探索）越大 [B, T-1]
    return F.pad(lp, (1, 0)), F.pad(ent, (1, 0)) # 左补一列 0 对齐回 [B,T]（补的列不在 mask 里，不参与训练）


# ============ 工具：GAE，从后往前递推每个 token 的优势 ============
def gae_reverse(rewards, values, traj_mask, bootstrap, gamma, lam):
    # rewards/values/traj_mask [T]，bootstrap 标量 → adv [T]
    # δ_t = r_t + γ·V(下一步) − V(s_t)；A_t = δ_t + γλ·A_{t+1}
    # 递推穿过整条轨迹（动作和 obs 都算数），末位总分才能传回最早的助手 token；
    # mask=0 只会出现在尾部 padding，跳过即可，绝不清零
    adv = zeros_like(rewards)                    # [T]
    lastgae = 0.0
    next_value = bootstrap                       # 轨迹末尾的"下一步价值"：正常终止为 0，截断用 V(last)
    for t in reversed(range(len(rewards))):
        if traj_mask[t] > 0:
            delta = rewards[t] + gamma * next_value - values[t]
            lastgae = delta + gamma * lam * lastgae
            adv[t] = lastgae
            next_value = values[t]
    return adv

# ============ 主循环：一轮 = 采样一批 + 更新几轮 ============
for prompt_batch in repeat_generator(prompt_loader):

    # ---- 1. 采样：跑完整条 agent 轨迹 ----
    # agent_rollout：actor 出动作 token，env/工具回 obs token，交错拼成一条序列
    seq, traj_mask, action_mask = agent_rollout(          # 三者均 [B, T]
        actor, env, prompt_batch, temperature=0.8, top_p=0.95, max_new_tokens=MAX_LEN)
    B, T = seq.shape
    # 定位每条轨迹末位：argmax 只认“第一个最大值”，flip 后一串 1 的第一个即原序列最后一个，再换算回原坐标
    last_idx = T - 1 - traj_mask.flip(-1).float().argmax(-1)      # [B]

    with torch.no_grad():                        # 采样阶段不建计算图
        old_logp, _ = token_logprobs(actor, seq) # 采样时（旧策略）log π，后面算 ratio 的基准
        ref_logp, _ = token_logprobs(ref, seq)   # 参考模型 log π，算 KL 用
        values   = critic(seq).values            # 每个 token 一个 V(s) [B, T]
        trajs      = [decode(s[m]) for s, m in zip(seq, traj_mask)]   # 每条轨迹文本（含 obs）
        traj_score = stack([rule_reward(r) for r in trajs])            # [B]

    # ---- 2. 奖励整形：KL 罚逐 token（只罚动作），轨迹总分落末位 ----
    log_ratio  = old_logp - ref_logp             # log(π_old / π_ref) [B, T]
    # k3 估计：kl_t = e^x − 1 − x，x = log_ratio。恒 ≥ 0（e^x ≥ 1+x）；
    # x 大正（概率暴涨）时 e^x 主导，指数级罚；x 大负（概率暴跌）时 ≈ |x|，线性罚
    kl_t       = log_ratio.exp() - 1 - log_ratio # [B, T]
    # KL 是“策略 vs 参考策略”的偏差，只在动作 token 上有意义；obs 是环境喂的，置 0
    token_r    = where(action_mask, -kl_coef * kl_t, 0.0)          # [B, T]
    token_r[arange(B), last_idx] += traj_score   # 整条轨迹一个总分，加在轨迹末位
    eos_finished  = seq[arange(B), last_idx] == EOS    # 正常终止：末位是 EOS
    len_truncated = ~eos_finished                # 没到 EOS = 撞上最大长度被截断

    # ---- 3. GAE：逐条算优势和回报 ----
    advantages = torch.zeros_like(values)        # [B, T]
    for i in range(B):
        advantages[i] = gae_reverse(
            rewards   = token_r[i],   values = values[i],
            traj_mask = traj_mask[i],
            bootstrap = where(len_truncated[i], values[i, last_idx[i]], 0.0),   # 截断补 V(last)，正常补 0
            gamma=1.0, lam=0.95)
    returns = advantages + values                # [B, T] critic 回归目标（必须在白化前算）
    advantages[action_mask] = whiten(advantages[action_mask])   # 全批动作 token z-score 归一化（只有它们进 loss）

    # ---- 4. PPO 更新：同一批数据训几轮 ----
    # obs token 的优势只是 GAE 传信用的中间量，不产生梯度；三项 loss 全用 action_mask 过滤
    for epoch in range(ppo_epochs):
        for mb in minibatch(seq, old_logp, values, returns, advantages, action_mask):
            new_logp, ent_t = token_logprobs(actor, mb.seq)   # 新策略 log π [mb_B, T] + 熵
            new_v    = critic(mb.seq).values       # [mb_B, T]
            log_ratio  = new_logp - mb.old_logp
            ratio    = log_ratio.exp()             # π_new / π_old [mb_B, T]

            # 裁剪：ratio 偏离 1 太远不再多赚；只对动作 token 取平均（token-mean）
            pg_loss = -torch.min(ratio * mb.adv,
                                 ratio.clamp(1-clip_eps, 1+clip_eps) * mb.adv)
            pg_loss = pg_loss[mb.action_mask].mean()          # 标量

            v_loss  = ((new_v - mb.ret)**2)[mb.action_mask].mean()   # 价值回归，标量

            ent_bonus = ent_t[mb.action_mask].mean()   # 动作 token 平均熵
            loss = pg_loss - ent_coef * ent_bonus + 0.5 * v_loss   # 熵项 = 探索奖励，防过早收敛
            opt.zero_grad(); loss.backward()
            clip_grad_norm_(train_params, 1.0)
            opt.step()
