# Hermeneutic Search: Training Notes & Lessons Learned

## 实验 1: 无 KL 约束的 GRPO 崩溃 (2026-05-17)

### 配置
- Model: Qwen2.5-3B (base)
- 1x A100-80G, batch=8, n=3 (24 samples/step)
- `use_kl_loss=False`, `grad_clip=1.0`
- format_score=0.1, answer_score=1.0
- wandb run: `hs_base_v2_05171010`

### 现象

| 阶段 | step | reward | entropy | grad_norm |
|------|------|--------|---------|-----------|
| 正常学习 | 1-100 | 0.06→0.25 | 0.7-0.8 | 2-4 |
| 过度自信 | 100-200 | 0.25-0.28 | 降到0.4-0.5 | 3-5 |
| 梯度爆炸 | 200-300 | 0.28→0.15 | 回升0.8-1.0 | spike到15-23 |
| 崩溃 | 300+ | 0.09-0.14 | 飙升1.5-2.3 | 5-15持续高位 |

### 根因分析

GRPO policy gradient: `∇L = -Â · ∇log π_θ(a|s)`

1. **Phase 1 (正常)**: 模型从 base 开始学习 format 和搜索，reward 上升
2. **Phase 2 (过度自信)**: 模型把概率集中到少数 pattern，entropy 下降。因 batch 太小（24），GRPO group normalization 方差估计不准
3. **Phase 3 (爆炸)**: 某些 action 的 `π_θ(a|s)` 被压到接近 0，下一步遇到这个 action 得 reward=1.0 时，`∇log π_θ = ∇π_θ / π_θ` 中 `1/π_θ` 项爆炸。虽然 `grad_clip=1.0` 截断了范数，但梯度方向已经错了
4. **Phase 4 (崩溃循环)**: entropy 升高 → 更多随机探索 → 更多意外 reward → 更大的 advantage/probability ratio → 持续不稳定

### 为什么 KL loss 能防止

- `KL(π_θ || π_ref)` 惩罚模型偏离初始策略
- 不让 `π_θ(a|s)` 对任何 action 压得太低（因为 `π_ref` 给了非零概率）
- 等价于给 `log π_θ` 加下界，防止 `1/π_θ` 爆炸
- 同时防止 entropy 过低（过度自信）

### 小 batch 加剧问题

- 24 samples/step vs SR1 的 1280 samples/step
- GRPO 需要 group 内多个 rollout 来估计 advantage，sample 太少方差极大
- 单个高 reward 样本可能主导整个 batch 的梯度方向

### 修复

```bash
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.kl_loss_type=low_var_kl
```

与 SR1 完全一致。Ref model 是固定的（训练开始时的初始权重，从不更新）。

---

## 实验 2: 加 KL 约束 (2026-05-17)

### 配置
- 同上 + `use_kl_loss=True, kl_loss_coef=0.001, kl_loss_type=low_var_kl`
- `save_freq=100`
- wandb run: `hs_base_kl_05171254`

### 预期
- Entropy 保持稳定（0.5-1.0 范围）
- Grad norm 不再 spike 到 15+
- Reward 稳步上升（虽然慢，因为单卡小 batch）

---

## 关键设计决策记录

### Reward 设计
- **SR1**: 纯 outcome EM (答对1.0，其他0.0)，明确不用 format reward
- **R1-Searcher**: 两阶段（Stage1 教格式+搜索，Stage2 答案驱动+格式惩罚-2）
- **ReSearch**: F1 + format bonus 0.1
- **我们当前**: EM + format_score=0.1（有 `<answer>` tag 但答错=0.1，无 tag=0.0）

### Transform 使用统计 (实验 1, 380 steps)
- cycles=1 (无 transform): 688 (97.2%)
- cycles=2 (1次 transform): 19 (2.7%)
- cycles=3 (2次 transform): 1 (0.1%)
- 答对+用了 transform: 96 次（占答对总数 354 的 27%）

### Invalid Action Feedback
参考 SR1 的设计：当模型没输出有效 tag 时，append 反馈 prompt 让模型重试：
```
My previous action is invalid. If I want to search, I should use <tool_call> with a search query. 
If I want to refine the question, I should put the new question between <transform> and </transform>. 
If I want to give the final answer, I should put the answer between <answer> and </answer>. 
Let me try again.
```
这防止了因格式错误直接终止，给模型更多学习机会。

### Context Reset 实现
- Rollout: `<transform>` 后替换 `agent_data.prompt_ids` 为 `[system + new_question]`
- Training: cycle 间插入 PAD separator → `cycle_attention_patch.py` 设 attention_mask=0 → flash-attn cu_seqlens 自动隔离
- 训推一致性: 两端 model 都只看当前 cycle 的 context

### Ref Model
- 固定不更新（从训练开始到结束始终是初始权重）
- 每 step 计算 `ref_log_prob` 用于 KL loss
- 与 PPO 的定期同步 ref 不同

---

## 实验 3: 修复 position_ids reset 实现真正 cycle 隔离

### 起因

审计发现原 `cycle_attention_patch.py` 的实现有 bug：
- `compute_position_id_with_mask` 用 cumsum-1，**不会** 在 attention_mask gap 处 reset
- 结果：cycle2 的 position_ids 是 `[5,6,7,8]` 而不是 `[0,1,2,3]`
- transformers 的 `_is_packed_sequence` 检测不到 packed，走标准 attention
- cycle2 在训练时能看到 cycle1 → 训推不一致

### 修复

替换 `compute_position_id_with_mask` 为自定义的 `compute_position_ids_with_reset`，在每个 attention_mask=0 gap 处重置 position：

```python
mask:         [0,0,0, 1,1,1,1,1, 0,0, 1,1,1,1, 0,0]
old (cumsum): [0,0,0, 0,1,2,3,4, 4,4, 5,6,7,8, 8,8]  ← 错: cycle2 不 reset
new (reset):  [0,0,0, 0,1,2,3,4, 0,0, 0,1,2,3, 0,0]  ← 对: cycle2 从 0 开始
```

### 为什么这一个改动就够了

1. `_is_packed_sequence(position_ids)` 检测 position_ids 是否非单调
2. 检测到 reset → True → 调用 `_prepare_from_posids` 推导 `cu_seqlens`
3. `cu_seqlens` 传给 `flash_attn_varlen_func` 做 block-diagonal attention

verl 不直接传 cu_seqlens 给 model — model 内部从 position_ids 自动推导。

### 不需要做的

- ❌ 改 verl 核心代码 (dp_actor.py)
- ❌ 传递 cycle_lengths 元数据到 batch
- ❌ 拆 batch 成独立 cycle (Verlog 方式)
- ❌ 用 `unpad_input_for_concatenated_sequences`（不支持 left-padding）

### Patch 实践的反思

当前 monkey-patch 不是最佳实践。短期可接受（快速迭代），长期应改成：
1. PR 上游：作为 verl 的通用 packed-sequence RL 支持
2. Subclass `RayPPOTrainer`：显式重写而非动态注入

### 参考

详见 `varlen_cu_seqlens_explained.md` — varlen 机制完整解释。
