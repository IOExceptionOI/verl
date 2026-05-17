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

---

## 实验 4: Reward Hacking 崩溃诊断 (2026-05-18)

### 配置（与实验 3 相同）

- Model: Qwen2.5-3B (base)
- 1x A100-80G, batch=8, n=3 (24 samples/step)
- `use_kl_loss=True, kl_loss_coef=0.001, kl_loss_type=low_var_kl`
- `max_response_length=3000`, `max_assistant_turns=3`
- Reward: `format_score=0.1`, `score=1.0` (`hermeneutic_qa_em.compute_score`)
- 完整 position_ids reset + cycle 隔离（实验 3 修复后）
- 共 552 steps，然后崩溃

### 现象 — Metrics 演化

| 阶段 | step | reward | entropy | response_length | clip_ratio | grad_norm | kl_loss | pg_loss |
|------|------|--------|---------|-----------------|------------|-----------|---------|---------|
| 初始学习 | 0-100 | 0.14 → 0.21 | ~0.6 | 中等 | 0.05-0.10 | 1-3 | 0.001-0.003 | 0.01-0.05 |
| 达到峰值 | 100-200 | 0.27 → 0.32 | 0.3-0.4 | 中等 | 0.10-0.15 | 2-4 | 0.005-0.01 | 0.02-0.04 |
| Entropy 塌缩 | 200-400 | 维持 0.32 | 0.08（持续下降）| 缓慢上升 | 0.20-0.40 | 3-6 | 0.02 | 0.01-0.02 |
| Length 飙升 | 400-500 | 0.30-0.32 | 0.05 | **急剧上升到 1500** | 0.50-0.80 | 4-10 | 0.03 | 0.005-0.01 |
| 完全崩溃 | 500-552 | 0.10-0.15 | **0.03** | **打满 1500** | **1.0** | **0.03**（接近0）| 0.04 | **0**（梯度冻结）|

### 真实失败模式 — 抓到的崩溃样本

模型学到的 reward hack：在**一次** `generate()` 内输出 30+ 次 `<answer>freedom</answer>` 重复，直到 token 数填满 `max_response_length`。例：

```
<answer>freedom</answer><answer>freedom</answer><answer>freedom</answer>
<answer>freedom</answer> ... (重复 30+ 次直到 ~1500 token)
```

由于我们的 `parse_hermeneutic_action` 只匹配**第一个** `<answer>` 来决定 action，模型确实终止在 cycle 1；但后续 token 已全部 append 到 `response_ids` 并加入 loss 计算。

### 因果链

1. **Reward 下限 = 0.1**：`format_score=0.1` 永远拿得到（只要有 `<answer>` tag）。哪怕答错，模型也保证 ≥ 0.1 的 reward 信号
2. **GRPO group std → 0**：`n=3` 时，group 内所有 3 个 rollout 都收敛到同一个 `<answer>X</answer>` hack（X 经常是 base 模型先验偏好的高频词，如 `freedom`、`yes`、`no`、`USA`）→ group reward std = 0 → **advantage = 0**
3. **pg_loss = 0**：PPO 公式 `pg_loss = -mean(advantage * ratio)`，advantage=0 时 pg_loss 严格为 0
4. **只剩 KL 梯度做工作**：但是
5. **模型已塌缩到 base 模型的重复模式**：base 模型本身就会偏向复制重复 token（无 chat tuning 抑制重复），actor 学到这个 hack 后实际上 = base 模型行为
6. **KL(actor||ref) → 0**：actor 和 ref 在这个塌缩区域分布几乎相同（都是 base） → KL 项也归零
7. **所有梯度归零** → 数学上训练冻结：`grad_norm=0.03`（≈ 噪声）、`pg_loss=0`、`kl_loss=0.04`（k3 estimator 的噪底）

### 五个缺失保护（与 SR1 对比）

| 项 | 我们 | SR1 | 说明 |
|----|------|-----|------|
| **vllm/sglang stop sequences** | 无 | 无（但有 postprocess 兜底）| 详见 `sr1_stop_handling.md` |
| **响应文本截断** | 无 | `_postprocess_responses` 切割在 `</search>`/`</answer>` | **最关键缺失**：我们让模型一次 generate 输出 30+ 个 `</answer>` 完全不砍 |
| **format reward** | 0.1 给到（错答也给）| 0.0（默认 `compute_score_em(format_score=0.)`）| 给了 reward 下限 0.1，是 hack 的奖励源 |
| **kl_loss_coef** | 0.001 | 0.001 | 相同（KL 不是问题根源，而是塌缩的副产物）|
| **模型** | Qwen2.5-3B base | Qwen2.5-3B base / 我们的 Tier 2 应切 Instruct | base 模型本身有重复 token 倾向 |

### `kl_loss → 0` 的真实含义

`use_kl_loss=True, kl_loss_type=low_var_kl` 用的是 k3 estimator：

```python
# low_var_kl (k3)
kld = exp(kl) - kl - 1   # kl = log_prob - ref_log_prob
```

这是个 **non-negative 距离估计**（Schulman blog: http://joschu.net/blog/kl-approx.html）。`kld → 0` 意味着 actor 和 ref 在被采样到的 token 分布上**一致**。

在我们的场景里这**不是好事**，因为：

- ref = 训练初始的 Qwen2.5-3B base
- actor 已塌缩到 base 自带的「复制 `<answer>` token」退化 loop
- 两者在这个 loop 上的 token 分布天然相同 → KL=0

所以「KL 接近 0」不代表「我们没偏离 base」，而是「我们和 base 都退化到了同一个洞」。这反过来证明：**KL 约束在塌缩之后丧失作用，必须在塌缩之前用 stop sequences + 响应截断 + 改 reward 截断 reward hack 的物理路径**。

### 修复优先级（详）

**Tier 1（必须，直接阻断 hack 物理路径）**

1. **加 stop sequences**（sglang `sampling_params` 加 `stop=["</answer>", "</transform>", "</tool_call>", ...]` 及换行变体；`no_stop_trim=True` 保留 tag）
2. **响应文本截断**（仿 SR1 `_postprocess_responses`：decode → split at first close tag → 重新 tokenize）
3. **改 reward**：`format_score=0.1 → 0.0`（断掉 hack 的奖励信号源）

**Tier 2（应该，降低塌缩概率）**

4. `kl_loss_coef`：0.001 → 0.005（延缓塌缩，但不能根治）
5. `optim.lr_warmup_steps_ratio`：0 → 0.285（与 SR1 一致），减小早期梯度方差
6. 切到 `Qwen2.5-3B-Instruct`（chat tuning 抑制重复 token，base 模型本身就有 `<answer>` 复制倾向）

**Tier 3（建议，加固鲁棒性）**

7. `n=3 → n=5`：增大 group size，减小 group std=0 概率
8. 单次 generate `max_new_tokens=512`（与 SR1 一致），多 turn 才靠循环堆叠
9. Length penalty / 重复检测：在 reward 里减去 `repeat_count(<answer>) > 3` × 0.05（如果 Tier 1-2 仍不足）

### 详细修复 patch 示例

参见 [`sr1_stop_handling.md`](./sr1_stop_handling.md) — 包含 SR1 stop 处理机制全调研 + Patch A/B/C/D 具体代码示例（加在 `hermeneutic_agent_loop.py` 的哪一行）。

---

## 实验 5: sglang stop sequences 修复决策 (2026-05-18)

### 背景

实验 4 诊断出 reward hack 的物理路径之一是：模型一次 generate 能输出 30+ 次 `</answer>` 直到打满 token budget。Tier 1 修复需要在 sglang 层面加 stop sequences。本次实验落地这个修复。

### 关键发现：sglang 默认 `no_stop_trim=False`

调研 sglang `SamplingParams` 时发现一个容易踩的坑：

- sglang `SamplingParams` 默认 `no_stop_trim=False`
- 含义：stop 字符串命中后，**stop 字符串本身会从输出中砍掉**
- 例：`stop=["</answer>"]` + 模型生成 `<answer>freedom</answer>...`，**默认输出为 `<answer>freedom`**（闭合 tag 被吃掉）

### 为什么这是个问题

我们的 reward function：

```python
re.search(r"<answer>(.*?)</answer>", solution_str)
```

**必须有闭合 `</answer>`** 才能匹配。否则：

- regex 匹配失败 → `extract_solution` 返回 None → reward = 0
- 整个训练变成「永远 reward=0」的废训练（比实验 4 的 reward hack 还糟）

### 修复决策

在 `hermeneutic_agent_loop.py` 的 sglang sampling_params 注入处同时加两个字段：

```python
sampling_params["stop"] = HERMENEUTIC_STOPS   # ["</tool_call>", "</transform>", "</answer>", + \n 变体]
sampling_params["no_stop_trim"] = True        # ← 关键：保留闭合 tag
```

### 与 SR1 (vllm) 路径的对比

SR1 用 vllm，vllm 的 stop 默认行为也是 trim。SR1 的处理方式是：

- **不给 vllm 设 stop**（生成跑满 max_tokens）
- **Python 层 postprocess** `split('</answer>')[0] + '</answer>'` 重新补回闭合 tag

我们用 sglang，理论上有四种组合：

| 方案 | 输出含闭合 tag | reward 能识别 | 复杂度 |
|------|---------------|---------------|--------|
| 只 A (stop, 默认 trim) | 否 | 否 | - |
| **A + no_stop_trim=True** | 是 | 是 | **最简洁** |
| 只 B (postprocess) | 是 | 是 | 中等 |
| A + B | 是 | 是 | 过度工程 |

**最终选 A + no_stop_trim=True**：sglang 原生保留闭合 tag，不需要额外 postprocess 步骤。Patch B（postprocess 兜底）在 sglang 原生保留 tag 的前提下可省，留作纯防御也可以——但当前选不加，保持最小修改面积。

### 与实验 4 Tier 1 修复路线的对齐

| 实验 4 Tier 1 项 | 实验 5 实际修复 | 备注 |
|---|---|---|
| 1. sglang stop sequences | Patch A：`stop=[...] + no_stop_trim=True` | 用 sglang 原生 trim 行为修正 |
| 2. 响应文本截断（postprocess） | **暂不加**（Patch A 已经在 sglang 内部停了，重复不会再产生）| 留作未来防御性兜底 |
| 3. `format_score=0.1 → 0.0` | 单独 patch reward function | 与 stop 修复正交 |

### 详细说明

参见 [`sr1_stop_handling.md`](./sr1_stop_handling.md) 新增的「## sglang 与 vllm 的 stop trim 行为差异」一节——记录了完整决策表和与 SR1 的对比。

### 一句话总结

> sglang 默认会砍掉 stop 字符串。我们的 reward function 依赖闭合 `</answer>`，所以必须显式 `no_stop_trim=True`。这是用 sglang 替代 vllm 时最容易踩的坑。

---

## 文档索引

- [`training_notes.md`](./training_notes.md)（本文）— 实验时间线 & lessons learned
- [`references.md`](./references.md) — Multi-turn RL / cu_seqlens / search-agent 相关工作引用
- [`varlen_cu_seqlens_explained.md`](./varlen_cu_seqlens_explained.md) — block-diagonal attention 机制详解
- [`sr1_stop_handling.md`](./sr1_stop_handling.md) — SR1 (Search-R1) 怎么处理 stop sequences 调研 + 我们的具体修复 patch 示例（针对实验 4 崩溃）
