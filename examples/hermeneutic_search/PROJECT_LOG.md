# Hermeneutic Search — 完整项目操作日志

> 时间: 2026-05-16 至 2026-05-18 (会话 hsr)
> 服务器: zju3, container verl-sr1, GPU 7 (单卡 debug)
> Branch: `hermeneutic-search`

## 一、核心设计

### 任务

基于 verl 0.8 + sglang multi-turn 实现 Hermeneutic Search RL 训练：模型在每个 cycle 内 search → transform → (可选)进入下一 cycle，最终输出 `<answer>`。Transform 时**真正 reset context**，让 cycle 间独立。

### 三个关键创新

1. **HermeneuticAgentLoop** (`tools/hermeneutic_agent_loop.py`)
   - 自定义 agent loop，cycle 间 reset prompt_ids
   - response_ids 紧凑打包多个 cycle，无 PAD separator
   - 通过 `extra_fields["cycle_boundary_positions"]` 传递 cycle 边界给 worker

2. **HermeneuticAgentLoopWorker** (`tools/hermeneutic_agent_worker.py`)
   - 用 verl 官方扩展点 `agent_loop_manager_class`（非 monkey-patch）
   - 重写 `_agent_loop_postprocess`，按 cycle 边界 reset `position_ids` → 0
   - 让 transformers `_is_packed_sequence` 检测到 packing → 自动用 flash_attn_varlen 做 block-diagonal attention，隔离 cycle 间信息

3. **HermeneuticRewardManager** (`tools/hermeneutic_reward_manager.py`)
   - 继承 `NaiveRewardManager` 只加一行：注入 `data.meta_info["global_steps"]` 到 `extra_info` → 让 reward function 能做 step-aware shaping

## 二、Reward 设计演化

### v1 初始（hs_base_kl 达到 reward 0.32-0.70）
```
correct: 1.0
correct + transform: 不区分（仍 1.0）
wrong + valid transform (>10 chars): 0.3 (= 0.1 format + 0.2 bonus)
wrong + no transform: 0.1
no answer tag: 0
```
- 跑 600+ 步，peak 0.70，后期 reward hacking 崩到 0.21
- `<answer>X</answer>` × 30 次刷分

### v2 失败尝试（v3-v8 各种调整都不涨）
- 加 anti-hack：multi-`<answer>` → 0 ✗ 太严格
- 删 format_score：wrong → 0 ✗ base model 完全没信号
- staged format (0.1→0.05→0)：✗ 削弱 bootstrap

### v9 当前（重启 base RL）
```python
# 3-component format reward, 各 0.1, max 0.3 in warmup
format_reward = 0.1 × (has_<answer>_tag + has_valid_<transform> + has_<tool_call>) × scale
# scale: 1.0 (step 0-100), 0.5 (100-200), 0 (200+)
# correct: 1.0 (+0.1 transform_bonus)
# has_valid_transform 阈值: > 10 chars (block "and" hack)
```

## 三、Stop Sequences 之旅

### 尝试 1: sglang `stop=[...]`
**Bug A**: `skip_tokenizer_init=True` 时 sglang `tail_str()` 调 `tokenizer.decode()` 崩溃。verl 默认 skip。

### 尝试 2: `skip_tokenizer_init=False`
**Bug B**: 触发 transformers `prepare_fa_kwargs_from_position_ids` 空 indices_q 崩溃。深层原因是我们 worker `response_start` bug + boundary clip bug。

### 最终方案: SR1 风格 Python postprocess truncate
```python
_MID_TAGS = ("</tool_call>", "</transform>")
_END_TAG = "</answer>"

def truncate_after_first_tag(text):
    # Priority: keep up to FIRST mid-action tag (preserves trajectory).
    # Else cut at first </answer>.
    mid_ends = [text.find(t) + len(t) for t in _MID_TAGS if text.find(t) != -1]
    if mid_ends:
        return text[:min(mid_ends)], True
    idx = text.find(_END_TAG)
    if idx != -1:
        return text[:idx + len(_END_TAG)], True
    return text, False
```

## 四、关键 Bug 修复

### Bug 1: Worker position_ids 4 个相关 bug
- `response_start = len(final_prompt_ids)` → 应该是 `internal.prompt_ids.shape[1]` (padded width)
- `cycle_boundary_positions` 没跟 `response_ids[:max_total_response]` 一起截断
- 写 arange 到 mask=0 的 padding 区域 → unpad 后丢 0
- 缺 monotonic boundary check

修复后：每个 cycle 的第一个 valid token 一定有 position 0，flash-attn 正确隔离。

### Bug 2: KL Loss 完全没生效
- `actor_rollout_ref.actor.use_kl_loss=False` (一开始默认值)
- 没 KL 约束 → 模型 entropy 崩 → grad 爆炸 → reward hacking
- 修复：用 SR1 一样的 `kl_loss=True, kl_loss_coef=0.001, low_var_kl`

### Bug 3: obs 太长吃光预算
- 每次 search 返回 3 docs ~500-1000 token，没截断
- 加 `MAX_OBS_TOKENS=500`（match SR1 `max_obs_length`）

### Bug 4: max_response_length=3000 过宽松
- base model 写满 3000 token 不收敛到 action
- 改回 1500（match hs_base_kl 成功值）

### Bug 5: experiment_name 复用导致自动 resume 旧 ckpt
- verl 通过 `experiment_name` 找 ckpt dir → 同名自动 resume
- 每次大改后必须新 experiment_name + 清旧 ckpt

## 五、verl 扩展点利用

| 功能 | 扩展点 | 文件 |
|------|--------|------|
| 自定义 agent loop | `agent_loop_manager_class` (FQN) | `tools/hermeneutic_agent_worker.py` |
| 自定义 reward function | `reward.custom_reward_function.path/name` | `verl/utils/reward_score/hermeneutic_qa_em.py` |
| 自定义 reward manager | `reward.reward_manager.source=importlib + module.path` | `tools/hermeneutic_reward_manager.py` |
| Agent loop registration | `agent_loop_config_path` yaml | `config/agent_loop_config.yaml` |
| Step 信息传递 | `data.meta_info["global_steps"]` (trainer 自动设) | RewardManager 注入 extra_info |

**所有扩展都通过 verl 官方机制，无 monkey-patch**。

## 六、关键经验/坑

1. **Base model 没 EOS 习惯** → 单次 generate 跑满 max_new_tokens → 必须 truncate
2. **truncate 影响 reward 多样性** → 早期可能反而降低 GRPO 信号
3. **SR1 truncate 优先级**: `</search>` > `</answer>` （我们对应 mid_tags > end_tag）
4. **sglang 多 token stop string 需要 tokenizer**，`skip_tokenizer_init=True` 会崩
5. **`_is_packed_sequence` 只看 position_ids reset**，不看 attention_mask 中间 gap → 修 position_ids 即可
6. **verl 0.8 reward 用 `RewardLoopWorker` (Ray actor)**，print() 不会被 wandb output.log 抓到（旧 verl 的 RewardManager 是 sync 调用，main process print 会被抓）
7. **wandb log_val_generations=N** 只在 val 时生效，需要 `test_freq>0`

## 七、文件结构

```
examples/hermeneutic_search/
├── PROJECT_LOG.md              # 本文件
├── training_notes.md           # 实验 1-4 详细记录
├── sr1_stop_handling.md        # SR1 stop 机制研究
├── varlen_cu_seqlens_explained.md  # flash-attn varlen 解释
├── references.md               # 相关工作引用 (Verlog, SUPO, R1-Searcher)
├── main_hermeneutic.py         # 入口（简单调 verl main）
├── run_train_1gpu_test.sh      # 训练脚本
├── config/
│   ├── hermeneutic_search_grpo.yaml
│   └── agent_loop_config.yaml
├── tools/
│   ├── hermeneutic_agent_loop.py     # 自定义 agent loop
│   ├── hermeneutic_agent_worker.py   # 自定义 worker (position_ids reset)
│   └── hermeneutic_reward_manager.py # 自定义 reward manager (注入 step)
├── data/
│   └── prepare_data.py
└── inference/
    └── eval_hermeneutic.py     # 推理评估
```

## 八、实验编号与结果

| Run | 模型 | format_score | max_resp | 结果 |
|-----|------|--------------|----------|------|
| hwtl5wuv | Instruct | 0.1 固定 | 1500 | step 80 reward 0.32 |
| hs_base_v2 | base | 0.1 + 0.2 trans bonus | 1500 | step 200 peak 0.32, 后期 hacking 崩 |
| hs_base_kl | base | 同上 + KL=0.001 | 1500 | step 240 peak **0.70**, 600 步崩到 0.21 |
| v3-v8 | base | 各种实验性 | 3000 | reward 起步慢、卡死 |
| **v9** | base | **3-comp**, max 0.3 warmup | **1500** | 当前运行 |

## 九、当前训练命令模板

```bash
ssh zju3 "docker exec -d verl-sr1 bash -c '
cd /workspace/verl
WANDB_RUN_ID=hs_v9_\$(date +%m%d%H%M) bash /workspace/verl/examples/hermeneutic_search/run_train_1gpu_test.sh 2>&1 | tee /workspace/verl/hermeneutic_1gpu_test.log
'"
```

## 十、待办

1. v9 跑 200+ 步验证 reward 能否回到 0.30+
2. 等多卡 (5,6,7) 释放后切回 3 GPU
3. 文档化：把 v9 结果加到 training_notes.md (实验 6)
4. 长期: PR 上游 verl，把 packed-sequence RL 支持作为通用功能
