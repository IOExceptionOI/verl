# Hermeneutic Search: Related Work & Technical References

## 核心设计：Context Reset + Block-Diagonal Attention for Multi-Turn RL

我们的做法：rollout 时 `<transform>` 后真正 reset prompt_ids，训练时在同一 sequence 内用 PAD separator + flash-attn `cu_seqlens` 隔离不同 cycle，实现训推一致的 per-cycle 独立 log_prob 计算。

**这个具体组合没有找到已发表的实现。** 各组件单独成熟，但融合方式是新的。

---

## 最相关：Verlog (CMU, NeurIPS 2025)

- **Paper**: [Verlog: Context-lite Multi-turn RL framework](https://openreview.net/forum?id=GmodkWwMV3)
- **Code**: [github.com/WentseChen/Verlog](https://github.com/WentseChen/Verlog)
- **Blog**: [CMU ML Blog](https://blog.ml.cmu.edu/2025/09/15/verlog-a-multi-turn-rl-framework-for-llm-agents/)
- **关键思路**: 每个 turn 当独立 sample，截断旧 context，只保留最近 `memory_length` 轮。发现 memory_length=1 或 2 效果最好。
- **与我们的区别**: Verlog 拆成独立 batch item，我们是打包在同一 sequence 内用 cu_seqlens 隔离。
- **必读** — 对 multi-turn RL 的 context 管理做了系统分析。

## Context Compression：SUPO (ByteDance + Stanford + CMU)

- **Paper**: [Scaling LLM Multi-turn RL with Summarization-based Context Management](https://arxiv.org/abs/2510.06727)
- **关键思路**: 周期性压缩 tool-use history 为 LLM 生成的 summary，然后继续训练。"软 context reset"。
- **与我们的区别**: 我们是硬 reset（context 完全清零），SUPO 是渐进压缩。

---

## Search Agent RL 论文（Reward 设计参考）

### Search-R1
- **Paper**: [arxiv.org/abs/2503.09516](https://arxiv.org/abs/2503.09516)
- **Code**: [github.com/PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1)
- **Reward**: 纯 outcome-based EM（答对1.0，其他0.0），不用 format reward
- **引用**: "we do not incorporate format rewards, as our learned model already demonstrates strong structural adherence"

### R1-Searcher
- **Paper**: [arxiv.org/abs/2503.05592](https://arxiv.org/abs/2503.05592)
- **Reward**: 两阶段 — Stage1 教搜索格式（format+retrieval reward），Stage2 答案驱动+格式惩罚(-2)
- **特色**: F1 score（非 EM），Stage2 format 是惩罚而非奖励

### ReSearch
- **Paper**: [arxiv.org/abs/2503.19470](https://arxiv.org/abs/2503.19470)
- **Reward**: F1 + 格式对给 0.1 小 bonus
- **特色**: 介于 SR1 和 R1-Searcher 之间

---

## cu_seqlens / Block-Diagonal Attention 技术参考

### SFT Packing（成熟方案）
- **Llama-Factory**: [PR #4224](https://github.com/hiyouga/LlamaFactory/pull/4224) — flash_attn_varlen_func 实现无交叉污染的 packing
- **Axolotl**: [Multipack docs](https://docs.axolotl.ai/docs/multipack.html) — monkeypatch attention 使用 varlen
- **HuggingFace**: [PR #31629](https://github.com/huggingface/transformers/pull/31629) — DataCollatorWithFlattening + padding_free
- **HF Blog**: [Packing with FA2](https://huggingface.co/blog/packing-with-FA2)

### RL Training Packing
- **NeMo-RL**: [Sequence Packing docs](https://docs.nvidia.com/nemo/rl/latest/design-docs/sequence-packing-and-dynamic-batching.html) — 明确使用 cu_seqlens_q/kv 传给 FlashAttention，2-3x 加速
- **OpenRLHF**: [github.com/OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — `--ds.packing_samples` 支持 RL 训练 packing

### FlashMask (ICLR 2025)
- **Paper**: [arxiv.org/abs/2410.01359](https://arxiv.org/abs/2410.01359)
- **关键**: Column-wise sparse attention mask，O(N) 内存，支持 block-diagonal + RLHF/RM 训练

---

## 其他 Multi-Turn RL 框架

| 项目 | 链接 | 特色 |
|------|------|------|
| RL-Factory | [github](https://github.com/Simple-Efficient/RL-Factory) | Plug-and-play RL post-training, RolloutHandler 管理 mask |
| RAGEN | [github](https://github.com/mll-lab-nu/RAGEN) | StarPO trajectory-level RL, 发现 "Echo Trap" |
| AgentGym-RL | [github](https://github.com/WooooDyy/AgentGym-RL) | Progressive horizon scaling |
| SWEET-RL | [arxiv.org/abs/2503.15478](https://arxiv.org/abs/2503.15478) | Critic model with additional training-time info |
| TSR (IBM) | [arxiv.org/abs/2602.11767](https://arxiv.org/abs/2602.11767) | Trajectory-search rollouts, optimizer-agnostic |

---

## 我们方案的定位

```
SFT Packing (cu_seqlens)          ← 成熟技术
       +
Verlog per-turn context reset     ← NeurIPS 2025
       +
Hermeneutic question transform    ← 我们的创新点
       =
Same-sequence block-diagonal RL   ← 新的组合（未见发表）
```
