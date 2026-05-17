# Varlen Attention, cu_seqlens, position_ids 详解

## 背景：为什么需要 varlen？

LLM 训练中，一个 batch 里序列长度差异大：

```
seq A: 50 tokens
seq B: 200 tokens  
seq C: 800 tokens
seq D: 100 tokens
```

传统做法 — pad 到最长：
```
所有 seq → pad 到 800
计算量: 4 × 800 × 800 = 2.56M token-pair
有效:   50² + 200² + 800² + 100² = 692.5K
浪费率: 73%
```

**Sequence Packing**：把多个短样本拼成一个长 sample：
```
原本: 4 × 800 = 3200 slots
打包: 1 × 1150 = 1150 slots  (节省 64%)
```

## 边界污染问题

但 packing 带来一个严重问题：

```
packed: [seqA tokens, seqB tokens, seqC tokens, seqD tokens]
                       ↑
                      seqB 第一个 token 能 attend 到 seqA 全部
                      → 错误！跨 document 信息泄露
```

标准 causal attention 是连续下三角：

```
        seqA  seqB  seqC  seqD
seqA  [  ✓    -    -    -   ]
seqB  [  ✓    ✓    -    -   ]   ← 看到 seqA (错)
seqC  [  ✓    ✓    ✓    -   ]
seqD  [  ✓    ✓    ✓    ✓   ]
```

需要的是 **block-diagonal causal**：

```
        seqA  seqB  seqC  seqD
seqA  [  ✓    -    -    -   ]
seqB  [  -    ✓    -    -   ]
seqC  [  -    -    ✓    -   ]
seqD  [  -    -    -    ✓   ]
```

## cu_seqlens：边界标记

`cu_seqlens` = **cumulative sequence lengths**（累积序列长度）。在扁平化的 token 序列中标记 document 边界。

```
packed:      [a1, a2, a3, b1, b2, b3, b4, c1]
              └seq A┘ └── seq B ──┘ └─C─┘
              len=3       len=4      len=1

cu_seqlens:  [0,    3,           7,      8]
              ↑     ↑            ↑       ↑
              A起  A结/B起        B结/C起 C结
```

`cu_seqlens` 是累积和。`Doc(i)` 的范围是 `[cu_seqlens[i], cu_seqlens[i+1])`。

`flash_attn_varlen_func` 用 `cu_seqlens` 在 kernel 内部隔离 document：每个 query 只能 attend 到同 document 内的 key。零浪费，零污染。

## position_ids：边界的另一种表达

position_ids 隐含编码了 document 边界 — **每个 doc 起点 reset 为 0**：

```
packed:       [a1, a2, a3, b1, b2, b3, b4, c1]
position_ids: [ 0,  1,  2,  0,  1,  2,  3,  0]
                       ↑          ↑       ↑
                      reset     reset   reset
```

position_ids 同时承担两个职责：
1. **RoPE 位置编码**（每个 doc 内部从 0 开始计算 position）
2. **Document 边界标记**（reset 暗示新 doc 开始）

## transformers 的自动检测机制

```python
def _is_packed_sequence(position_ids, batch_size):
    """Check if position_ids indicates packed sequences."""
    if position_ids is None:
        return False
    expected = torch.arange(seqlen) + position_ids.min()
    # 如果 position_ids 不是单调递增 → packed
    return batch_size == 1 and (expected != position_ids).any()
```

逻辑：
- `[0,1,2,3,4,5,6,7]` → 单调，**False**（不是 packed）
- `[0,1,2,0,1,2,3,0]` → 非单调（有 reset），**True**（packed）

检测到 packed 后，`_prepare_from_posids` 自动从 position_ids 推导 cu_seqlens：

```python
def _prepare_from_posids(query, key, value, position_ids):
    # 找所有 position_ids == 0 的位置 → doc 起点
    starts = (position_ids == 0).nonzero()
    cu_seqlens = build_cu_seqlens_from_starts(starts, total_len)
    return q, k, v, cu_seqlens, max_seqlen
```

然后调用 `flash_attn_varlen_func` 用这个推导出的 cu_seqlens。

## verl + transformers 的完整调用链

```
1. verl dp_actor: unpad_input(input_ids, attention_mask)
   → input_ids_rmpad: [a1,a2,a3,b1,b2,b3] (扁平)
   → position_ids_rmpad: [0,1,2,0,1,2]    ← 关键!
   → cu_seqlens: [0, 6]                    ← verl 不用，丢弃

2. verl: model(input_ids_rmpad, position_ids_rmpad)
   ← attention_mask=None，cu_seqlens 不传

3. transformers (Qwen2 attention):
   _flash_attention_forward(q, k, v, position_ids=position_ids)

4. flash_attention_forward 内部:
   if _is_packed_sequence(position_ids, batch_size=1):
       # 检测到 reset → True
       cu_seqlens = _prepare_from_posids(...)
       # 推导: cu_seqlens = [0, 3, 6]
       
       return flash_attn_varlen_func(q, k, v, cu_seqlens, ...)
   else:
       return flash_attn_func(q, k, v, causal=True)
```

## 应用到 Hermeneutic Search

我们在一个 sample 内打包多个 cycle：

```
cycle1 (Q0 → search → transform): 5 tokens
cycle2 (Q1 → search → answer):    4 tokens

packed:        [c1_1, c1_2, c1_3, c1_4, c1_5, c2_1, c2_2, c2_3, c2_4]
position_ids:  [   0,    1,    2,    3,    4,    0,    1,    2,    3]
                                                  ↑ reset → 隔离 cycle
```

效果：
- cycle2 token 完全看不到 cycle1（block-diagonal attention）
- cycle2 RoPE 从 0 开始（与 rollout 时一致）
- 等效于两次独立 forward pass，只用一次计算

## 核心结论

| 角色 | position_ids | cu_seqlens |
|------|-------------|-----------|
| 粒度 | per-token | per-document |
| 表达 | "我在 doc 内第几个" | "doc 起止在哪" |
| 用途 | RoPE + 边界检测 | flash-attn kernel 参数 |
| 谁产生 | 数据准备阶段 | 自动从 position_ids 推导 |

**关键事实**：在 verl + transformers 路径下，**只需要正确设置 position_ids**，cu_seqlens 由 transformers 自动推导。不需要手动覆盖 cu_seqlens。

## 参考

- [transformers/modeling_flash_attention_utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_flash_attention_utils.py) — `_is_packed_sequence`, `_prepare_from_posids`
- [flash-attn varlen funcs](https://github.com/Dao-AILab/flash-attention) — `flash_attn_varlen_func`
- [Llama-Factory packing](https://github.com/hiyouga/LLaMA-Factory/pull/4224) — SFT packing 实现
- [HF blog: Packing with FA2](https://huggingface.co/blog/packing-with-FA2)
