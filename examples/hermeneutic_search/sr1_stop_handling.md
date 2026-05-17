# SR1 (Search-R1) 是怎么真正处理 Stop 的

> 调研背景：我们在实验 4 中观察到模型学会用 30+ 次重复 `<answer>freedom</answer>` 打满 1500 tokens 的 reward hack。需要确认 SR1 是不是给 vllm 设了 stop tokens、还是仅靠 postprocess 截断，并把发现转写为我们的修复方案。
>
> 调研对象（zju3）：
> - `/data3/tengee_workplace/Search-R1/search_r1/llm_agent/generation.py`（rollout loop）
> - `/data3/tengee_workplace/Search-R1/verl/workers/rollout/vllm_rollout/vllm_rollout.py`（vllm 封装）
> - `/data3/tengee_workplace/Search-R1/verl/trainer/config/ppo_trainer.yaml`（默认配置）
> - `/data3/tengee_workplace/Search-R1/train_grpo_4gpu.sh`（实际运行命令）
> - `/data3/tengee_workplace/Search-R1/scripts/data_process/nq_search.py`（prompt 模板）
> - `/data3/tengee_workplace/Search-R1/infer.py`（单机推理）
> - `/data3/tengee_workplace/verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py`（我们的 sglang 路径）
> - `/data3/tengee_workplace/verl-sglang-env/lib/python3.12/site-packages/sglang/srt/sampling/sampling_params.py`（sglang stop 字段）

---

## 1. SR1 是否给 vllm 设了 stop tokens？

**结论：训练时不设，只在 postprocess 截断。推理时（`infer.py` 单机版）用 `transformers.StoppingCriteria` 在 token 序列匹配 `</search>` 时停。**

### 训练路径（rollout）— 没有 stop

`verl/workers/rollout/vllm_rollout/vllm_rollout.py` L105-122：

```python
kwargs = dict(
    n=1,
    logprobs=1,
    max_tokens=config.response_length,   # 500
)
# 只把 config 里命中 SamplingParams 字段的参数转发
for k in config.keys():
    if hasattr(SamplingParams(), str(k)):
        kwargs[k] = config.get(k)
self.sampling_params = SamplingParams(**kwargs)
```

`verl/trainer/config/ppo_trainer.yaml` 的 `rollout` 段没有 `stop`、`stop_token_ids` 字段。`train_grpo_4gpu.sh` 也没有 `actor_rollout_ref.rollout.stop=...` 覆盖。

**意味着 vllm 调用时 `stop=None`，不会因 `</search>` 或 `</answer>` 而提前停。一次 `generate()` 一直跑到 `max_response_length=500` token 或自然 EOS。**

### 训练路径（postprocess）— 文本切割

`search_r1/llm_agent/generation.py:54-66` 的 `_postprocess_responses`：

```python
def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
    """Process responses to stop at search operation or answer operation."""
    responses_str = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

    responses_str = [resp.split('</search>')[0] + '</search>'
             if '</search>' in resp
             else resp.split('</answer>')[0] + '</answer>'
             if '</answer>' in resp
             else resp
             for resp in responses_str]
    ...
    responses = self._batch_tokenize(responses_str)
    return responses, responses_str
```

逻辑：

1. vllm 生成 ≤ 500 token 的原始 response（一次性，不停）
2. 解码成字符串
3. **如果存在 `</search>`：取第一个 `</search>` 之前 + 自身**
4. **否则如果存在 `</answer>`：取第一个 `</answer>` 之前 + 自身**
5. 否则保留原文（无效 action）
6. 重新 tokenize

因此「模型在 `</answer>` 后继续生成的 30 次重复」会被这一步直接砍掉，**响应被压缩到第一次 `</answer>` 为止**。后续重复 token 既不进入训练，也不参与 reward 计算。

### 推理路径（`infer.py`）— 真正用 stop

`infer.py:82-84` 用 `transformers.StoppingCriteria` 匹配 token 序列：

```python
target_sequences = [
    "</search>", " </search>", "</search>\n",
    " </search>\n", "</search>\n\n", " </search>\n\n",
]
stopping_criteria = transformers.StoppingCriteriaList(
    [StopOnSequence(target_sequences, tokenizer)]
)
```

注意：推理仅 stop 在 `</search>`（因为搜完要回环），而不在 `</answer>`（答完直接结束循环）。

---

## 2. SR1 的 `max_response_length=500` 怎么和 `max_turns=2` 互相作用？

**结论：每个 turn 一次独立 generate，每次最多 500 token；postprocess 砍 `</search>`/`</answer>`；rolling state 把砍后的 response + 新 observation 拼回原 prompt 再喂下一轮。**

关键参数（`train_grpo_4gpu.sh`）：

| 参数 | 值 | 含义 |
|------|---|------|
| `data.max_prompt_length` | 4096 | 整段 rolling context 上限 |
| `data.max_response_length` | 500 | **单次 generate 的 max_tokens** |
| `data.max_obs_length` | 500 | 每次搜索返回的 information block 上限 |
| `data.max_start_length` | 2048 | 初始 user prompt 上限 |
| `max_turns` | 2 | 主循环最多迭代 2 次（外加一次 final rollout）|

`run_llm_loop` (`generation.py:220-319`) 的结构：

```
turn 0: generate(prompt) → response_500  → postprocess(砍 </search>) → search → obs
turn 1: generate(prompt+resp0+obs0) → response_500 → postprocess → search → obs
        ↑ max_turns=2 退出主循环
# 主循环结束后的 final rollout（do_search=False）
final: generate(...) → 强制不再 search（即使模型还想 search 也忽略）
```

`max_response_length=500` 是**每次 generate 的预算**，而不是总 trajectory 预算。所以 SR1 每个 trajectory 最多生成 `(max_turns+1) * 500 = 1500` 个有效 token（postprocess 砍后还会更少），即使理论上模型可以填满。

`max_prompt_length=4096` 是 rolling context 的硬上限。`_update_rolling_state` 每次都重建并取最后 4096 个 token：

```python
effective_len = new_attention_mask.sum(dim=1).max()
max_len = min(self.config.max_prompt_length, effective_len)
new_rollings = ... new_input_ids[:, -max_len:] ...
```

---

## 3. SR1 的 `state_masking` 是什么？我们有等价机制吗？

**结论：`state_masking` = 让 retrieved information（`<information>...</information>`）不参与 loss。我们也有等价机制，但实现路径不同。**

### SR1 怎么做

设置（`train_grpo_4gpu.sh`）：

```
actor_rollout_ref.actor.state_masking=true
```

`_create_loss_mask` (`verl/trainer/ppo/ray_trainer.py:865-879`)：

```python
def _create_loss_mask(self, batch, metrics):
    response_length = batch.batch['responses'].shape[-1]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    loss_mask = batch.batch['info_mask'][:, -response_length:]
    batch.batch['loss_mask'] = loss_mask
    ...
```

`info_mask` 在哪里构造？看 `generation.py:120-143` 的 `_info_masked_concatenate_with_padding`：传 observation 块时把对应位置全部置为 `pad_id`，于是后面 `_create_loss_mask` 把 `attention_mask` 不为 pad 的位置当作 loss 计算位置。`dp_actor.py:212` 选 `loss_mask`，`dp_actor.py:241` 把 `response_mask = data['loss_mask']` 覆盖。

效果：**模型自己生成的 token loss 算；retrieved information 段（环境注入的）loss 不算**。

### 我们的等价机制

`hermeneutic_agent_loop.py:186-189`（search 分支）：

```python
obs_formatted = f"\n<tool_response>\n{obs_text}\n</tool_response>\n"
obs_ids = self.tokenizer.encode(obs_formatted, add_special_tokens=False)

current_cycle_response_ids.extend(obs_ids)
current_cycle_response_mask.extend([0] * len(obs_ids))   # ← 关键：mask=0
current_cycle_logprobs.extend([0.0] * len(obs_ids))
```

invalid action 反馈（L233-235）和 transform 分支拼接的 new prompt（L273）也都标 `response_mask=0`。

最终 `AgentLoopOutput.response_mask` 直接作为 loss mask 喂给 verl。等价于 SR1 的 `state_masking`，只是命名为 `response_mask`、且不依赖 `info_mask` 中间张量。

---

## 4. SR1 用什么 prompt template？里面是否暗示 `<|im_end|>` 之类的停止符？

**结论：用裸 base 模板，不依赖 Qwen chat template；`<|im_end|>` 不会出现；rollout 时也不 apply_chat_template。**

`scripts/data_process/nq_search.py:23-36`：

```python
if template_type == 'base':
    """This works for any base model"""
    prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
```

整段就是裸文本，没有 `<|im_start|>system\n...<|im_end|>` 的包裹。`base` 变体面向 base 模型（无 chat tuning），所以 `</search>` 是模型唯一可学的停止暗示。

Qwen2.5 系列的 EOS：`infer.py:14` 写 `curr_eos = [151645, 151643]`（`<|im_end|>` + `<|endoftext|>`），但只在单机推理时显式拿来用；训练 rollout 走 vllm 的默认 EOS，配合 postprocess 砍。

我们的 `hermeneutic_agent_loop.py` 用 `apply_chat_template(messages, tools=self.tool_schemas)`（L98, L209-211），所以会插入 `<|im_start|>system\n...<|im_end|>` 等特殊 token。`hermes` format 还会要求 tool_call schema。这是**第一处显著差异**：我们的 prompt 比 SR1 复杂得多，模型需要同时学 chat template + tool_call JSON + 自定义 tag。

---

## 5. 我们 verl + sglang 路径里 sampling_params 怎么设 stop？

**结论：sglang 的 `SamplingParams.__init__` 显式接受 `stop: Optional[Union[str, List[str]]]`，可以从 agent_loop 里把 stop list 加到 `sampling_params` dict 中。**

### sglang 端：什么字段能传

`/data3/tengee_workplace/verl-sglang-env/lib/python3.12/site-packages/sglang/srt/sampling/sampling_params.py:41-69`：

```python
def __init__(
    self,
    max_new_tokens: int = 128,
    stop: Optional[Union[str, List[str]]] = None,           # ← 字符串或 list
    stop_token_ids: Optional[List[int]] = None,             # ← token id list
    stop_regex: Optional[Union[str, List[str]]] = None,     # ← 正则
    ...
    no_stop_trim: bool = False,                             # ← 是否保留 stop 文本
    ...
):
    self.max_new_tokens = max_new_tokens
    self.stop_strs = stop
    if stop_token_ids:
        self.stop_token_ids = set(stop_token_ids)
    ...
```

注意 `no_stop_trim=False`（默认）会把 stop 字符串从输出中去掉。我们想保留 `</answer>` 在响应里，所以应该设 `no_stop_trim=True`。

### verl 端：怎么转发到 sglang

`/data3/tengee_workplace/verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py:358-410`：

```python
async def generate(
    self,
    prompt_ids: torch.Tensor,
    sampling_params: dict[str, Any],
    request_id: str,
    ...
) -> TokenOutput:
    ...
    sampling_params["max_new_tokens"] = max_new_tokens
    return_logprob = sampling_params.pop("logprobs", False)

    request = {
        "rid": request_id,
        "input_ids": prompt_ids,
        "sampling_params": sampling_params,   # ← 整个 dict 透传给 sglang
        "return_logprob": return_logprob,
        ...
    }
    generate_request = GenerateReqInput(**request)
    output = await self.tokenizer_manager.generate_request(generate_request, None).__anext__()
```

**整个 `sampling_params` dict 透传给 `GenerateReqInput`，再被 sglang 内部按字段映射给 `SamplingParams`。**所以只要在 dict 里塞 `"stop": [...]` 就能生效。

### verl 默认构造 sampling_params 的位置

`/data3/tengee_workplace/verl/verl/experimental/agent_loop/agent_loop.py:521-528`：

```python
sampling_params = dict(
    temperature=config.temperature,
    top_p=config.top_p,
    top_k=config.top_k,
    repetition_penalty=1.0,
    logprobs=config.calculate_log_probs,
)
```

默认**不**含 `stop`。最干净的注入点是在 `HermeneuticAgentLoop.run()` 里，进入 generate 调用之前往这个 dict 里加 `stop`。

---

## 三个跟我们当前实现不一致的关键点

| 序号 | 项目 | SR1 | 我们 | 影响 |
|---|------|----|------|----|
| 1 | sampling 时 stop | 不设（依赖 postprocess 截断） | 不设（也不截断） | **我们的模型在一次 generate 里能输出 30+ 个 `</answer>` 直到打满 1500 token，是 reward hack 的物理基础** |
| 2 | 响应文本截断 | `resp.split('</search>')[0] + '</search>'` 在 postprocess 里硬砍 | 没有任何截断；`parse_hermeneutic_action` 只匹配第一个 tag 用来决定 action，但后续生成的 token 都进 response_ids 并算 loss | 同上：哪怕 vllm 不停，砍掉就行；我们没砍 |
| 3 | format_score | `qa_em.compute_score_em` 默认 `format_score=0.`（仅给答对 1.0） | `hermeneutic_qa_em.compute_score` 显式 `format_score=0.1` | 给了 reward 下限 0.1，是 hack 的奖励信号源 |
| 4 | 模型 prompt | 裸 base 模板 + `<search>`/`<answer>` 自定义 tag | `apply_chat_template` + hermes tool_call + `<tool_call>`/`<transform>`/`<answer>` | 我们的 prompt 更复杂；不直接致命，但加剧学习难度 |
| 5 | state_masking | 在 `info_mask` → `loss_mask` 中处理 | 在 `response_mask=0` 中处理（语义等价）| 无差异 |
| 6 | kl_loss_coef | 0.001 | 0.001 | 相同（不是问题根源；KL→0 是塌缩到 base 的结果而非原因）|

---

## 具体修复 patch 示例

下面 patch 直接对应实验 4 诊断中提到的 Tier 1 修复，**只改 `hermeneutic_agent_loop.py` 和 `hermeneutic_qa_em.py` 两个文件**，不动 verl 核心。

### Patch A：给 sglang sampling_params 加 stop sequences

`hermeneutic_agent_loop.py` 在 `run()` 进入主循环之前注入：

```python
# === Patch A: 在 generate 时强制在 close tag 之后停 ===
# sglang 支持 `stop=[...]`（str list）。一次 generate 看到任意 stop string
# 就立刻停，避免模型一次生成 30+ 个重复 </answer>。
# 用 no_stop_trim=True 保留 stop string 在输出里，让 parse 还能命中 tag。
HERMENEUTIC_STOPS = [
    "</tool_call>",
    "</transform>",
    "</answer>",
    "</tool_call>\n",
    "</transform>\n",
    "</answer>\n",
    "</tool_call>\n\n",
    "</transform>\n\n",
    "</answer>\n\n",
]
sampling_params = dict(sampling_params)  # 拷贝以免污染上游
sampling_params.setdefault("stop", HERMENEUTIC_STOPS)
sampling_params.setdefault("no_stop_trim", True)
```

把这段放在 L78 之后（拿到 `sampling_params` 参数立刻处理）。

> 注意：sglang 的 stop 匹配是**字符串**匹配（不是 token 匹配），所以列出常见的 trailing newline 变体能提高命中率。`no_stop_trim=True` 保留这些字符串在输出里，这样 `parse_hermeneutic_action` 的正则才能匹配到。

### Patch B：双保险——postprocess 文本截断

即使 sglang stop 没命中（比如 partial 字符串、tokenizer 拆字），仍要在 Python 层兜底截断。在 `parse_hermeneutic_action` 已经判出 action 之后、把 response 加进 cycle 之前砍掉多余部分：

```python
def truncate_after_first_tag(text: str) -> tuple[str, int]:
    """返回 (截断后文本, 砍掉的字符数). 仿 SR1 _postprocess_responses。"""
    # 按出现的第一个 close tag 截断（保留 tag 自身）
    for tag in ("</tool_call>", "</transform>", "</answer>"):
        idx = text.find(tag)
        if idx != -1:
            return text[: idx + len(tag)], len(text) - (idx + len(tag))
    return text, 0
```

然后在 generate 出来后立刻 apply：

```python
response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
response_text_trimmed, trimmed_chars = truncate_after_first_tag(response_text)
if trimmed_chars > 0:
    # 重新 tokenize 截断后的文本
    response_ids = self.tokenizer.encode(response_text_trimmed, add_special_tokens=False)
    response_logprobs = response_logprobs[: len(response_ids)]
action, content = parse_hermeneutic_action(response_text_trimmed)
```

这一步严格模仿 SR1 `_postprocess_responses` L61-66 的做法。

> 替代实现（不重新 tokenize，更省 CPU）：在 token 层面查找 `</answer>` 对应的 token id 序列再做切片，但需要预 encode 一次拿 token id。SR1 是直接重新 tokenize，简单但偶尔会产生 ±1 个 token 的偏差（合并不同）。我们的 batch 小，直接 retokenize 完全够用。

### Patch C：改 reward — 把 format_score 设为 0

修改 `verl/utils/reward_score/hermeneutic_qa_em.py`：

```python
def compute_score(
    solution_str, ground_truth, data_source=None, extra_info=None,
    format_score=0.0,   # ← 之前 0.1，现在 0.0；与 SR1 一致
    score=1.0, **kwargs
):
    ...
```

或者在训练命令行 override：

```bash
reward.custom_reward_function.format_score=0.0  # 需要 reward function 支持从 kwargs 取
```

但因为 `compute_score` 是被 verl 通过 `**kwargs` 调用、再 forward 到内部，最稳妥还是直接改文件默认值。

### Patch D（建议，非必须）：限制单次 generate max_tokens

当前 `max_response_length=3000`（`run_train_1gpu_test.sh:27`），让模型一次 generate 就能跑出极长输出。SR1 设 `max_response_length=500`（单次预算），多 turn 才靠循环堆叠。建议在 agent_loop 内强制单次预算：

```python
# 单次 generate 最多生成 max_per_call_tokens（< 总 budget），与 SR1 对齐
MAX_PER_CALL_TOKENS = 512
sampling_params.setdefault("max_new_tokens", MAX_PER_CALL_TOKENS)
```

放在 Patch A 同一段。这层是兜底——即便 stop 完全失效，模型也只能在 512 token 内重复，远不到 1500 的崩溃区。

---

## 完整修复路线（与实验 4 诊断对应）

把实验 4 的 Tier 1/2/3 落实到代码层面：

**Tier 1（必须）**

1. Patch A：sglang stop sequences（`</tool_call>`、`</transform>`、`</answer>` 及换行变体）
2. Patch B：Python 兜底截断（仿 SR1 `_postprocess_responses`）
3. Patch C：`format_score=0.1 → 0.0`，断掉 reward hack 的奖励信号源

**Tier 2（应该）**

4. `kl_loss_coef`：0.001 → 0.005，给塌缩更强约束（注意：实验 4 已证明 KL 在塌缩后会 → 0 失效，所以 Tier 2 只是延缓而非根治；Tier 1 仍是必须）
5. `optim.lr_warmup_steps_ratio`：从 0 加到 0.285（与 SR1 一致），减小早期梯度方差
6. 切到 `Qwen2.5-3B-Instruct`：base 模型本身就有重复 `</answer>` 的倾向（chat tuning 会抑制）

**Tier 3（建议）**

7. `n=3 → n=5`：增大 group size，减小 group std=0 的概率
8. Patch D：单次 generate `max_new_tokens=512`，硬性上限
9. 添加 length penalty / 重复 token 检测（如果 Tier 1-2 仍不够，可以在 reward 里减去 `repeat_count > 3` 的 `<answer>` 数 × 0.05）

---

## 引用

- SR1 generation loop：`/data3/tengee_workplace/Search-R1/search_r1/llm_agent/generation.py`
  - L54-75 `_postprocess_responses` — 文本截断逻辑
  - L120-143 `_info_masked_concatenate_with_padding` — info_mask 构造
  - L220-319 `run_llm_loop` — max_turns / final rollout 结构
- SR1 vllm 封装：`/data3/tengee_workplace/Search-R1/verl/workers/rollout/vllm_rollout/vllm_rollout.py:105-122` — `SamplingParams` 构造，无 `stop`
- SR1 PPO trainer：`/data3/tengee_workplace/Search-R1/verl/trainer/ppo/ray_trainer.py:865-879` — `_create_loss_mask`
- SR1 actor：`/data3/tengee_workplace/Search-R1/verl/workers/actor/dp_actor.py:212,241` — `loss_mask` 覆盖 `response_mask`
- SR1 prompt：`/data3/tengee_workplace/Search-R1/scripts/data_process/nq_search.py:23-36` — `base` 模板
- SR1 推理 stop：`/data3/tengee_workplace/Search-R1/infer.py:82-84` — 仅 `</search>` 变体
- SR1 reward：`/data3/tengee_workplace/Search-R1/verl/utils/reward_score/qa_em.py:88-101` — `format_score=0.` 默认
- SR1 训练命令：`/data3/tengee_workplace/Search-R1/train_grpo_4gpu.sh` — 完整 hyperparam
- verl sglang server：`/data3/tengee_workplace/verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py:358-410` — sampling_params 透传
- verl 默认 sampling_params：`/data3/tengee_workplace/verl/verl/experimental/agent_loop/agent_loop.py:521-528`
- sglang SamplingParams：`/data3/tengee_workplace/verl-sglang-env/lib/python3.12/site-packages/sglang/srt/sampling/sampling_params.py:31-95` — `stop`/`stop_token_ids`/`stop_regex`/`no_stop_trim` 字段
- 我们 reward：`/data3/tengee_workplace/verl/verl/utils/reward_score/hermeneutic_qa_em.py` — `format_score=0.1` 默认
- 我们 agent loop：`examples/hermeneutic_search/tools/hermeneutic_agent_loop.py` — 修复目标

---

## sglang 与 vllm 的 stop trim 行为差异

> 补充于实验 5 修复决策之后。Patch A 的「为什么要 `no_stop_trim=True`」需要单独拎出来讲清楚，避免后续被误删/误改。

### sglang `no_stop_trim` 行为（关键！）

sglang `SamplingParams` 默认 `no_stop_trim=False`，意味着：

- 当 stop 字符串触发时，**stop 字符串本身会从输出中砍掉**

举例：

```
模型生成: "<answer>freedom</answer>再生成点别的"
设 stop=["</answer>"]:
  no_stop_trim=False (默认): 输出 "<answer>freedom"           ← </answer> 被砍掉
  no_stop_trim=True:        输出 "<answer>freedom</answer>"   ← 保留闭合 tag
```

源代码位置：`/data3/tengee_workplace/verl-sglang-env/lib/python3.12/site-packages/sglang/srt/sampling/sampling_params.py` 中 `no_stop_trim: bool = False` 默认值。

### 影响

我们的 reward function 用 `re.search(r"<answer>(.*?)</answer>", ...)` 提取答案，**必须有闭合 `</answer>`**。否则：

- regex 匹配失败
- `extract_solution` 返回 None
- reward = 0

也就是说，如果用默认 `no_stop_trim=False`，stop 命中后 `</answer>` 被砍，模型即使「答对」了，reward function 也会判 0。这会让训练直接变成「永远 reward=0」的废训练。

### 修复

`sampling_params["no_stop_trim"] = True` — 让 sglang 保留 stop 字符串。

### 与 SR1 的对比

SR1 用 vllm，vllm 的 stop 默认行为也是 trim。SR1 通过 postprocess `split('</answer>')[0] + '</answer>'` 重新补回闭合 tag（见上文「训练路径（postprocess）— 文本切割」一节，L51-56）。

我们用 sglang，可以选择：

- (A) `stop=[...] + no_stop_trim=True` — sglang 原生保留闭合
- (B) `stop=[...] + postprocess` — 类似 SR1 重新补回

**结论：方案 A 更简洁，不需要额外 postprocess 步骤。**

### 决策表

| 方案 | 输出含闭合 tag | reward 能识别 | 复杂度 |
|------|---------------|---------------|--------|
| 只 A (stop, 默认 trim) | 否 | 否 | - |
| **A + no_stop_trim=True** | 是 | 是 | 最简洁 |
| 只 B (postprocess) | 是 | 是 | 中等 |
| A + B | 是 | 是 | 过度工程 |

最终选 **A + no_stop_trim=True**：在 sglang sampling_params 里同时设 `stop=[...]` 和 `no_stop_trim=True`，不再加 Python 层 postprocess（前面 Patch B 写的双保险在 sglang 原生保留 tag 的前提下可以省掉，留作纯防御性兜底也行）。
