# Prompt 格式与 Chat Template

> 整理自 verl 数据格式探究：为什么 prompt 字段是 message 列表而不是字符串？ChatML / Jinja2 / chat template 是怎么协作的？

## 目录

- [核心问题](#核心问题)
- [两种 LLM API 范式](#两种-llm-api-范式)
- [ChatML 与 `im` 的含义](#chatml-与-im-的含义)
- [Jinja2 模板引擎](#jinja2-模板引擎)
- [Chat Template 的真正目的：三层解耦](#chat-template-的真正目的三层解耦)
- [verl 为什么用 list 格式](#verl-为什么用-list-格式)
- [完整链路](#完整链路)
- [实操验证](#实操验证)

---

## 核心问题

verl 的 GSM8K 数据中 prompt 字段是这样的：

```python
'prompt': [
    {
        'role': 'user',
        'content': 'Natalia sold clips to 48 of her friends...'
    }
]
```

**为什么是个 list 而不是 string？** 直觉上「调用模型 API 不就是传一段文字吗？」

答案：现代 chat 模型 API 几乎都是 list 格式。「传字符串」是老式 completion API 的范式，已经被淘汰。

---

## 两种 LLM API 范式

### 范式 1：Completion API（老式，纯字符串）

GPT-3 时代的 `text-davinci-003`、HuggingFace `pipeline("text-generation")` 都是这种：

```python
prompt = "Translate to French: Hello world"
response = openai.Completion.create(model="text-davinci-003", prompt=prompt)
# 输出: "Bonjour le monde"
```

模型本质上做的是「**给我一段文字，我接着写**」。

要表达「这是用户问题，那是助手回答」只能自己用约定俗成的格式：

```python
prompt = """
User: What is 2+2?
Assistant: 4
User: What about 3+3?
Assistant:"""
```

**问题**：
- 不同人用不同标签（`User:` / `Human:` / `### Q:` / `<|user|>`）
- 模型预训练时没专门学过这种格式，效果不稳定
- multi-turn 对话维护麻烦
- 跨模型不兼容

### 范式 2：Chat Completion API（新式，message list）

GPT-3.5-turbo 之后，OpenAI 推出 Chat Completion API，输入变成 message list：

```python
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user",      "content": "What about 3+3?"},
    ]
)
```

**特点**：
- 不再自己拼字符串，而是结构化地说「这是 system / user / assistant 的什么内容」
- Claude / Gemini / 本地 vLLM/SGLang 兼容接口 / HuggingFace 都用这种格式
- 即使你看到的是字符串接口，底层一定是这个 list

---

## ChatML 与 `im` 的含义

### ChatML 是什么

**ChatML (Chat Markup Language)** 是 OpenAI 在 2023 年设计的一种「对话标记语言」，用一对特殊 token 来分隔每一条消息：

```
<|im_start|>role
content
<|im_end|>
```

- `<|im_start|>` = **instant message start**，标记一条消息的开头
- `<|im_end|>` = **instant message end**，标记一条消息的结尾
- 中间紧跟 `role`（user / assistant / system）和 `content`

### 关键：这些是特殊 token，不是普通字符串

`<|im_start|>` 和 `<|im_end|>` 是模型词表里的**保留 token**，每个对应一个独立的 token id：

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

print(tok.encode('<|im_start|>', add_special_tokens=False))
# [151644]   ← 单个 token id

print(tok.encode('<|im_end|>', add_special_tokens=False))
# [151645]
```

模型在预训练 / SFT 阶段就专门学过：
- 看到 `151644` 之后下一个一定是 role 名称
- 看到 `151645` 表示当前 message 结束

### 不同模型的特殊 token 完全不同

| 模型 | 消息分隔格式 |
|---|---|
| Qwen2.5 | `<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n` |
| Llama 3 | `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>...` |
| Mistral | `<s>[INST] Hello [/INST]` |
| DeepSeek | `<｜begin▁of▁sentence｜>User: Hello\n\nAssistant:` |

Qwen 沿用了 ChatML 格式所以也用 `<|im_start|>` / `<|im_end|>`，但 Llama / Mistral 等模型有自己的特殊 token，名字完全不同。

> 💡 `im` 这两个字母本身无所谓 —— 重要的是它们对应**模型词表里固定保留的 token**，模型「认得」它们，知道这是消息边界。

---

## Jinja2 模板引擎

### Jinja2 是什么

**Jinja2 是一个 Python 模板引擎**：让你写一段「带占位符和控制流的文本」，然后把变量塞进去渲染出最终字符串。

### 最简单的例子

```python
from jinja2 import Template

tpl = Template("Hello, {{ name }}! You are {{ age }} years old.")
print(tpl.render(name="Alice", age=30))
# 输出: Hello, Alice! You are 30 years old.
```

`{{ name }}` 是占位符，`render()` 时被替换。

### 带控制流

```python
tpl = Template("""
{%- if user.is_admin -%}
Welcome, admin {{ user.name }}!
{%- else -%}
Hello, {{ user.name }}.
{%- endif -%}

{%- for item in items %}
- {{ item }}
{%- endfor %}
""")
```

语法元素：
- `{% if %}` / `{% else %}` / `{% endif %}` —— 条件分支
- `{% for %}` / `{% endfor %}` —— 循环
- `{{ var }}` —— 变量插值
- `{%- ... -%}` 中的 `-` —— 去掉前后空白（避免渲染出多余换行）

### Jinja2 的常见用途

| 场景 | 谁在用 |
|---|---|
| HTML 渲染 | Flask、Django |
| 配置文件生成 | Ansible、Helm（K8s） |
| 邮件模板 | 各种 SaaS |
| **LLM chat template** | HuggingFace transformers |
| 代码生成 | OpenAPI codegen |

### 为什么 chat template 用 Jinja2

chat template 本质上是「**根据 messages 列表的内容，按规则生成一段字符串**」—— 这正是 Jinja2 的强项。

每个模型把自己的渲染规则写成 Jinja2 字符串存进 `tokenizer_config.json` 的 `chat_template` 字段，HuggingFace transformers 加载 tokenizer 时读取这段模板，调用 `apply_chat_template()` 时用它渲染。

### Qwen2.5 chat template 实例（节选）

```jinja
{%- if messages[0]['role'] == 'system' %}
    {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
    {%- if message.role == "user" %}
        {{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>assistant\n' + message.content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
```

逻辑：
1. 如果第一条是 system 消息 → 输出 `<|im_start|>system\n...<|im_end|>`
2. 遍历每条消息 → 按 role 输出对应的 ChatML 块
3. 如果要让模型续写 → 末尾加 `<|im_start|>assistant\n` 引子

---

## Chat Template 的真正目的：三层解耦

「适配不同格式」只是表面，更深层的目的是**解耦应用代码和模型实现**。

### 没有 chat template 的世界

```python
# ❌ 应用代码塞满模型相关的 if-else
if model_name.startswith("qwen"):
    text = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
elif model_name.startswith("llama"):
    text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>..."
elif model_name.startswith("mistral"):
    text = f"<s>[INST] {user_msg} [/INST]"
# ... 每加一个新模型都要改代码
```

### 有 chat template 的世界

```python
# ✅ 一行搞定，不管什么模型
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
```

### 三层架构

```
┌─────────────────────────────────┐
│ 应用层：messages 列表             │  ← 开发者只关心这层
│ [{role: 'user', content: '...'}]│     与 OpenAI API 一致
└────────────┬────────────────────┘
             │
             │ apply_chat_template()
             ↓
┌─────────────────────────────────┐
│ 模型层：模型特定的字符串格式        │  ← 模型作者定义
│ <|im_start|>user\n...<|im_end|> │     存在 tokenizer_config.json
└────────────┬────────────────────┘
             │
             │ tokenize()
             ↓
┌─────────────────────────────────┐
│ 计算层：token id 序列              │  ← 模型实际「看到」的东西
│ [151644, 872, 198, ..., 151645] │     直接送进 forward
└─────────────────────────────────┘
```

### 三个层次的好处

1. **开发者写一次代码，可以换任意模型** —— 训练脚本 / 推理应用 / 数据预处理代码全都用 messages 列表，换模型时只改 `model_name`，逻辑不变

2. **模型作者可以自由设计自己的格式** —— SFT 阶段用了什么格式，把对应 Jinja2 写进配置就行，不用迁就上层 API

3. **数据可以跨模型复用** —— 一份数据既能训 Qwen 也能训 Llama，不用每个模型维护一份

### 隐藏好处：避免格式不匹配的灾难

模型在 SFT 阶段是按特定格式训练的，**推理时格式必须完全一致**，差一个 `\n` 都可能让效果暴跌。

真实案例：早期很多人用 Llama 2 chat 时手写 `[INST]` 标签，漏掉了 `[INST]` 和用户消息之间的空格，模型表现明显变差。

`apply_chat_template()` 由 tokenizer 作者维护，能保证**格式 100% 对**。你只传 messages list，剩下的让它处理。

---

## verl 为什么用 list 格式

回到最初的问题：**verl 的 prompt 字段为什么是 list 不是 string？**

因为：
- **数据是 list → 通用、和模型解耦**
- 模型加载时各自带 chat template → 训练时自动转成自己的格式
- 同一份 GSM8K parquet 文件，今天训 Qwen2.5，明天换 Llama3，**数据完全不用改**

如果数据存的是已经渲染好的 string，那一份数据就只能训一个模型 —— 换模型就得重新预处理整个数据集。

### 还有：模型「真正看到」的永远是 token 序列

不管你传 string 还是 list，最终给模型的都是一串 token id：

```
[151644, 8948, 198, 2610, 525, ..., 151645, 198, 151644, 872, 198, ..., 151645]
```

差别只是「**怎么从你的输入构造这个 token 序列**」：
- 老式：你自己拼字符串，tokenizer 转 token
- 新式：你传 list，tokenizer 用 chat template 渲染成字符串再转 token

新式只是把「拼字符串」这一步从开发者手里移到了 tokenizer 内部，由模型作者负责正确性。

---

## 完整链路

```
verl 数据预处理脚本
   ↓
parquet 文件（prompt 字段是 messages list）
   ↓
verl dataloader 读出
   ↓
tokenizer.apply_chat_template(messages)
   ↓ (Jinja2 模板渲染)
模型特定的字符串
"<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n"
   ↓ (tokenize)
token id 序列
[151644, 872, 198, ..., 151644, 77091, 198]
   ↓
actor 模型 forward / 生成 response
```

整个过程中，**每一层的抽象都隐藏了下一层的细节**，让上层不需要知道下层怎么工作。

---

## 实操验证

容器内跑：

```bash
python3 << 'EOF'
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

# 1. 看 chat template 长啥样
print("=== chat_template (前 800 字符) ===")
print(tok.chat_template[:800])
print()

# 2. 看 messages 列表渲染成字符串
msgs = [
    {'role': 'system', 'content': 'You are a math tutor.'},
    {'role': 'user', 'content': 'What is 2+2?'},
]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print("=== 渲染后的字符串 ===")
print(repr(text))
print()

# 3. 看渲染后的字符串 tokenize 成 token id
ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
print("=== token id 序列 ===")
print(ids[:30], '...')
print()

# 4. 看几个特殊 token 的 id
print("=== 特殊 token ===")
print('<|im_start|>:', tok.encode('<|im_start|>', add_special_tokens=False))
print('<|im_end|>  :', tok.encode('<|im_end|>', add_special_tokens=False))
EOF
```

预期输出：
- chat_template 是一段 Jinja2 字符串
- 渲染后字符串包含 `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...`
- token id 序列里能看到 `151644`、`151645` 等特殊 token
- 末尾应该是 `<|im_start|>assistant\n` 的 token（因为 `add_generation_prompt=True`）

### 想看不同模型的渲染差异

```python
from transformers import AutoTokenizer

msgs = [{'role': 'user', 'content': 'Hello'}]

for name in [
    'Qwen/Qwen2.5-0.5B-Instruct',
    'meta-llama/Llama-3.1-8B-Instruct',  # 需要 HF 授权
    'mistralai/Mistral-7B-Instruct-v0.2',
]:
    try:
        tok = AutoTokenizer.from_pretrained(name)
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        print(f'\n=== {name} ===')
        print(repr(text))
    except Exception as e:
        print(f'{name}: skipped ({e})')
```

---

## 一句话总结

> **应用代码用 messages 列表（OpenAI 风格的高层抽象），模型作者用 Jinja2 chat template 把列表渲染成自己专属的字符串格式（含 ChatML 等特殊 token），最后 tokenize 成 token id 喂给模型。**
>
> 这层架构让数据和模型解耦：一份数据训所有模型，一份代码兼容所有模型。

verl 数据格式遵循这个设计 —— `prompt` 字段存的是 messages 列表，训练时自动按当前模型的 chat template 渲染。这就是为什么 GSM8K 同一份 parquet 既能训 Qwen、也能训 Llama、也能训 DeepSeek，**不用动数据**。

---

## 相关概念速查

| 概念 | 是什么 | 在哪里 |
|---|---|---|
| Completion API | 字符串输入，模型续写（老式） | OpenAI text-davinci-003 等 |
| Chat Completion API | messages 列表输入（新式） | OpenAI gpt-4、Claude、Gemini、本地 vLLM 兼容接口 |
| ChatML | OpenAI 设计的对话标记格式 | Qwen 等模型沿用 |
| `<|im_start|>` / `<|im_end|>` | ChatML 的消息边界特殊 token | 模型词表里的保留 token |
| Jinja2 | Python 模板引擎 | HuggingFace 用它实现 chat template |
| chat_template | Jinja2 模板字符串 | `tokenizer_config.json` 的字段 |
| `apply_chat_template()` | tokenizer 方法，渲染 messages → string | HuggingFace transformers API |
| `add_generation_prompt` | 渲染时加上「让 assistant 开始说话」的引子 | apply_chat_template 的参数 |
