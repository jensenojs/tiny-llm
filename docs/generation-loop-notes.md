# 生成循环与对照排查

这份笔记只服务一件事：我的 `simple_generate` 在真机上零输出，怎么定位它。所有输出块都是本机实跑粘贴，命令与 prompt 口径一起给出——同一个模型、不同的 prompt 口径会给出完全不同的输出，这一点后面单独说。

## 现场

```python
>>> simple_generate(model, tokenizer, prompt, sampler=sampler, max_tokens=8)
>>>
```

零输出，返回 `None`，不抛异常。同一套测试跑 `pdm run test --week 1 --day 6` 全绿。

## 期望输出：ref 在同一条命令下打出什么

```bash
pdm run main --solution tiny_llm_ref --loader week1 --model qwen3-0.6b --max-tokens 8
```

```
Using tiny_llm_ref solution
Using week1 loader for Qwen/Qwen3-0.6B-MLX-4bit
A large language model (LLM)
```

首次运行会报 `ImportError: cannot import name '_ext'`，先跑 `pdm run build-ext-ref` 编译扩展。

ref 是 tiny_llm 的参照，`mlx_lm` 不是。下文出现 `mlx_lm` 只为一件事：证明上游（权重、tokenizer、模板）健康。

## 上游是好的：mlx_lm 在同一口径下

```python
>>> resp = list(stream_generate(mlx_model, tokenizer, p, max_tokens=8))
>>> [r.token for r in resp]
[32, 3460, 4128, 1614, 320, 4086, 44, 8]
>>> ''.join(r.text for r in resp)
'A large language model (LLM)'
```

与 ref 一致。于是上游排除，病灶在我们的实现里。

## prompt 有两种口径，别混着比

```python
>>> tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
'...<|im_start|>assistant\n'

>>> tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
'...<|im_start|>assistant\n<think>\n\n</think>\n\n'
```

`main.py` 传的是后者（`--enable-thinking` 默认关），模板因此在 assistant 头后补一个空的思考块。前者让模型自己开 `<think>`，生成的第一批 id 就变成 `[151667, 198, 32313, ...]`，文本是 `'<think>\nOkay, the user wants a'`。

同一模型、同一 `max_tokens`，两种口径的输出完全不同。做对照时先确认两边用的是哪个 prompt，否则比出来的差异全部无效。上面 ref 与 mlx 的对照都用第二个口径（33 个 prompt token）。

## 对比顺序：先 id，再 logits，最后才比文本

id 直接反映模型算得对不对；文本还要过解码，两层混在一起就分不清病灶。实测最末一步：

```python
>>> r = resp[-1]
>>> r.token, tokenizer.decode([r.token])
(8, ')')
```

8 个位置逐一列出来：`32 'A'`、`3460 ' large'`、`4128 ' language'`、`1614 ' model'`、`320 ' ('`、`4086 'LL'`、`44 'M'`、`8 ')'`。片段带前导空格，标点与缩写在词表里各自成行。

## GenerationResponse 的字段

一次调用产出 8 个对象，每个 token 一个，字段按产出方分四类：

```python
>>> r.prompt_tokens, round(r.prompt_tps, 1), r.generation_tokens, round(r.generation_tps, 1)
(33, 86.3, 8, 278.8)
>>> r.peak_memory, r.finish_reason, r.from_draft
(0.456, 'length', False)
>>> r.logprobs.shape, r.logprobs.dtype
((151936,), mlx.core.bfloat16)
```

**模型产出**：`token`（词表行号）与 `logprobs`（长度 151936，bfloat16，log-softmax，全负；采样器消耗的就是这条向量）。

**解码器产出**：`text`，本步的增量片段。源码 docstring（`mlx_lm/generate.py:275`）写明可以是空串；`stream_generate` 在 `:729` 与 `:743` 取段。

**运行时仪表**：`prompt_tokens`（33，与 prompt 的 id 数一致）、`prompt_tps`、`generation_tokens`、`generation_tps`、`peak_memory`（GB）。这些数在同一台机器上跨会话会变：另一次会话读出 `prompt_tps=24.09`、`peak_memory=2.108`，本次是 `86.3` 与 `0.456`；差异与冷启动、同时加载的模型数量有关（推断，未做对照实验），跨会话比较没有意义。

**控制面**：`finish_reason` 为 `'length'` 表示撞到 `max_tokens`，`'stop'` 表示模型给了结束符；`from_draft` 标记投机解码草稿。

## 文本与 token 不是一一对应

`text` 允许为空串：一个 token 的字节内容可能是不完整的 UTF-8 字符片段，此时解码器扣住不发，等下一位。所以对比实现时要比 id，不能只比拼出来的字符串——字符串相同也可能源自不同的 id 路径。

## 复验清单

```bash
pdm run build-ext-ref
pdm run main --solution tiny_llm_ref --loader week1 --model qwen3-0.6b --max-tokens 8
```

```python
pdm run python -i debug_repl.py --week 1
simple_generate(model, tokenizer, prompt, sampler=sampler, max_tokens=8)
resp = list(stream_generate(mlx_model, tokenizer, prompt, max_tokens=8))
[r.token for r in resp]
''.join(r.text for r in resp)
```

注意 `debug_repl.py` 构造的 prompt 不带 `enable_thinking=False`，与 `main.py` 的口径不同；要跟 ref 比，先用同样的口径对 `apply_chat_template` 参数。
