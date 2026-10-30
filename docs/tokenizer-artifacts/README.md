# 分词器产物

~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-MLX-4bit/snapshots/

`prompt 文本 → encode → id 数组` 这一段依赖模型目录里的四五个文件，权重只是其中之一。它们的格式互不相同、字段名相近，混起来就会出现"我明明改了模板却没生效"这类问题。

配套两个脚本，都能直接跑：

```
# 盘点目录
.venv/bin/python docs/tokenizer-artifacts/inspect_snapshot.py
# 模板排版 / 渲染 / 编码
.venv/bin/python docs/tokenizer-artifacts/render_chat_template.py
```

## 目录里有什么

以 `Qwen/Qwen3-0.6B-MLX-4bit` 的本地快照为例（数字由 `inspect_snapshot.py` 当场读出）：

| 文件 | 大小 | 谁读它 | 作用 |
| --- | --- | --- | --- |
| `config.json` | 988B | `AutoConfig`、`mlx_lm.load` | 模型结构：层数、头数、hidden size、`vocab_size` |
| `tokenizer_config.json` | 9.5KB | `AutoTokenizer` | 分词器行为 + `chat_template`（一段 Jinja 源码） |
| `tokenizer.json` | 10.9MB | fast tokenizer | 词表 + 合并规则 + 预分词/归一化/解码配置，全打包 |
| `vocab.json` | 2.6MB | slow tokenizer | `token → id` 字典 |
| `merges.txt` | 1.6MB | slow tokenizer | BPE 合并规则 |
| `model.safetensors` | 302MB | `mlx_lm.load` | 权重 |
| `model.safetensors.index.json` | 48.6KB | `mlx_lm.load` | 权重分片索引 |

`README.md`、`LICENSE`、`.gitattributes` 是仓库元数据，运行时不读。

## 三份词表，运行时只读一份

同一个词表在这个目录里存在三遍：

```
vocab.json + merges.txt     151,643 条 + 151,291 条     slow 路径
tokenizer.json              151,643 条 + 151,387 条     fast 路径
len(tokenizer)              151,669 条                  含 26 个 added token
```

实测运行时走的是 fast 路径：

```
tokenizer class : Qwen2Tokenizer
is_fast         : True
vocab size      : 151669
```

`vocab.json` 和 `merges.txt` 是给 slow 路径留的兼容副本，这次没被读。改词表要改 `tokenizer.json`；只改 `vocab.json` 不会影响 `encode` 的结果。

两处 merges 的数量差 96（151,291 对 151,387），原因没查。不影响本次调用链，记在这里免得下次重新发现一遍。

## chat_template 在哪

模板是一段 **4100 字符的 Jinja 源码**，存在 `tokenizer_config.json` 的 `chat_template` 字段里。同一目录下另外两个 JSON 都不含模板：`tokenizer.json` 放词表和合并规则，`config.json` 放模型结构。

这个模型的 `tokenizer_config.json` 里没有 `chat_template_type` 字段，所以 `mlx_lm` 走 Jinja 渲染。若模型声明了 `chat_template_type`，`mlx_lm/tokenizer_utils.py:620` 会改成导入 `mlx_lm/chat_templates/<type>.py` 里的 Python 函数，那时整条 Jinja 链路都不执行。

## 把模板读成结构

模板从 JSON 里取出来是一整行带 `\n` 转义的字符串，直接读会串在一起。脚本按 Jinja 标签切行、按 `if`/`for` 缩进：

```
{%- if tools %}
  {{- '<|im_start|>system\n' }}
  {%- if messages[0].role == 'system' %}
    {{- messages[0].content + '\n\n' }}
  {%- endif %}
  {{- "# Tools\n\nYou may call one or more functions ..." }}
  {%- for tool in tools %}
    {{- "\n" }}
    {{- tool | tojson }}
  {%- endfor %}
{%- else %}
```

这样设计，是因为对话格式属于模型本身。模型在训练时见过的是 `<|im_start|>role\n...<|im_end|>` 这种排布，推理时的输入必须用同样的格式，模型才能认出角色边界。模板跟权重放在同一个快照目录里，调用方只需要提供 `role`/`content` 消息列表，不需要知道每个模型的格式细节。格式里有条件分支：是否带工具、是否有 system 消息、是否开启思考。消息条数也不固定，所以模板需要 `if`/`for` 这类控制流，单纯的字符串替换写不出来。

两处细节：

`{%- ... -%}` 里的减号是空白控制符，吞掉标签前后的空白。这就是渲染结果里看不到模板缩进的原因。

`{{- '...\n...' }}` 里的 `\n` 属于**渲染结果**，是产出的字符；源码排版用的换行则什么也不产出。把前者替换成物理换行，会让这两层混在一起，渲染结果也就错了。

## 渲染结果与接口

默认参数（与 `main.py` 一致：system + user 两条消息、`add_generation_prompt=True`、`enable_thinking=False`）渲染出 180 字符：

```
«<|im_start|>»system⏎
You are a helpful assistant.«<|im_end|>»⏎
«<|im_start|>»user⏎
Give me a short introduction to large language model.«<|im_end|>»⏎
«<|im_start|>»assistant⏎
«<think>»⏎
⏎
«</think>»⏎
⏎
```

末尾那段 `<think>\n\n</think>\n\n` 来自模板的 `add_generation_prompt` 分支和 `enable_thinking is false` 分支叠加。它是空的思考块，模型从这里往下接。

这 180 个字符就是站 1 的输入。编码后 `n=33`：

```
   idx       id  token
     0   151644  '<|im_start|>'
     1     8948  'system'
     2      198  'Ċ'
     9   151645  '<|im_end|>'
    29   151667  '<think>'
    30      271  'ĊĊ'
    31   151668  '</think>'
    32      271  'ĊĊ'
```

`Ċ` 是一个换行符，`Ġ` 是一个前导空格 —— 这两个是 byte-level BPE 的可读化标记，不对应真实字符。`n=33` 与 `main.py` 跑出来的站 2 日志一致，两边可以互相校验。

`<|im_start|>` 与 `<think>` 能被整体编码成一个 id，靠的是它们登记在 added tokens 里（26 个），而不是靠 special token 标记。实测这个 tokenizer 的 `all_special_tokens` 只有两个元素：`<|im_end|>` 和 `<|endoftext|>`。想在文本里认出全部结构标记，要把 added vocab 一起算上。

## 两处数字要对得上

```
config.vocab_size            151,936   embedding 的行数
len(tokenizer)               151,669   能产出的 id 上界
```

前者是模型权重的形状，后者是分词器的能力。两者不等，差 267 行：embedding 末尾留了 tokenizer 用不到的 padding 行。合法 id 范围是 `0 .. 151,668`。

差值的符号决定后果：`vocab_size > len(tokenizer)` 只是浪费；反过来意味着有 id 没有对应的 embedding，模型会越界读。

`inspect_snapshot.py` 把这三项（`vocab.json`、`tokenizer.json`、`config.json`）都和 `len(tokenizer)` 比了一遍，任一项不相等就直接打出来。

## 边界

- 数字来自 `Qwen/Qwen3-0.6B-MLX-4bit` 的一个快照，换模型全部重跑一遍，别照抄。
- 脚本按 `Qwen2Tokenizer` 的字段组织输出；别的模型家族可能有额外文件（`special_tokens_map.json`、`added_tokens.json`），缺文件时会跳过，不报错。
- 模板渲染走的是 `transformers` 的实现。`mlx_lm` 若被配置成 `chat_template_type` 路径，`render_chat_template.py` 的结果与实际生成时用的模板可能不同，此时以 `chat_template_type` 指向的 Python 函数为准。
