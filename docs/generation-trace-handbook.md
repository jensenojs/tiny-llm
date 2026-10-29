# 生成链路的逐站记录

屏幕上出现乱码时，这一层给不出任何定位信息：它只说"字节不是合法文本"，不说"字节从哪来"。要定位，只能沿链路往上游走，每一站取一个值，和它应该是什么比。

这份记录把这条链路上的每一站写成一行，行尾带上产生它的代码位置。跑一次，就得到一次前向的完整形状。

## 跑法

```bash
TINY_LLM_TRACE=trace.log .venv/bin/python main.py \
    --solution tiny_llm --loader week1 --model qwen3-0.6b --max-tokens 8
```

不设 `TINY_LLM_TRACE` 就只往 stderr 打。stdout 留给生成的文本，观测不混进被测的通道。

## 站与判据

| 编号 | 站 | 进来什么 | 出去什么 | 判据 |
| --- | --- | --- | --- | --- |
| 0 | args | — | 模型参数 | 与 config 对得上 |
| 1 | prompt | 渲染后的字符串 | 同一串 | 尾部是 assistant 起跑线 |
| 2 | encode | 字符串 | `(L,)` int32 行号 | `decode(ids) == prompt` |
| 3 | model | `(B, L)` int32 | `(B, L, V)` 打分 | 入口 dtype 全程不变；长度 = 已有 token 数 |
| 3 | mask | `L, S` | `(L, S)` | dtype 等于 scores 的 `float32`；第 i 行放行 `i + (S-L) + 1` 列 |
| 4 | logits | `(B, L, V)` | `(V,)` 最后一位 | top3 与 gap |
| 5 | softmax | 打分 | 概率 | 和为 1（bf16 下允许千分位误差）；熵 |
| 6 | sampler | 概率 | 一个 id | 贪心时等于 argmax |
| 7 | new id | id | id 与它的字 | 解出来的字是否接得上 |
| 8 | detokenizer | id | 新增文本段 | 允许空串（半个字符没凑齐） |
| 9 | screen | 文本段 | 屏幕 | 累计字符数 |

第 3 站有两行：model 是入口和总出口，mask 由 attention 在加法那一侧现造，28 层各造一张同形同值的表，打印只留第一层那一次。

## 一次真实运行

这是 Bug2 现场的那次运行：掩码还在向 ids 借类型，第二次前向起 dtype 跟着变成 int64。现在的代码跑不出这一段，留作对照。

前两次前向，其余略：

```
[0 args]        E=1024 V=151936 layers=28 Hq=16 H=8 D=128 I=3072 tie=True  @ src/tiny_llm/qwen3_week1.py:344
[1 prompt]      chars=180  tail='ssistant\n<think>\n\n\n\n'  @ src/tiny_llm/generate.py:173
[2 encode]      n=33  roundtrip=True  tail='\n\n</think>\n\n'     @ src/tiny_llm/generate.py:176
[2 encode]      dtype=mlx.core.int32                              @ src/tiny_llm/generate.py:188
── forward 1  prefill ──────────────────────────────────────      @ src/tiny_llm/generate.py:92
[3 model]       prefill  in=(1, 33)  mlx.core.int32  dtype ✓  len ✓  @ src/tiny_llm/generate.py:98
[3 mask]        mlx.core.int32  zeros/row=[1, 2, 3, 4, 5, 6]  want=[1, 2, 3, 4, 5, 6]
[4 logits]      top3=[32, 39814, 34253]  gap=2.62
[5 softmax]     p=[0.852, 0.062, 0.054]  sum=0.996  H=0.62
[6 sampler]     fn → 32
[7 new id]      32 = 'A'
[8 detokenizer] 'A'
── forward 2  decode ───────────────────────────────────────
[3 model]       decode   in=(1, 34)  mlx.core.int64  dtype ✗ want=mlx.core.int32  len ✓
[3 mask]        mlx.core.int64  zeros/row=[34, 34, 34, 34, 34, 34]  want=[1, 2, 3, 4, 5, 6]
[4 logits]      top3=[7222, 85251, 126556]  gap=0.12
[5 softmax]     p=[0.105, 0.093, 0.036]  sum=1.008  H=6.09
[6 sampler]     fn → 7222
[7 new id]      7222 = 'ş'
[8 detokenizer] 'ş'
```

## 怎么读

第一次前向每一行都正常。第二次前向有五行同时变化，而它们是同一条因果链上的先后：

1. `in` 的类型从 int32 变成 int64，`dtype ✗`。
2. 掩码按输入的类型现造，于是也是 int64；`zeros/row` 每一行都等于序列长度，说明整张表每行都放行所有列。
3. 分布变平：`gap` 从 2.62 掉到 0.12，熵从 0.62 升到 6.09（词表上限约 11.9）。
4. 采样挑中 `7222`，解出来是 `ş`。
5. 屏幕上出现 `ş`。

判断顺序是从上往下：先看入口的类型，再看由它派生的表，最后看后果。反过来读会把症状当成原因。

掩码改到 attention 侧现造、取 `query.dtype` 之后，同样两次前向的 `[3 mask]` 两行都是 `mlx.core.float32`，`zeros/row` 两次都是 `[1, 2, 3, 4, 5, 6]`。入口那行仍报 `dtype ✗`：`argmax` 返回 uint32，拼回句尾时整条数组升成 int64。这一项现在只是观测，不再牵动掩码。

## 在 nvim 里跳

```
set errorformat=%m\ @\ %f:%l
:cfile trace.log
:copen
```

光标停在某条上回车，跳到产生那行日志的代码。缩进的续行不带位置，所以列表里只有站行。

## 这份记录管不到什么

它记录的是"值合不合法、形状对不对"，不记录"这个值在语义上是不是本该如此"。所有检查都会放行的错误，在这里同样一路放行——它只能把链条缩短到两次相邻的站之间，剩下的判断仍要人来下。
