# 分组查询注意力（GQA）的数学运算

本文从输入出发，把 GQA 的每一次形状变换推导一遍：一个 token 向量怎么变成 Q/K/V，H_q 个 Q 头怎么和 H 个 KV 头对上，广播在哪里发生，输出怎么合回原样。读完应该能自己写出 `scaled_dot_product_attention_grouped`。

## 记号

| 符号 | 含义 |
| --- | --- |
| B | batch，一批几个序列 |
| L | query 的序列长度 |
| S | key/value 的序列长度 |
| E | 隐藏维（hidden_size），也就是嵌入向量维度 |
| H_q | query 头数（num_heads） |
| H | key/value 头数（num_kv_heads），H_q 的约数 |
| D | 每头维度（head_dim） |
| n_repeats | H_q // H，每个 KV 头被几个 Q 头共享 |

通常 E = H_q * D。K/V 为什么有自己的长度 S，见倒数第二节——先记住算子接口允许 L ≠ S。

## 输入与权重

x 是 token 序列查嵌入表得到的张量。嵌入矩阵的形状是 `(vocab_size, E)`，每个 token 查表得到一个 E 维向量，整批输入就是：

```
x: (B, L, E)
```

这个 E 是全模型统一的表示宽度，叫 hidden_size：嵌入层输出是它，每个模块（注意力、FFN）的输出也是它，层间传递保持 E（本课程和 Qwen 都是嵌入维度 = 隐藏维，同一个数）。为什么进出必须保持 E：残差连接 `x = x + attention(x)` 要求两个操作数同维——完整论证（含 FFN 内部升到约 4E 再缩回的细节）在 Day 1 笔记 `mha-and-test-infra-notes.md` 3.3 节，残差只约束子层的进出口，不约束内部。

四个权重矩阵（MLX/PyTorch 的 Linear 权重形状是 `(out_features, in_features)`）：

```
wq: (H_q * D, E)
wk: (H * D, E)
wv: (H * D, E)
wo: (E, H_q * D)
```

wq 的输出维是 H_q * D，wk/wv 的输出维是 H * D——这个差异就是 GQA 省参数的地方：K/V 的投影矩阵比 Q 小 n_repeats 倍。

## 线性投影：token 变成 Q/K/V 向量

每个 token 是一个 E 维向量。线性投影就是把它乘上一个 `(out, E)` 的矩阵。乘权重矩阵的转置，让 (E) 对上 (E)：

```
q = x @ wq.T
  (B, L, E) @ (E, H_q*D) → (B, L, H_q*D)
q = q.reshape(B, L, H_q, D)
```

reshape 的含义：H_q*D 个数字按行优先顺序切成 H_q 组，每组 D 个，第 i 组就是第 i 个头的向量。

K/V 同理，但输出维是 H*D：

```
k = (x @ wk.T).reshape(B, L, H, D)   # (B, L, H, D)
v = (x @ wv.T).reshape(B, L, H, D)   # (B, L, H, D)
```

关键：K/V 的序列长度来自 x。x 有 L 个 token，每个 token 算出一个 K 向量，所以 K 是「L 个 K 向量排成的序列」。序列长度属于输入数据（有多少 token），不属于权重矩阵——权重矩阵只是转换器，没有长度这个概念。

本文前面的推导以 **prefill** 为例：x 是整段 prompt，Q/K/V 都从它算出来，长度都是 L。**decode** 场景下 x 只有刚生成的一个 token，投影出的 Q/K/V 长度都是 1——随后 K/V 会拼接 cache 里的历史变成 S，Q 不拼、保持 1。两种场景的完整对照见「为什么 K/V 有自己的序列长度 S」一节。

## 换轴：头维和序列维排好

注意力计算的约定布局是 `(B, H, 序列, D)`——头维在前，方便每个头独立算一个 `(序列, D)` 的注意力。

```
q: (B, L, H_q, D) --transpose(0,2,1,3)--> (B, H_q, L, D)
k: (B, L, H, D)   --transpose(0,2,1,3)--> (B, H, S, D)
v: (B, L, H, D)   --transpose(0,2,1,3)--> (B, H, S, D)
```

transpose(0,2,1,3) 交换第 1、2 轴。K/V 这里长度记作 S——如果 K/V 和 Q 来自同一个 x，S = L；decode 阶段 K/V 是拼接后的历史，S ≠ L。

## 形状对齐：H_q 个 Q 头对上 H 个 KV 头

Q 有 H_q 个头，K/V 只有 H 个，直接 matmul 对不上。H_q = H * n_repeats，所以把 Q 的头维拆开：

```
q: (B, H_q, L, D) --reshape--> (B, H, n_repeats, L, D)
```

行优先拆：原来的头索引 i 变成 `(i // n_repeats, i % n_repeats)`。第 0..n_repeats-1 个头归第 0 组，第 n_repeats..2*n_repeats-1 个头归第 1 组，以此类推。

K/V 不拆，而是在头维后面插一个长度 1 的轴：

```
k: (B, H, S, D) --reshape--> (B, H, 1, S, D)
v: (B, H, S, D) --reshape--> (B, H, 1, S, D)
```

现在两边的前三维是 `(B, H, n_repeats)` 和 `(B, H, 1)`——只差中间那个 1。

## 打分：matmul 的广播

matmul 的规则：最后两维做矩阵乘法，前面的维从右往左逐轴比较，相等或一方是 1 就通过（1 被复制成对方的长度）。

```
k_t = swapaxes(k, -1, -2):  (B, H, 1, D, S)
scores = q @ k_t:  (B, H, n_repeats, L, D) @ (B, H, 1, D, S) → (B, H, n_repeats, L, S)
```

拆开看：

- 前三维：`(B, H, n_repeats)` vs `(B, H, 1)` → n_repeats 对 1，广播成 n_repeats，结果 `(B, H, n_repeats)`
- 后两维：`(L, D) @ (D, S)` → `(L, S)`

那个「1 → n_repeats」的复制就是 GQA 的核心：**一个 KV 头被 n_repeats 个 Q 头共享**。数学上就是把同一个 K 矩阵复制 n_repeats 份，分别和组内每个 Q 头打分。

乘 scale：

```
scale = 1 / sqrt(D)
scores = scores * scale
```

（如果调用时传入 scale，用传入的值；没传就用 1/sqrt(D)。）

## 掩码

mask 加到 scores 上：允许的位置加 0，禁止的位置加 -inf（softmax 之后权重为 0）。

mask 可能是三种形态：

- **None**：不加。
- **字符串 "causal"**：内部生成因果掩码 `(L, S)`（Task 2 的 `causal_mask`），再按下面的规则加。
- **数组**：形状可能是 `(L, S)`，也可能是带头维的 `(B, H_q, L, S)`。

加法要过广播，两种数组的处理不同：

- `(L, S)`：直接加。从右往左 `(L,S)` 对 `(B,H,n_repeats,L,S)`，S 对 S、L 对 L，前面自动补 1 复制。
- `(B, H_q, L, S)`：不能直接加——H_q 对 n_repeats，只有 H_q = n_repeats（即 H = 1）时才碰巧通过。要先把 H_q 维拆成 `(H, n_repeats)`：

```
mask: (B, H_q, L, S) --reshape--> (B, H, n_repeats, L, S)
scores = scores + mask
```

## softmax 与加权

softmax 沿最后一维（S 个 key），每个 query 对全部 key 的权重归一化成和 1：

```
attn = softmax(scores, -1):  (B, H, n_repeats, L, S)
```

加权求和：attn 对 V 做 matmul：

```
out = attn @ v:  (B, H, n_repeats, L, S) @ (B, H, 1, S, D) → (B, H, n_repeats, L, D)
```

- 前三维：`(B, H, n_repeats)` vs `(B, H, 1)` → 广播成 `(B, H, n_repeats)`
- 后两维：`(L, S) @ (S, D)` → `(L, D)`

含义：每个 query 位置输出一个 D 维向量，是 S 个 value 向量的加权平均。

## 合头与输出投影

把拆开的两维合回去：

```
out: (B, H, n_repeats, L, D) --reshape--> (B, H_q, L, D)     # H × n_repeats 合并回 H_q
out: (B, H_q, L, D) --transpose(0,2,1,3)--> (B, L, H_q, D)   # 换回序列在前
out: (B, L, H_q, D) --reshape--> (B, L, H_q*D)               # 拼回一个大向量
output = out @ wo.T:  (B, L, H_q*D) @ (H_q*D, E) → (B, L, E)
```

输出形状和输入 x 一样：每个 token 还是 E 维向量，但这个向量混入了它关注到的上下文。

## 完整流程

| 步骤 | 形状 |
| --- | --- |
| 输入 x | (B, L, E) |
| q = x @ wq.T | (B, L, H_q*D) |
| q.reshape | (B, L, H_q, D) |
| q.transpose | (B, H_q, L, D) |
| q.reshape（拆头） | (B, H, n_repeats, L, D) |
| k = x @ wk.T → reshape → transpose | (B, H, S, D) |
| k.reshape（插轴） | (B, H, 1, S, D) |
| v 同 k | (B, H, 1, S, D) |
| scores = q @ kᵀ | (B, H, n_repeats, L, S) |
| scores += mask | (B, H, n_repeats, L, S) |
| attn = softmax(scores) | (B, H, n_repeats, L, S) |
| out = attn @ v | (B, H, n_repeats, L, D) |
| out.reshape → transpose → reshape | (B, L, H_q*D) |
| output = out @ wo.T | (B, L, E) |

## 为什么 K/V 有自己的序列长度 S

要回答这个问题，先看这个算子在什么场景下被调用。

自回归生成的性质是：一次前向只产出**一个** token 的分布。生成 N 个 token，就要跑 N 次前向，每次都要做一次注意力。

最朴素的做法是每次把整个序列（prompt 加上已生成的全部 token）重新喂进去，重算一遍 Q/K/V。但其中大部分是重复计算：前 t-1 个 token 的 K/V 在之前的步骤里已经算过，而且**不会再变**——位置 i 的 K/V 由位置 i 的输入表示决定，这个表示在因果注意力下只依赖位置 1..i，不受后面 token 影响。既然不变，就没有必要重算。

**KV cache** 就是把这个不变量存下来：已生成部分的 K/V 存进缓存，每步只算新 token 的 Q/K/V，然后让新 token 的 Q 去和缓存里全部历史的 K 打分、对全部历史的 V 加权。

为什么缓存 K/V 而不是 Q？Q 是"发问方"：每个位置的 Q 只在计算自己的输出时用一次，用完就不再需要。K/V 是"被查方"：每个位置的 K/V 会被它之后所有位置的 Q 查询。被反复查询的那一侧才值得缓存。

于是同一个注意力算子有两种调用形态：

**prefill（预填充）**：处理完整 prompt 的那一次前向。输入是整段 `x: (B, L, E)`，Q/K/V 都从它算出来，Q 和 K/V 长度相同，`L = S`。这一步顺带把全部 K/V 写进 cache。

**decode（解码）**：之后每生成一个 token 的一次前向。输入只有刚生成的那一个 token，`x: (B, 1, E)`，所以 Q 的长度 `L = 1`。K/V 从 cache 取历史（长度 = 已生成 token 数），再拼上新 token 的 K/V，所以 `S = 历史长度 + 1`。

两种形态下注意力分数的形状也不同（先忽略 batch 和头维）：

- prefill：`(L, S) = (L, L)` 的方阵，每个位置对全部位置打分；
- decode：`(1, S)` 的一行，这个新 token 对全部历史打分，softmax 在 S 个 key 上归一化。

训练时只有 prefill 形态：整段序列一次前向，每个位置同时对全部位置打分（teacher forcing——训练时真实前缀已知，所有位置可以并行算，不必一个一个来）。decode 形态只在推理生成时出现。

同一个算子要同时服务这两种形态，所以接口把 K/V 的长度单独记作 S，而不是沿用 Q 的 L。测试里用 L=3、S=7 就是在模拟长度不等：Q 是 3 个位置、K/V 是 7 个位置，验证实现没有写死 `L = S`。

### decode 场景下的三处差异

前面「完整流程」按 prefill 写的（L = S）。decode 场景逐步走一遍，只有三处实质差异：

**① 输入 token 数从 L 变成 1。** decode 的 x 是 `(B, 1, E)`，投影新算出的 q/k/v 长度都是 1。之后所有形状里 L 的槽位都变成 1，S 保持历史长度。

**② K/V 要拼接 cache。** decode 时投影只产出 1 个新 K/V，把它拼到 cache 的历史后面，才得到长度 S 的 K/V。这一步在投影之后（具体在拆头换轴之前还是之后，取决于 cache 存的是哪种形态）。Q 不拼——历史 token 的 Q 在它们各自那一步已经用完，后续不再需要。

**③ 掩码不需要。** 因果掩码存在的原因是 prefill 一次算 L 个位置的分数，每个位置的 K 里包含它后面的 token，必须遮掉未来。decode 只有一行（新 token），它看到的 S 个 K 全部是历史加自己，没有未来——未来还没生成。所以 decode 天然不需要遮。如果调用时仍然传 `"causal"`，`causal_mask(1, S)` 按偏移规则 `S - L = S - 1` 生成全 0 矩阵，自动退化成不遮。

## 拼车的语义：哪个 Q 头用哪个 KV 头

拆头时的行优先顺序决定了共享关系。原来第 i 个 Q 头，reshape 后落在 `(i // n_repeats, i % n_repeats)`，所以：

```
Q 头 i 使用 KV 头 i // n_repeats
```

例：H_q = 4, H = 2, n_repeats = 2

| Q 头 | KV 头 |
| --- | --- |
| 0 | 0 |
| 1 | 0 |
| 2 | 1 |
| 3 | 1 |

## 与 MHA、MQA 的关系

- H = H_q（n_repeats = 1）：退化为 MHA，每个 Q 头有自己的 KV 头。
- H = 1：MQA（Multi-Query Attention），所有 Q 头共享唯一一个 KV 头。
- 1 < H < H_q：GQA。

参数节省：K/V 投影矩阵从 `(H_q*D, E)` 缩到 `(H*D, E)`，KV cache 也从 H_q 份缩到 H 份。

## 与已有笔记的关系

广播的逐轴对齐规则（从右往左、1 是占位符）在 `docs/tensor-operations-notes.md` 里推导过；本文用到的都是同一条规则。
