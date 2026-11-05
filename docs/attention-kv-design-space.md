# 注意力的键值存放: MHA、GQA、MQA、MLA 的比较

> 自回归生成的 decode 阶段, 每产生一个 token, 注意力层都要把此前全部位置的键 (key) 与值 (value) 从内存读一遍. 随着上下文增长, 由上下文生成的 KV Cache 大小及其单步读取量，会超过模型权重本身, 成为每一步的主要开销; 它们占的内存也决定了一台机器能同时容纳多长的上下文. MHA、GQA、MQA、MLA 是缩小这部分存储的四种做法. 四者都受同一条形状约束: 每个头的查询 (query) 与键必须等宽, 因为两者要做点积. 四者也都要和模型效果相权衡. MLA 改变了存放的内容, 又带出第二条约束. 本篇从读取量出发, 先找出能改的因子, 再依次看减少份数 (MHA → GQA → MQA) 与改变存放物 (MLA) 两条路各自换来什么、付出什么, 最后标出四者的等价处与断裂处.
>
> 数字以 Qwen3-0.6B (GQA) 和 DeepSeek-V2 (MLA) 为例. 代码块各自独立, 只依赖 Python 与 mlx, 可以保存成文件运行, 也可以整块贴进交互式解释器; 每块下方的 text 块是实际输出, 运行环境为 mlx 0.32.0、Python 3.10.18、arm64. 文献中的结论都对照原文核对过, 出处列在末尾.

## decode 每一步都要重读全部历史

注意力层为每个位置 t 算出三个向量: 查询 q_t、键 k_t 与值 v_t. 位置 t 的输出是历史上各位置的值的加权和, 权重由 q_t 与每个 k_j 的点积经过 softmax 得到:

```text
score_{t,j} = q_t · k_j                 j = 0, 1, …, t   (实现中还要除以一个与宽度有关的常数)
w_{t,j}     = softmax_j(score_{t,j})    对 j 归一化, 权重和为 1
out_t       = Σ_j w_{t,j} · v_j
```

decode 每次只新增一个位置. 因果掩码使位置 j 的计算只用到 j 及其之前的输入, 后来的 token 不会改变 k_j 与 v_j, 所以它们算出一次就可以存下来. 这份存储叫 KV cache, 每一层各有一份. 第 t 步只需为新位置算 q_t、k_t、v_t, 但打分和加权求和要用到全部 t+1 个 k 与 v, 于是每一步都要把整份 cache 从内存读进计算单元.

Qwen3-0.6B 每层为每个位置存 1024 个数的 k 和 1024 个数的 v (1024 这个宽度由哪些因子组成, 在「读取量由哪几个因子相乘」一节拆开), 共 28 层, 以 bf16 存放, 每个数 2 字节. 下面的脚本算出 cache 的大小, 并和每一步同样要读一遍的权重对比:

```python
kv_width = 1024                    # Qwen3-0.6B 每层每个位置的 k 宽度, v 同宽
layers = 28
nbytes = 2                         # bf16
weights = 596.0e6                  # 全部参数个数, 含与出口共用的嵌入表

per_layer = 2 * kv_width * nbytes  # k 与 v 各一份
per_token = per_layer * layers
print(f"每 token 每层   2 × {kv_width} × {nbytes} B = {per_layer} B = {per_layer // 1024} KiB")
print(f"每 token 28 层  {per_token} B = {per_token // 1024} KiB")
for S in (2048, 32768):
    print(f"S = {S:>5}       每步读 cache {per_token * S / 2**30:.2f} GiB")

print(f"每步读权重 bf16        {weights * 2 / 1e9:.2f} GB, 与 cache 相等时 S = {weights * 2 / per_token:,.0f}")
print(f"每步读权重 4 bit 打包  {weights * 0.5 / 1e9:.2f} GB, 与 cache 相等时 S = {weights * 0.5 / per_token:,.0f}")
```

```text
每 token 每层   2 × 1024 × 2 B = 4096 B = 4 KiB
每 token 28 层  114688 B = 112 KiB
S =  2048       每步读 cache 0.22 GiB
S = 32768       每步读 cache 3.50 GiB
每步读权重 bf16        1.19 GB, 与 cache 相等时 S = 10,393
每步读权重 4 bit 打包  0.30 GB, 与 cache 相等时 S = 2,598
```

每一步读的权重量是固定的, 读的 cache 量随历史长度 S 线性增长. 权重以 bf16 存放时, 两者在 S ≈ 10,393 处相等; 权重保持 4 bit 打包 (不计 scales 与 biases) 时, 交点提前到 S ≈ 2,598. 越过交点之后, 每一步的读取以 cache 为主.

```text
每步读的字节
  │                                   ╱ cache: 112 KiB × S
  │                                 ╱
  │ ──────────────────────────────╳──────────── 权重 1.19 GB (bf16)
  │                             ╱ S ≈ 10,393
  │                          ╱
  │                       ╱
  └──────────────────────────────────────────── S
```

多个请求同时 decode 时, 权重读一次就供全部请求使用, cache 则每个请求各有一份, 交点按请求数成比例提前. decode 通常受内存带宽限制, 在这个前提下, 长上下文与批量服务两种场景里, 缩小 cache 都直接减少每一步要读的字节, 也增加一台机器能同时容纳的上下文总长.

## 读取量由哪几个因子相乘

前面算出的每层 1024 个数由两个因子组成. Qwen3 的注意力把查询分成 H 个头, 每个头宽 D; 键与值分成 G 组, 每组宽 D; 1024 = G × D = 8 × 128. 四个记号各对应模型配置文件 config.json 里的一个字段:

```text
记号  config.json 字段        Qwen3-0.6B   含义
E     hidden_size             1024         层与层之间传递的向量宽度, 注意力的输入与输出都是这个宽度
H     num_attention_heads     16           查询头的个数
G     num_key_value_heads     8            键值组的个数, 每组一份 k 与一份 v
D     head_dim                128          每个头 (每组) 的宽度
```

一层注意力里各个量的形状由这四个数决定. 下图中 L 是这一步新送入的位置数, S 是连同历史在内的总位置数:

```text
x (L, E)
 ├─ W_q ─→ (L, H·D) = (L, 2048) ─ 切成 16 段, 每段 128 ──────────────┐
 ├─ W_k ─→ (L, G·D) = (L, 1024) ─ 切成 8 段 ─→ 追加进 cache ──┐      │  第 h 个查询头
 └─ W_v ─→ (L, G·D) = (L, 1024) ─ 切成 8 段 ─→ 追加进 cache ──┴──────┴→ 读第 h // (H/G) 组
                                                                         打分 (H, L, S) → 加权 (H, L, D)
                                                                         → 拼回 (L, H·D) ─ W_o ─→ (L, E)
```

D 必须直接从配置读取. 若按 E / H 推算, 得到 64, 只有实际值的一半, 后面所有关于 cache 的数字都会少算一半. Qwen3-0.6B 的 H·D = 2048, 比 E 宽一倍, 头宽是单独设定的超参数, 不由 E 与 H 决定; 查询与键的归一化层 (q_norm、k_norm) 和旋转位置编码的宽度都是 128, 可以用来核对.

每一步读取的 cache 字节数于是写成六个因子之积 (这里假定 v 与 k 同宽, Qwen3 正是如此; 「能改什么」一节说明这一点并非必需):

```text
每步读的 cache 字节 = 2 × G × D × b × N × S
                     │   └─┬─┘   │   │   └─ 历史长度: 滑动窗口、线性注意力改变的是它
                     │     │     │   └───── 层数: 由模型规模决定
                     │     │     └───────── 每个数的字节数: KV 量化改变的是它
                     │     └─────────────── 组数 G 与组宽 D: 本篇讨论的对象 (MLA 把 2·G·D 整体换成另一个宽度)
                     └───────────────────── k 与 v 各一份
```

b、N、S 三项各有专门的方法. 本篇只讨论 2·G·D 这一项, 也就是每层为每个位置存多少个数; 后文会看到, 这一项的各种改法最终都落在「cache 里存什么」上, 而另外两项也可以从同一个角度提问: 每个数存几位, 是在不改变存放物的前提下压缩它的表示; 历史长度 S 能否不让 cache 随它增长, 是把「与头数无关」换成「与长度无关」.

## 能改什么: 一条形状约束与效果的代价

要缩小 2·G·D, 先要分清哪些地方受硬约束, 哪些地方只是取舍.

形状上的硬约束来自打分. 打分是 q 与 k 的点积, 两个向量逐位相乘再求和, 所以每个查询头与它所读的那组键必须等宽. 值只参与加权求和, 权重是打分经过 softmax 得到的标量, 个数等于历史位置数 S, 与值的宽度无关. 下面的例子让 q 与 k 宽 8, v 宽 5, 再把 k 换成宽 6:

```python
import mlx.core as mx

mx.random.seed(0)
L, S = 3, 4                       # 3 个 query 位置, 4 个历史位置
q = mx.random.normal((L, 8))      # q 与 k 同宽 8
k = mx.random.normal((S, 8))
v = mx.random.normal((S, 5))      # v 宽 5, 与 q、k 不同

w = mx.softmax(q @ k.T / 8**0.5, axis=-1)
out = w @ v
print("权重", w.shape, "每行和", [round(x, 6) for x in w.sum(axis=-1).tolist()])
print("输出", out.shape)

k_bad = mx.random.normal((S, 6))  # k 宽 6, 与 q 不同
try:
    q @ k_bad.T
except ValueError as e:
    print("q 宽 8, k 宽 6:", e)
```

```text
权重 (3, 4) 每行和 [1.0, 1.0, 1.0]
输出 (3, 5)
q 宽 8, k 宽 6: [matmul] Last dimension of first input with shape (3,8) must match second to last dimension of second input with shape (6,4).
```

v 的宽度与 q、k 不同时, 注意力照常计算, 输出随之变为 5 维; k 与 q 不等宽时, 打分无法计算. 多数模型让 v 与 k 同宽, 这是实现上的方便, k 与 v 可以共用同一套切分和存放逻辑. 后文的 DeepSeek-V2 就是一个例外: 每个头的键宽 128 + 64 = 192, 值宽 128.

这条约束限定了缩小 G·D 的方式. 只缩小键一侧的 D 不可行, 要缩就得连同查询一起缩, 每个头用来打分的向量随之变短. 能在键值一侧单独调整的只剩组数 G.

效果上的代价来自头与头之间的分工. 设置多个查询头, 是为了让每个头用自己的 k 打分、读自己的 v, 在同一段历史里关注不同的位置、取回不同的内容. G 小于 H 时, 同一组的头共用一份 k 与 v, 彼此只能靠各自的 q 区分; G 越小, 共用一份的头越多.

四种形态面对的是同一个取舍: 在每头 q、k 等宽的约束下, 用多少存储换多少头的独立性. 显存与带宽的压力推动 G·D 变小, 效果要求每个头尽量保有自己的 k 与 v. 后面两种解法, 一种减少份数, 另一种改变存放的内容.

## 减少份数: MHA → GQA → MQA

组数 G 是键值一侧唯一能单独调整的因子. 从 G = H 一路减到 G = 1, 得到三种形态 (查询头数 H = 16, 每头宽 D = 128):


| 架构 | 查询头（Query 分组） | 共享映射 | 实际缓存的 Key/Value | 份数 | 单 token 单层显存 (bf16) |
| --- | --- | --- | --- | --- | --- |
| **MHA** | `q0`, `q1`, `q2`, `q3` ... `q15` | 1 对 1 | `k0v0`, `k1v1`, `k2v2`, `k3v3` ... `k15v15` | 16 份 | 8 KiB |
| **GQA** | `[q0, q1]`, `[q2, q3]` ... `[q14, q15]` | 2 对 1 | `k0v0`, `k1v1` ... `k7v7` | 8 份 | 4 KiB |
| **MQA** | `[q0, q1, q2, q3 ... q15]` | 16 对 1 | `k0v0` | 1 份 | 0.5 KiB |


MHA (multi-head attention, Vaswani 等 2017) 中每个查询头有自己的一份 k、v. MQA (multi-query attention) 让全部查询头共用一份. GQA (grouped-query attention) 介于两者之间, 把查询头分成 G 组, 组内共用一份. Qwen3-0.6B 取 G = 8, 每两个查询头共用一组.

GQA 的计算与 MHA 形式相同. 把每组 k、v 复制给它服务的每个头, 得到一份 MHA 形状的 k、v, 再按 MHA 计算, 结果与 GQA 逐元素一致:

```python
import mlx.core as mx

mx.random.seed(0)
H, G, D, S = 4, 2, 8, 5
q = mx.random.normal((H, D))              # 一个 query 位置, 4 个头
k = mx.random.normal((G, S, D))           # cache 里只有 2 组
v = mx.random.normal((G, S, D))

# GQA 的写法: 第 h 个头读第 h // (H/G) 组
out_gqa = mx.stack([
    mx.softmax(k[h // (H // G)] @ q[h] / D**0.5) @ v[h // (H // G)]
    for h in range(H)
])
# 把每组复制给它服务的头, 得到一份 MHA 形状的 k/v, 再按 MHA 计算
k_rep = mx.repeat(k, H // G, axis=0)      # (H, S, D)
v_rep = mx.repeat(v, H // G, axis=0)
out_mha = mx.stack([mx.softmax(k_rep[h] @ q[h] / D**0.5) @ v_rep[h] for h in range(H)])
print("GQA vs 复制后的 MHA  最大差", float(mx.abs(out_gqa - out_mha).max()))
print("cache 里的 k:", k.shape, " 复制后:", k_rep.shape)
```

```text
GQA vs 复制后的 MHA  最大差 0.0
cache 里的 k: (2, 5, 8)  复制后: (4, 5, 8)
```

GQA 因此可以看成一个受约束的 MHA: 同组各头的 k、v 被强制取相同的值. cache 只存未复制的 G 份, 复制在读取时完成; tiny-llm 的 scaled_dot_product_attention_grouped 用 reshape 加广播实现这一步, 复制品不写进内存.

约束越强, 效果损失越大, 不同文献报告的损失量级不一致. Shazeer 2019 提出 MQA, 报告解码明显加快, 质量只有轻微下降. Ainslie 等 2023 指出 MQA 会造成质量下降, 并提出 GQA: 用原预训练 5% 的计算量从 MHA checkpoint 继续训练, GQA 的质量接近 MHA, 速度接近 MQA. DeepSeek-V2 技术报告附录 D.1 在 7B dense 模型上做了对照, 三个模型都训练 1.33T token, 通过调整层数把参数量对齐到约 7B (MQA 7.1B, 另外两个 6.9B):

```text
基准 (指标)        MQA    GQA (8 组)   MHA
BBH (EM, 3-shot)   33.2   35.6         37.0
MMLU (5-shot)      37.9   41.2         45.2
C-Eval (5-shot)    30.0   37.7         42.9
CMMLU (5-shot)     34.6   38.4         43.5
```

在这组较难的基准上, 四项分数都按 MHA、GQA、MQA 的顺序递减. G 减到 1 之后, 这条路已经到底.

## 改变存放物: MLA

到了 MQA，份数已经砍到了极致（只有 1 份）。如果要继续缩小显存，沿着“砍份数”这条路已经无路可走。

如果还想继续压显存，同时又不想让所有头只能看到一模一样的键，就只能改变缓存里面存的内容。MLA（multi-head latent attention, DeepSeek-AI 2024）的核心变化只有一句话：显存里不存任何一个头的真实 Key 和 Value，只存一个公用的低维压缩向量。存压缩包，用时再解压。

要看清“存进显存的到底是什么形状”，只拿 **1 个刚进来的 token** 对比 MHA 和 MLA：

在 **MHA** 里（以 DeepSeek-V2 的尺寸为例），1 个 token 进入注意力层时，输入 $x$ 是一排 **5120 个数**，形状为 `(1, 5120)`。这一层有 **128 个注意力头**，每个头的 Key 宽 **128**。MHA 在写入缓存前就要为 128 个头分别算出各自的 Key：

```text
第 0 号头的 Key:   [ 128 个数 ]
第 1 号头的 Key:   [ 128 个数 ]
...
第 127 号头的 Key: [ 128 个数 ]
```

把这 128 个头的 Key 叠在一起存进显存，哪怕只有 1 个 token，它在显存里也是一张 `128 × 128` 的二维矩阵（128 行，每行 128 个数，共 $128 \times 128 = 16384$ 个数；Value 同样是一张 `128 × 128` 的矩阵）。

到了 **MLA**，模型把“分给 128 个头”这件事推迟到算注意力的时候再做，整个过程拆成两步：

1. **写进缓存时：还没分头，只存一排 512 个数**
   1 个 token 的输入 $x$（一排 5120 个数）先乘上一张所有头共用的下投影权重矩阵 $W_{\text{down}}$（形状 `5120 × 512`），直接压成**一排 512 个数**，记作 $c$：

   ```text
   1 个 token 的输入 x        下投影权重矩阵 W_down        存进显存的压缩向量 c
      (1 × 5120)          ×      (5120 × 512)        =       (1 × 512)
   [ 5120 个数的一排 ]                                    [ 512 个数的一排 ]
   ```

   这里需要区分“做压缩的权重”与“存进显存的结果”：
   * 做压缩的工具 $W_{\text{down}}$ 是一张 `5120 × 512` 的**权重矩阵**。
   * 真正写进 KV cache 的结果 $c$，对 1 个 token 来说**只是一排 512 个数（一个长度为 512 的向量）**，不再有“128 个头”那一维。等到积累了 $S$ 个历史 token，每个 token 各自存一排 512 个数，按时间顺序一行行摞起来，整份 cache 才是一张 `S × 512` 的表。

2. **打分读取时：每个头用自己的矩阵，把那 512 个数还原成 128 个数**
   当具体某个头（比如第 0 号头）要计算注意力时，显存里只有那一排公共的 512 个数 $c$。第 0 号头拿出一张属于它自己的键上投影矩阵 $U_0$（形状 `512 × 128`）去乘 $c$，算出属于第 0 号头的 128 维 Key（$k_0$）；第 1 号头拿出自己的矩阵 $U_1$ 去乘同一排 $c$，算出第 1 号头的 128 维 Key（$k_1$）：

   ```text
   显存里公共的 c             第 0 号头的解压矩阵 U_0       第 0 号头自己的 Key (k_0)
      (1 × 512)          ×         (512 × 128)         =         (1 × 128)

   显存里同一排 c             第 1 号头的解压矩阵 U_1       第 1 号头自己的 Key (k_1)
      (1 × 512)          ×         (512 × 128)         =         (1 × 128)
   ```

   Value 的一侧完全一样：每个头用另一张属于自己的值上投影矩阵 $V_h$（形状也是 `512 × 128`），同样去乘显存里那一排 $c$，还原出该头自己的 128 维 Value（$v_h$）。

因为每个头的解压矩阵 $U_h$、$V_h$ 里的权重参数各不相同，还原出来的 $k_h$、$v_h$ 也互不相同，各个头依然保有独立的注意力视野；而在显存里，系统从头到尾只保存了那一排 512 个数 $c$，不必为 128 个头分别存数据。

顺着上面这条数据流，后文用到的记号对应关系如下：

```text
数据流中的量 / 权重              记号       1 个 token 的形状 (DeepSeek-V2)   论文 / mlx_lm 源码中的名字
────────────────────────────────────────────────────────────────────────────────────────────────────────────
进入注意力层的输入向量           x          (1, E)   = (1, 5120)              h_t / x (hidden_size = 5120)
下投影权重矩阵                   W_down     (E, d_c) = (5120, 512)            W^{DKV} / kv_a_proj_with_mqa
存进显存的压缩向量               c          (1, d_c) = (1, 512)               c_t^{KV} / c
压缩向量 c 的宽度                d_c        标量 512 (代码变量记作 dc)        d_c / kv_lora_rank
第 h 个头的键解压矩阵            U_h        (d_c, D) = (512, 128)             W_h^{UK} / embed_q
第 h 个头的值解压矩阵            V_h        (d_c, D) = (512, 128)             W_h^{UV} / unembed_out
解压后第 h 个头的 Key 与 Value   k_h, v_h   (1, D)   = (1, 128)               k_{t,h}^C, v_{t,h}^C (每头宽 D = d_h = 128)
单独存放的位置键及其宽度         k_pe, d_r  (1, d_r) = (1, 64)                k_t^R, d_h^R / qk_rope_head_dim
```

把单个 token 的向量写成列向量形式，第 $h$ 个头的键与值就是两步矩阵相乘：

$$c = W_{\text{down}} x, \qquad k_h = U_h c = U_h (W_{\text{down}} x), \qquad v_h = V_h c = V_h (W_{\text{down}} x)$$

* 括号里的 $c = W_{\text{down}} x$ 是存进显存的那一排压缩数字，长度为 $d_c$。
* 外面的 $U_h$ 与 $V_h$ 是第 $h$ 个头各自的解压矩阵，把长 $d_c$ 的 $c$ 还原成长 $D$ 的 $k_h$ 与 $v_h$。

前三种机制也可以写进 $k_h = U_h (W_{\text{down}} x)$ 这个式子，区别在于它们的 $U_h$ 只做切片选取：

* **MHA**：显存里存所有 $H$ 个头拼起来的完整数组，宽度为 $H \cdot D$。第 $h$ 个头的 $U_h$ 从这个数组里切出第 $h$ 段数据，相当于乘单位矩阵的切片。
* **GQA 和 MQA**：显存里存 $G$ 份（宽度 $G \cdot D$）或 1 份（宽度 $D$），同组的多个头用各自的 $U_h$ 切出同一段数据。
* **MLA**：显存里存宽 $d_c = 512$ 的压缩向量 $c$。每个头的 $U_h$ 是稠密权重矩阵，通过矩阵乘法把 512 维的 $c$ 映射成该头自己的 128 维键。

为了对比四种做法在数学结构上的差别，可以把一个位置的全部 $H$ 个头的键竖着拼成一个长向量 $K_{\text{all}}$（总长度 $H \cdot D$）。既然每个头的 $k_h$ 都由输入 $x$（宽 $E$）经线性变换得到，拼出来的 $K_{\text{all}}$ 就等于一张形状为 $(H \cdot D, E)$ 的总投影矩阵 $W_{\text{stack}}$ 乘以 $x$。

这张总投影矩阵的**秩**，衡量的是算出来的 $H \cdot D$ 个数里**真正相互独立的数字有几个**：

* **MHA**：$H$ 个头的权重行彼此独立，$H \cdot D$ 个数全部独立，矩阵的秩等于 $H \cdot D$。
* **GQA**：只有 $G$ 组独立的权重行，其余的头把同组的行原样复制一遍，独立的数字只有 $G \cdot D$ 个，矩阵的秩等于 $G \cdot D$。
* **MQA**：只有 1 组权重行被复制了 $H$ 遍，独立的数字只有 $D$ 个，矩阵的秩等于 $D$。
* **MLA**：先把 $x$ 用 $(d_c, E)$ 的 $W_{\text{down}}$ 压成 $d_c$ 个公共数字，再用 $H$ 个头各自的 $U_h$（竖着拼起来是 $(H \cdot D, d_c)$）去组合这 $d_c$ 个数字。虽然每个头组合出来的键向量互不相同，但它们全部由中间那 $d_c$ 个数线性组合而成，背后独立的数字只有 $d_c$ 个，总矩阵 `U_all @ W_down` 的秩等于 $d_c$。

下面的脚本用玩具尺寸（输入宽 $E = 32$，头数 $H = 4$，每头宽 $D = 4$，GQA 组数 $G = 2$，MLA 压缩宽度 $d_c = 6$）构造这四种 $(16, 32)$ 的总投影矩阵，算出同一个输入 $x$ 下第 0 头与第 1 头的键之差，并用奇异值分解数出非零奇异值的个数：

```python
import mlx.core as mx

mx.random.seed(0)
E, H, D, G, dc = 32, 4, 4, 2, 6   # 输入宽 32, 4 个头, 每头宽 4, GQA 2 组, MLA 压到 6 维
x = mx.random.normal((E,))

def stacked_k_weight(kind):
    """返回 (H*D, E) 的总投影矩阵: 4 个头的 k 竖着拼成 16 维向量 = W_stack @ x."""
    if kind == "MHA":
        return mx.random.normal((H * D, E))                  # 4 个头各自独立, 共 16 行独立权重
    if kind == "GQA":
        w = mx.random.normal((G * D, E))                     # 只有 2 组 (8 行), 每组复制给 2 个头
        return mx.repeat(w.reshape(G, D, E), H // G, axis=0).reshape(H * D, E)
    if kind == "MQA":
        w = mx.random.normal((D, E))                         # 只有 1 组 (4 行), 复制给全部 4 个头
        return mx.repeat(w[None], H, axis=0).reshape(H * D, E)
    if kind == "MLA":
        W_down = mx.random.normal((dc, E))                   # (6, 32): 所有头共用的下投影
        U_all = mx.random.normal((H * D, dc))                # (16, 6): 4 个头各自的 (4, 6) 上投影竖着拼起来
        return U_all @ W_down                                # (16, 32): 两步相乘合成的总矩阵

for kind in ("MHA", "GQA", "MQA", "MLA"):
    W = stacked_k_weight(kind)
    k_heads = (W @ x).reshape(H, D)                          # 算出这一个位置的 4 个头: (4, 4)
    diff_01 = float(mx.abs(k_heads[0] - k_heads[1]).max())   # 看第 0 头和第 1 头的键是否相同
    s = mx.linalg.svd(W, compute_uv=False, stream=mx.cpu)
    rank = int((s > 1e-4 * s[0]).sum())                      # 独立自由度的个数 (矩阵的秩)
    print(f"{kind}  总投影 {str(W.shape):>8}  k_0 与 k_1 最大差 {diff_01:6.3f}  独立自由度 (秩) {rank:>2}")
```

```text
MHA  总投影 (16, 32)  k_0 与 k_1 最大差 10.760  独立自由度 (秩) 16
GQA  总投影 (16, 32)  k_0 与 k_1 最大差  0.000  独立自由度 (秩)  8
MQA  总投影 (16, 32)  k_0 与 k_1 最大差  0.000  独立自由度 (秩)  4
MLA  总投影 (16, 32)  k_0 与 k_1 最大差 20.082  独立自由度 (秩)  6
```

四种形态算出的秩（16、8、4、6）恰好等于各自在键这一侧需要存进 cache 的数字个数。GQA 和 MQA 减少独立自由度的方式是让同组的头取完全相同的键（`k_0 与 k_1 最大差 0.000`）；MLA 则让每个头保留互不相同的键（`k_0 与 k_1 最大差 20.082`），只要求这些键都由同一个 $d_c$ 维的压缩向量 $c$ 线性组合生成。更省的是，值的一侧也复用同一个 $c$，用另一组每头矩阵 $V_h$ 从 $c$ 还原出 $v_h$，不必为值再存一份压缩向量。

这个视角也引出一个问题：Qwen3-0.6B 的输入 $x$ 宽度 $E = 1024$，而 GQA 每个位置要存 8 组 $k$ 与 8 组 $v$（共 $2 \times 8 \times 128 = 2048$ 个数），存下来的键值比输入 $x$ 还宽一倍。为什么不直接缓存 1024 个数的输入 $x$？同样地，MLA 只缓存压缩向量 $c$（512 个数），每步计算注意力时，是否需要把全部历史位置的 $c$ 重新投影回 $k$ 与 $v$？

## 第二条约束: 从缓存点到打分只能经过线性映射

自回归 decode 阶段每步仅产生 1 个新 token，但注意力打分需要全部 $S$ 个历史位置的键与值。若采用「仅缓存低维向量，每步现算历史键值」的做法，以 Qwen3-0.6B 在 $S = 32768$ 为例：

```python
E, G, D, N, S = 1024, 8, 128, 28, 32768
nbytes = 2

cache_x = E * nbytes                      # 只存进入注意力的输入 x
cache_kv = 2 * G * D * nbytes             # 存 k 与 v
recompute = 2 * S * E * (2 * G * D)       # 每步把 S 个位置的 x 重新乘 W_k 与 W_v, 一次乘加记 2
print(f"缓存 x:    每 token 每层 {cache_x // 1024} KiB")
print(f"缓存 k/v:  每 token 每层 {cache_kv // 1024} KiB")
print(f"只缓存 x 时每步重算全部历史的 k/v, S = {S}:"
      f" 每层 {recompute / 1e9:.1f} GFLOP, 28 层 {recompute * N / 1e12:.2f} TFLOP")
```

```text
缓存 x:    每 token 每层 2 KiB
缓存 k/v:  每 token 每层 4 KiB
只缓存 x 时每步重算全部历史的 k/v, S = 32768: 每层 137.4 GFLOP, 28 层 3.85 TFLOP
```

每生成一个 token 需要额外执行 3.85 TFLOP 的矩阵乘法，节省显存带宽所换来的收益不足以弥补算力开销。

MLA 同样只缓存压缩向量 $c$ 而不存成品的每头键值，但它无需在每步解压历史序列，依赖的是矩阵乘法的结合律：将原本作用于历史序列的投影，转移到当前单位置的向量上。

### 打分阶段的运算重排

设定符号如下：当前新位置单个头的查询向量为 $q_h$（形状 $1 \times 128$）；cache 中保存的 $S$ 个历史位置压缩向量堆叠为矩阵 $C$（形状 $S \times 512$）；键解压矩阵为 $U_h$（形状 $512 \times 128$）；值解压矩阵为 $V_h$（形状 $512 \times 128$）。

若按先解压历史键、再打分的次序计算：

$$\text{scores}_h = q_h \cdot K_h^\top = q_h \cdot (C U_h)^\top = q_h \cdot U_h^\top \cdot C^\top$$

结合律允许改变括号位置，运算过程可写作两种数据流：

```text
路径 A：先解压历史键（右结合：q_h · (C · U_h)^T）
  输入：显存 cache C (S × 512)
  步骤 1：C (S × 512) ──乘 U_h (512 × 128)──> 物化中间张量 K_h (S × 128)
  步骤 2：q_h (1 × 128) ──与 K_h^T (128 × S) 做 GEMV──> scores (1 × S)
  代价：步骤 1 必须物化一个尺寸与序列长度 S 成正比的中间矩阵 (S × 128)，产生大量写回与重读。

路径 B：变换当前查询（左结合：(q_h · U_h^T) · C^T）
  输入：当前查询 q_h (1 × 128)
  步骤 1：q_h (1 × 128) ──乘 U_h^T (128 × 512)──> 变换后的查询 q̃_h (1 × 512)
  步骤 2：q̃_h (1 × 512) ──直接与显存 cache C^T (512 × S) 做 GEMV──> scores (1 × S)
  代价：步骤 1 仅对当前 1 个 token 变换，中间张量仅为 (1 × 512)，保留在寄存器中；
        中间结果尺寸完全与序列长度 S 脱钩，不需要物化任何历史长度的张量。
```

### 加权求和阶段的运算重排

Softmax 归一化后得到注意力权重 $w_h$（形状 $1 \times S$）。若按先解压历史值、再加权的次序：

$$\text{out}_h = w_h \cdot V_{\text{hist}, h} = w_h \cdot (C V_h) = (w_h C) \cdot V_h$$

同样利用结合律消除中间张量物化：

```text
路径 A：先解压历史值（右结合：w_h · (C · V_h)）
  输入：显存 cache C (S × 512)
  步骤 1：C (S × 512) ──乘 V_h (512 × 128)──> 物化中间张量 V_{hist, h} (S × 128)
  步骤 2：w_h (1 × S) ──加权求和 V_{hist, h} (S × 128)──> out_h (1 × 128)
  代价：必须物化尺寸为 (S × 128) 的中间张量。

路径 B：先加权聚合、再解压（左结合：(w_h · C) · V_h）
  输入：显存 cache C (S × 512)
  步骤 1：w_h (1 × S) ──在 512 维上加权求和 C (S × 512)──> 聚合向量 c_sum (1 × 512)
  步骤 2：c_sum (1 × 512) ──乘 V_h (512 × 128)──> out_h (1 × 128)
  代价：步骤 1 将整个历史序列直接折叠为单个 1 × 512 向量，V_h 的解压矩阵乘法仅执行一次。
```

DeepSeek-V2 技术报告将上述结合律变换称为**矩阵吸收**（absorption）：键解压矩阵 $U_h$ 吸收入查询投影，值解压矩阵 $V_h$ 吸收入输出投影。在 decode 阶段，显存中的历史缓存 $C$ 全程无需物化为各头的具体键值，消除随序列长度 $S$ 扩张的解压开销。

### 为什么普通 Transformer 无法重排乘法顺序

把 $q (U^\top C^\top)$ 换成 $(q U^\top) C^\top$，在矩阵乘法上只是利用了结合律，先算当前 token 的向量乘法，省去对历史矩阵 $C$ 的变换。这种重排之所以需要专门修改模型结构，原因在于普通 Transformer 无法直接做这个变换：

1. **计算量与内存读取量的取舍**：
   重排之后，不再需要生成 $(S \times 128)$ 的历史中间矩阵，但每一步注意力打分和加权求和的向量长度从 $D = 128$ 变成了 $d_c = 512$（每一步的乘加次数增加到 4 倍）。在 decode 阶段，速度受限于显存带宽，增加运算量来减少显存读写是合算的；但在 prefill 阶段，速度受限于芯片算力，增加运算量会导致速度变慢。因此推理引擎在 prefill 阶段仍需按原顺序计算。

2. **RoPE 旋转矩阵随位置变化，无法交换乘法顺序**：
   重排乘法顺序的前提，是历史向量 $c_j$ 到最终打分之间只有固定的线性变换。标准 Transformer 使用了旋转位置编码（RoPE, Su 等 2021）。

RoPE 会让第 $j$ 个历史位置的键乘以对应的旋转矩阵 $R(j)$：

$$k_{j,h} = R(j) \cdot U_h c_j$$

当前查询 $q_h$ 与历史键 $k_{j,h}$ 的内积为：

$$\text{score}_{t,j,h} = (R(t) q_h)^\top (R(j) U_h c_j) = q_h^\top R(t - j) U_h c_j$$

旋转矩阵 $R(t - j)$ 的数值依赖具体的位置编号 $j$，每个历史位置都不同。同时矩阵乘法不满足交换律：

$$R(t - j) U_h \neq U_h R(t - j)$$

$U_h$ 无法越过 $R(t - j)$ 提前与 $q_h$ 相乘。只要对压缩后的键直接施加 RoPE，就无法重排计算顺序，每一步 decode 依然要为全部 $S$ 个历史位置分别计算旋转后的键。

### 解耦 RoPE

DeepSeek-V2 为了能够重排乘法顺序，将键拆分为两路：

* **内容键**：由 $c_j$ 解压得到（维度 $D = 128$），**不做 RoPE 旋转**。因为不包含随位置变化的矩阵，可以调整乘法顺序，先将 $U_h^\top$ 与查询相乘。
* **位置键**：从输入 $x_j$ 另外投影出一个向量 $k_{\text{pe}, j}$（维度 $d_r = 64$），所有头共用这一份向量，旋转后写入 cache；查询侧各头分别投影一个位置查询 $q_{\text{pe}, h}$（维度 64）并旋转。
* **打分求和**：最终注意力分数为两路内积相加：

$$\text{score}_{t, j, h} = \frac{(q_h U_h^\top) \cdot c_j^\top + q_{\text{pe}, h} \cdot k_{\text{pe}, j}^\top}{\sqrt{D + d_r}}$$

每层每个位置存两项：内容压缩向量 $c$（宽度 $d_c = 512$）和共用的位置键 $k_{\text{pe}}$（宽度 $d_r = 64$），总共 $512 + 64 = 576$ 个数。共用位置键的形式与 MQA 相同。

下面的脚本（$S = 6, H = 4, D = 8, d_c = 12, d_r = 4$）验证上述结合律变换及其在 RoPE 下的表现：

```python
import mlx.core as mx

mx.random.seed(0)
E, H, D, dc, S = 32, 4, 8, 12, 6

x = mx.random.normal((S, E))
W_q = mx.random.normal((H, E, D))
W_down = mx.random.normal((E, dc))
U = mx.random.normal((H, dc, D))
V = mx.random.normal((H, dc, D))

c = x @ W_down
q = x[-1] @ W_q

def rope(t, pos, base=10000.0):
    d = t.shape[-1]
    ang = pos[:, None] * base ** (-mx.arange(0, d, 2) / d)[None, :]
    a, b = t[..., 0::2], t[..., 1::2]
    cos, sin = mx.cos(ang), mx.sin(ang)
    return mx.stack([a * cos - b * sin, a * sin + b * cos], axis=-1).reshape(t.shape)

pos = mx.arange(S).astype(mx.float32)
diff = lambda a, b: float(mx.abs(a - b).max())

# 一、无 RoPE: 路径 A (物化历史) vs 路径 B (矩阵吸收)
k = c @ U
v = c @ V
w_ex = mx.softmax((q @ k.swapaxes(-1, -2)) / D**0.5, axis=-1)
out_explicit = (w_ex @ v)[:, 0, :]

q_abs = q @ U.swapaxes(-1, -2)
w_ab = mx.softmax((q_abs @ c.T) / D**0.5, axis=-1)
out_abs = ((w_ab @ c) @ V)[:, 0, :]
print("一、无 RoPE: 最大绝对误差", diff(out_explicit, out_abs))

# 二、键施加 RoPE: R(j) 破坏结合律
k_rot = rope(k, pos)
q_rot = rope(q, pos[-1:])
w_rot = mx.softmax((q_rot @ k_rot.swapaxes(-1, -2)) / D**0.5, axis=-1)
out_rot = (w_rot @ v)[:, 0, :]

q_try = q_rot @ U.swapaxes(-1, -2)
w_try = mx.softmax((q_try @ c.T) / D**0.5, axis=-1)
out_try = ((w_try @ c) @ V)[:, 0, :]
print("二、施加 RoPE 后强行结合: 最大绝对误差", diff(out_rot, out_try))

# 三、解耦 RoPE: 内容分支吸收, 位置分支独立点积
dr = 4
W_qr = mx.random.normal((H, E, dr))
W_kr = mx.random.normal((E, dr))
k_pe = rope(x @ W_kr, pos)
q_pe = rope(x[-1] @ W_qr, pos[-1:])
pe_scores = q_pe @ k_pe.T

scale = (D + dr) ** -0.5
w_dec_ex = mx.softmax((q @ k.swapaxes(-1, -2) + pe_scores) * scale, axis=-1)
out_dec_ex = (w_dec_ex @ v)[:, 0, :]

w_dec_ab = mx.softmax((q_abs @ c.T + pe_scores) * scale, axis=-1)
out_dec_ab = ((w_dec_ab @ c) @ V)[:, 0, :]
print("三、解耦 RoPE: 最大绝对误差", diff(out_dec_ex, out_dec_ab))
```

```text
一、无 RoPE: 最大绝对误差 0.0
二、施加 RoPE 后强行结合: 最大绝对误差 33.53173828125
三、解耦 RoPE: 最大绝对误差 0.0
```

测试结果表明：解耦 RoPE 使得内容分支的矩阵吸收在数学上严格精确，而直接对解压后的键施加 RoPE 会导致结合律完全失效。

同一机理亦解释了 Qwen3-0.6B 为何无法通过仅缓存输入 $x$ 节省存储：其注意力层在 $W_k$ 投影后包含 `k_norm`（RMSNorm，属于非线性缩放）与 RoPE 两处变换。非线性与依赖位置的算子共同封死了运算重排的可能，导致历史位置的重新计算在数学上无法转移到单 token 查询一侧。

有了 $d_c + d_r = 512 + 64 = 576$ 这个完整结构，现在可以把 MHA、GQA、MQA、MLA 四种形态的真实存储量与 K/V 侧参数量放在同一张表里对比。DeepSeek-V2 共 60 层，$E = 5120, H = 128, D = 128, d_c = 512, d_r = 64$；下面同时列出 Qwen3-0.6B 的尺寸（其中 MLA 一行是按同样 $E, H, D$ 假想的，Qwen3 实际采用 $G = 8$ 的 GQA）：

```python
def table(name, E, H, D, N, G=None, dc=None, dr=None):
    print(f"{name}: E={E} H={H} D={D} 层数={N}")
    rows = [("MHA", 2 * H * D, 2 * E * H * D)]
    if G:
        rows.append((f"GQA G={G}", 2 * G * D, 2 * E * G * D))
    rows.append(("MQA", 2 * D, 2 * E * D))
    if dc:
        # MLA 每层每位置存 c (宽 dc) 与 k_pe (宽 dr)
        # K/V 侧参数 = 下投影 E*(dc + dr) + H 个头的键/值上投影 2*H*D*dc
        rows.append(("MLA", dc + dr, E * (dc + dr) + 2 * H * D * dc))
    for kind, cache, params in rows:
        print(f"  {kind:9} 每 token 每层 {cache:>6} 个数 = {cache * 2 / 1024:>5.2f} KiB"
              f"   K/V 侧参数 {params / 1e6:>7.2f}M   每 token 全部层 {cache * N:>9,} 个数")


table("Qwen3-0.6B, MLA 一行按同样的 E/H/D 假想", 1024, 16, 128, 28, G=8, dc=512, dr=64)
table("DeepSeek-V2", 5120, 128, 128, 60, dc=512, dr=64)
print("MLA 折合的 GQA 组数:", (512 + 64) / (2 * 128))
```

```text
Qwen3-0.6B, MLA 一行按同样的 E/H/D 假想: E=1024 H=16 D=128 层数=28
  MHA       每 token 每层   4096 个数 =  8.00 KiB   K/V 侧参数    4.19M   每 token 全部层   114,688 个数
  GQA G=8   每 token 每层   2048 个数 =  4.00 KiB   K/V 侧参数    2.10M   每 token 全部层    57,344 个数
  MQA       每 token 每层    256 个数 =  0.50 KiB   K/V 侧参数    0.26M   每 token 全部层     7,168 个数
  MLA       每 token 每层    576 个数 =  1.12 KiB   K/V 侧参数    2.69M   每 token 全部层    16,128 个数
DeepSeek-V2: E=5120 H=128 D=128 层数=60
  MHA       每 token 每层  32768 个数 = 64.00 KiB   K/V 侧参数  167.77M   每 token 全部层 1,966,080 个数
  MQA       每 token 每层    256 个数 =  0.50 KiB   K/V 侧参数    1.31M   每 token 全部层    15,360 个数
  MLA       每 token 每层    576 个数 =  1.12 KiB   K/V 侧参数   19.73M   每 token 全部层    34,560 个数
MLA 折合的 GQA 组数: 2.25
```

表中「K/V 侧参数」一列，对 MHA、GQA、MQA 指 $W_k$ 与 $W_v$ 两张矩阵的参数量；对 MLA 指下投影（含 $W_{\text{down}}$ 与位置键投影，共 $E \cdot (d_c + d_r)$）加上 $H$ 个头的键、值上投影（$U_h$ 与 $V_h$，共 $2 \cdot H \cdot D \cdot d_c$），不计查询一侧。在存储量上，每位置 576 个数折合 $576 / (2 \times 128) = 2.25$ 组 GQA（与 DeepSeek-V2 报告表 1 的换算一致）。在参数量上，增减随模型规模而变：按 Qwen3-0.6B 的小尺寸（$H = 16$），MLA 的 K/V 侧参数（2.69M）比 8 组 GQA（2.10M）略多；按 DeepSeek-V2 的大尺寸（$H = 128, E = 5120$），MHA 的 $W_k$、$W_v$ 各是 $(16384, 5120)$ 的大矩阵，MLA 的 K/V 侧参数只有 MHA 的约 12%（19.73M 对 167.77M）。

效果方面，DeepSeek-V2 报告附录 D.2 在两种规模的 MoE 模型上对比了 MLA 与 MHA，同规模的两个模型只有注意力机制不同（小模型约 16B 总参数、训练 1.33T token；大模型约 250B 总参数、训练 420B token）：

```text
                          小 MoE              大 MoE
                          MHA       MLA       MHA       MLA
激活参数                  2.5B      2.4B      25.0B     21.5B
每 token KV cache (个数)  110.6K    15.6K     860.2K    34.6K
BBH (EM, 3-shot)          37.9      39.0      46.6      50.7
MMLU (5-shot)             48.7      50.0      57.5      59.0
C-Eval (5-shot)           51.6      50.9      57.9      59.2
CMMLU (5-shot)            52.3      53.4      60.7      62.5
```

KV cache 分别缩小到 MHA 的 14% 与 4%，八项评测分数中有七项更高，仅小模型的 C-Eval 略低。这组数据来自提出 MLA 的同一团队在两个规模上的单次对照，未报告方差，足以说明在这组配置下大幅压缩 cache 没有牺牲模型质量。

$d_c$ 在 MLA 里扮演的角色与 $G$ 在 GQA 里相同：它决定了 cache 的宽度，也决定了全部头的键与值能张成多大的共享子空间。DeepSeek-V2 取 $d_c = 4 D = 512$、$d_r = D / 2 = 64$。

## 吸收的代价与两条执行路径

吸收省掉了逐个历史位置的还原，但打分与加权的宽度从每头 $D = 128$ 变成了压缩向量宽度 $d_c = 512$。显式还原与矩阵吸收两种做法的运算量随本轮新位置数 $L$ 的变化方式完全不同：

* **显式还原**：先把全部 $S$ 个历史位置的 $c$（宽 $d_c = 512$）还原成每个头的 $k_h$、$v_h$（宽 $D = 128$），这笔开销与历史长度 $S$ 成正比，但与新位置数 $L$ 无关；随后每一对（新位置，历史位置）都在较窄的 $D = 128$ 维上打分和加权。
* **矩阵吸收**：先对 $L$ 个新位置的查询乘 $U_h^\top$、最后对 $L$ 个新位置的输出乘 $V_h$，这笔开销只与 $L$ 成正比；但随后每一对（新位置，历史位置）都必须在较宽的 $d_c = 512$ 维上打分和加权。

下面按 DeepSeek-V2 的单头尺寸（$D = 128, d_c = 512, d_r = 64$），在历史长度 $S = 32768$ 时比较单个头在这两条路径下的浮点运算次数（一次乘加计 2 FLOP）：

```python
dc, D, dr = 512, 128, 64          # DeepSeek-V2: 压缩宽度 d_c, 每头宽度 D, 位置键宽度 d_r
S = 32768

expand = 2 * 2 * dc * D           # 一个历史位置: 由 c 还原出这个头的 k 与 v (两次矩阵向量乘)
absorb = 2 * 2 * D * dc           # 一个新位置: q 乘 U_h^T, 输出乘 V_h (两次矩阵向量乘)
core_ex = 2 * (D + dr) + 2 * D    # 显式还原后每对 (新位置, 历史位置): 打分宽 128+64, 加权宽 128
core_ab = 2 * (dc + dr) + 2 * dc  # 矩阵吸收后每对 (新位置, 历史位置): 打分宽 512+64, 加权宽 512


def explicit(L):
    return S * expand + L * S * core_ex


def absorbed(L):
    return L * absorb + L * S * core_ab


print(f"每对位置的打分与加权: 还原 {core_ex}, 吸收 {core_ab}, 比 {core_ab / core_ex:.1f}")
print(f"{'L':>6} {'还原':>12} {'吸收':>12} {'吸收/还原':>9}")
for L in (1, 5, 64, 170, 2048, 32768):
    print(f"{L:>6} {explicit(L):>12.3e} {absorbed(L):>12.3e} {absorbed(L) / explicit(L):>9.3f}")

L_star = S * expand / (absorb + S * (core_ab - core_ex))
print(f"S = {S} 时, L < {L_star:.0f} 吸收的运算量更小")
```

```text
每对位置的打分与加权: 还原 640, 吸收 2176, 比 3.4
     L           还原           吸收     吸收/还原
     1    8.611e+09    7.157e+07     0.008
     5    8.695e+09    3.578e+08     0.041
    64    9.932e+09    4.580e+09     0.461
   170    1.216e+10    1.217e+10     1.001
  2048    5.154e+10    1.466e+11     2.844
 32768    6.958e+11    2.345e+12     3.370
S = 32768 时, L < 170 吸收的运算量更小
```

decode 阶段每次只送入 $L = 1$ 个新位置，矩阵吸收的运算量还不到显式还原的 1%（$0.008$ 倍）。而当一次送入的新位置数 $L$ 超过约 170 之后，显式还原的运算量反而更小；prefill 阶段一次送入成千上万个位置，应当走显式还原路径（若把还原出的 $k_h$、$v_h$ 持久存进 cache，显存占用就会退回 MHA 的大小，因此显式还原只用于算完即扔的 prefill）。

除了总运算量，decode 阶段的速度还取决于**算术强度**，即每读 1 字节 cache 做多少次浮点运算。因为 decode 受内存带宽限制，把同一份缓存数据读进芯片后复用的次数越多、算术强度越高，越能把空闲的算力利用起来：

```python
D, dc, dr, nbytes = 128, 512, 64, 2


def mha_family(H, G):
    flops = H * (2 * D + 2 * D)   # 每个头: 打分 D 次乘加, 加权 D 次乘加
    read = G * 2 * D * nbytes     # 每组读一份 k 与一份 v
    return flops / read


def mla_absorbed(H):
    flops = H * (2 * (dc + dr) + 2 * dc)
    read = (dc + dr) * nbytes     # 全部头共用一份 c 与一份 k_pe
    return flops / read


print("每读 1 字节 cache 做的浮点运算数 (decode, 一个历史位置, 一层)")
print(f"  H=16   MHA {mha_family(16, 16):6.1f}  GQA G=8 {mha_family(16, 8):6.1f}"
      f"  MQA {mha_family(16, 1):6.1f}  MLA {mla_absorbed(16):6.1f}")
print(f"  H=128  MHA {mha_family(128, 128):6.1f}  MLA {mla_absorbed(128):6.1f}")
```

```text
每读 1 字节 cache 做的浮点运算数 (decode, 一个历史位置, 一层)
  H=16   MHA    1.0  GQA G=8    2.0  MQA   16.0  MLA   30.2
  H=128  MHA    1.0  MLA  241.8
```

MHA 每读出一份 $k$、$v$ 只供 1 个头使用，每读 1 字节只做 1 次浮点运算，计算单元绝大部分时间在等内存。GQA 把一份 $k$、$v$ 复用给 $H / G$ 个头，算术强度提高到 $H / G$ 倍。MLA 把同一份 $(c, k_{\text{pe}})$ 复用给全部 $H$ 个头，而且每个头在 512 维上打分与加权，算术强度在 $H = 128$ 时达到 241.8 FLOP/Byte。只要这个数值还没有超过硬件的峰值算力与内存带宽之比，多出来的浮点运算都由原本空转等待内存的算力完成，每一步的耗时依然由读取字节数决定。

`mlx_lm` 0.31.3 的 DeepSeek-V3 实现（`mlx_lm/models/deepseek_v3.py`）正是按上面的分界拆成两条执行路径。代码中的字段与前文记号一一对应：输入 $x$ 先投影并切成 $c$（宽 $d_c = 512$）与 $k_{\text{pe}}$（宽 $d_r = 64$），$c$ 再过一次 RMSNorm（`kv_a_layernorm`），然后只把 $c$ 与旋转后的 $k_{\text{pe}}$ 写进 cache——因为 `kv_a_layernorm` 发生在写进 cache 之前，从缓存里的 $c$ 到打分之间依然只有线性映射，不影响后续的矩阵吸收。计算注意力时，先算出位置部分的打分 `pe_scores`，再以 `L == 1` 为界分岔：

```text
decode (L = 1)
  q_nope (每头 D=128) ── embed_q (U_h^T) ──> 每头 d_c=512 ──┐
  c (d_c=512, 从 cache 读) ─────────────────────────────────┼─> 在 512 维上打分与加权 ──> unembed_out (V_h) ──> 每头 D=128 ──> o_proj
  pe_scores (q_pe · k_pe, 宽 d_r=64 的位置打分) ─────────────┘
  每个头的 k_h、v_h 始终没有出现

prefill (L > 1)
  c (d_c=512) ── embed_q (U_h) ──────> 每头 k_h (D=128) ──┐
  c (d_c=512) ── unembed_out (V_h) ──> 每头 v_h (D=128) ──┼─> 在 128 维上打分与加权 ──> o_proj
  pe_scores ──────────────────────────────────────────────┘
```

同一个包里的 DeepSeek-V2 旧实现（`mlx_lm/models/deepseek_v2.py`）则是反面例子：它在写入 cache 之前，先把 $c$ 显式还原成全部 128 个头的 $k_h$、$v_h$，再把位置键 $k_{\text{pe}}$ 复制 128 份拼在每个头的 $k_h$ 后面，最后把拼好的大张量写进 cache。结果每个位置每层存了 $128 \times (128 + 64 + 128) = 40,960$ 个数，比标准 MHA 的 $32,768$ 个数还多 25%。MLA 架构提供了每个位置每层只存 576 个数的数学结构，真正省下显存必须靠实现层缓存 $c$ 与 $k_{\text{pe}}$ 并在 decode 时执行矩阵吸收——同一模型架构的两种代码实现，cache 占用相差约 71 倍（$40960 / 576 \approx 71.1$）。

## 等价处与断裂处

把四种形态放在一起对比（每位置每层存的个数以 $H = 16, D = 128, G = 8$ 为例，MLA 取 DeepSeek-V2 的 $d_c = 512, d_r = 64$）：

```text
        cache 每位置每层       各头之间共享什么                         需要吸收   RoPE 的位置
MHA     2·H·D = 4096           不共享                                   否         k 旋转后存入
GQA     2·G·D = 2048           同组的头共享同一份 k、v                  否         k 旋转后存入
MQA     2·D   = 256            全部头共享同一份 k、v                    否         k 旋转后存入
MLA     d_c + d_r = 576        全部头的 k、v 共享同一个 d_c 维子空间    是         内容部分不旋转, 位置部分 k_pe 另存一份
```

等价处有三处：

* **GQA 与复制后的 MHA**：数学计算相同；实现上 GQA 在 cache 里只存 $G$ 份，读取时靠形状重排与广播让同组的头复用同一份数据，不在显存中物化复制品。
* **缓存宽度与投影矩阵的秩**：四种形态在键一侧需要缓存的宽度，都等于全部头堆叠后的总投影矩阵 $W_{\text{stack}}$ 的秩。MHA、GQA、MQA 通过让多个头的权重行相同来降低秩，MLA 通过中间宽度为 $d_c$ 的公共下投影 $W_{\text{down}}$ 来限制秩。
* **MLA 的矩阵吸收与显式还原**：两者算出的注意力输出在数学上恒等（仅差浮点舍入误差），但浮点运算量随新位置数 $L$ 朝相反方向变化——decode（$L = 1$）时吸收省算力，prefill（$L$ 大）时显式还原省算力，工程实现因此按 $L$ 分成两条路径。

断裂处有两处，都出现在从 MHA 一族过渡到 MLA 的地方：

* **计量单位的变化**：MHA、GQA、MQA 的 cache 由若干份真实的 $k$、$v$ 组成，份数 $G$ 从 $H$ 降到 1，「存了几份」是衡量它的自然单位；MLA 的 cache 里存的 $c$ 不是任何一个头的键或值，只能用子空间宽度 $d_c$ 加上位置键宽度 $d_r$ 衡量。把 576 个数折合成 $2.25$ 组 GQA 只是存储字节数上的比照，MLA 中并不存在 $2.25$ 份键值。
* **复用机制引入的计算图约束**：GQA 复用 cache 靠的是强迫同组多个头的键值相等，这是加在**模型参数**上的约束；MLA 复用 cache 靠的是把解压矩阵 $U_h$、$V_h$ 吸收到查询侧和输出侧，这要求**从缓存点到打分之间只能经过不随历史位置变化的线性映射**，这是加在**计算图拓扑**上的新约束。MHA 一族缓存的是已经过完归一化与旋转、直接参与打分的 $k$ 本身，缓存点与打分之间没有运算，因此这条约束在 MHA 一族里从未显现；一旦跨入 MLA（或者试图在 Qwen3 里只缓存输入 $x$），这条约束立刻生效，迫使 MLA 把带 RoPE 的位置键 $k_{\text{pe}}$ 单独拆出来存放，也堵死了 Qwen3 靠缓存输入 $x$ 省显存的路径。

把视角拉远，本篇讨论的四种形态都在回答「cache 的大小能否不随查询头数 $H$ 增长」，MLA 用共享低维子空间给出了兼顾容量与多头独立性的答案。回到开篇的读取量公式 `2 × G × D × b × N × S`，另外两个因子也在回答平行的问题：一个是历史长度 $S$——滑动窗口让每层只保留最近若干个位置，线性注意力层把逐位置追加的 cache 换成固定大小的状态矩阵，两者都让 cache 不随 $S$ 无限增长（其中线性注意力在思路上与 MLA 平行，都是改变存放物，只是压缩的维度从头数 $H$ 换成了序列长度 $S$）；另一个是每个数的字节数 $b$——KV 量化不改变存放物，只减少每个数的比特数，但它与 MLA 结合时会产生新的问题：MLA 缓存的是压缩向量 $c$，量化 $c$ 产生的舍入误差会经过每个头的解压矩阵 $U_h$、$V_h$ 放大或扩散进打分与输出，这与在 MHA/GQA 中直接量化已经成型的 $k$、$v$ 具有不同的误差传播性质。

## 回到 tiny-llm

tiny-llm 实现的 Qwen3-0.6B 属于 GQA（$H = 16, G = 8, D = 128$）。`src/tiny_llm/attention.py` 中的 `scaled_dot_product_attention_grouped` 直接从 `query` 的最后一维读取头宽 $D$，并按同一个 $D$ 重排 `key` 与 `value`，因此代码实现上要求 $q, k, v$ 三者等宽（由「能改什么」一节可知，数学上 $v$ 可以与 $q, k$ 不同宽，如 DeepSeek-V2 在拼接位置键后每头 $q, k$ 宽 192 而 $v$ 宽 128）。同一个函数把查询重排成 `(G, H/G, L, D)`、把键值重排成 `(G, 1, S, D)`，利用广播实现「第 $h$ 个查询头自动读取第 `h // (H/G)` 组键值」，显存中不生成复制后的张量。Week 2 的 KV cache 每层存的是经过 `k_norm` 与 `RoPE` 之后的键和值，每个位置写入 $G \cdot D = 1024$ 个数的 $k$ 与同样多的 $v$；「第二条约束」一节解释了为什么缓存点只能选在 `k_norm` 与 `RoPE` 之后。

## 边界

本篇只讨论每层为每个位置存多少个数。读取量公式里的另外三项各有专门的方法：每个数的字节数 $b$ 对应 KV 量化；历史长度 $S$ 对应滑动窗口（`mlx_lm` 的 `gemma3_text`、`gpt_oss` 实现含 `sliding_window` 配置）与线性注意力层（`mlx_lm` 的 `qwen3_next` 实现中 `GatedDeltaNet` 层用固定大小的状态代替逐位置的 cache）；层数 $N$ 由模型规模决定。它们可以与本篇的四种形态叠加使用。

DeepSeek-V2 中 $d_c = 4 D = 512$ 与 $d_r = D / 2 = 64$ 两个超参数取值直接沿用原报告，本篇没有展开它们的消融实验依据。两张模型效果对照表直接引自原报告附录，没有自行复训。运算量与算术强度两个脚本只计入注意力核心的打分与加权，不含投影矩阵、softmax 与其他层，硬件上的端到端耗时需另行实测。

## 参考文献

- Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin, 2017. Attention Is All You Need. NeurIPS. arXiv 1706.03762.
- Shazeer, 2019. Fast Transformer Decoding: One Write-Head is All You Need. arXiv 1911.02150.
- Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai, 2023. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. EMNLP. arXiv 2305.13245.
- DeepSeek-AI, 2024. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv 2405.04434. 本篇引用其 2.1 节 (低秩键值联合压缩、解耦 RoPE、表 1)、3.1.2 节 (超参数) 与附录 D (表 8、表 9).
- Su, Lu, Pan, Murtadha, Wen, Liu, 2021. RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv 2104.09864.
- mlx_lm 0.31.3 源码: `mlx_lm/models/deepseek_v3.py` (`DeepseekV3Attention`), `mlx_lm/models/deepseek_v2.py` (`DeepseekV2Attention`).
