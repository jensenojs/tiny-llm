# mlx_lm 的 qwen3.py: 模型定义源码导读

Week 1 Day 5 的 Task 3 要求把 `mlx_lm` 加载的真模型拆开，用自己写的零件重装。动手前值得把那份模型定义的源码通读一遍：`.venv/lib/python3.10/site-packages/mlx_lm/models/qwen3.py`，223 行，六个类。这份文件是你手写零件的工业版参考答案：`scaled_dot_product_attention_grouped`、`RMSNorm`、`RoPE`、`Qwen3MLP`、`Qwen3TransformerBlock`、`Qwen3MultiHeadAttention`、`Embedding.as_linear`，每个都能在这里找到对应物，逐类对照能看出手写实现和工业实现差在哪。

本文与 docs/model-tree-notes.md 分工：那份讲加载后的对象树长什么样、权重怎么搬运（运行态）；这份讲源码本身怎么读、每个类在定义什么（定义态）。文中 `qwen3.py:42` 指该文件第 42 行。装的 mlx_lm 版本不同行号会漂移，对不上时用这条命令定位实际文件：

```bash
python -c "import mlx_lm.models.qwen3 as m; print(m.__file__)"
```

## 文件地图

读陌生模型文件的第一步看 imports（qwen3.py:3-12），它告诉你哪些逻辑定义在本文件、哪些是公共件。qwen3.py 自己只定义六个类，公共逻辑全部从兄弟模块导入：`swiglu` 来自 `activations.py`，`BaseModelArgs`、`create_attention_mask`、`scaled_dot_product_attention` 来自 `base.py`，`initialize_rope` 来自 `rope_utils.py`。读完 qwen3.py 之后按需查那三个文件。

| qwen3.py | 定义的类 | 职责 | 你的对应物 |
|---|---|---|---|
| :15-29 | `ModelArgs` | 配置 dataclass，字段对齐 config.json | （无，你的 `__init__` 直接收散参数） |
| :32-89 | `Attention` | 单层注意力 | `Qwen3MultiHeadAttention`（src/tiny_llm/qwen3_week1.py:21） |
| :92-100 | `MLP` | 前馈网络 | `Qwen3MLP`（qwen3_week1.py:133） |
| :103-126 | `TransformerBlock` | 一层的组装 | `Qwen3TransformerBlock`（qwen3_week1.py:175） |
| :129-160 | `Qwen3Model` | 嵌入 + N 层 + final norm | `Qwen3ModelWeek1`（qwen3_week1.py:260） |
| :163-223 | `Model` | 外壳：出口投影、加载清理、多卡切片 | （出口投影部分，见「Model」节） |

文件顺序就是依赖顺序：配置在最上，零件居中，组装其次，外壳收尾，从上往下读即可。

还有一个打印树与源码对不上的点先说清：源码里写的是 `nn.Linear` / `nn.Embedding`，你 print 出来的树上却是 `QuantizedLinear` / `QuantizedEmbedding`。加载 4bit 模型时 load 流程会把普通模块换成量化版，这个差异的来源和搬运方式 docs/model-tree-notes.md 已讲过，本文不重复。

## ModelArgs: 配置与权重分两路

`ModelArgs`（qwen3.py:15-29）是个 `@dataclass`，十二个必填字段加一个可选的 `rope_scaling`，全部是标量：hidden_size、层数、头数、head_dim、eps、theta、tie 开关。这份字段清单就是 HF `config.json` 的子集，名字一一对应。

构造用的数字和推理用的张量从两条路进来：`load` 读 config.json 得到配置字典，读 safetensors 得到权重字典，互不掺和。配置字典到 `ModelArgs` 的翻译由 `BaseModelArgs.from_dict` 完成（base.py:14-21）：

```python
@classmethod
def from_dict(cls, params):
    return cls(
        **{
            k: v
            for k, v in params.items()
            if k in inspect.signature(cls).parameters
        }
    )
```

机制一条：用 `inspect.signature` 拿到 dataclass 声明过的字段名，config 里多出来的键（`attention_bias`、`bos_token_id` 之类）直接丢弃。模型代码因此只跟自己认得的字段走，HF config 加字段不用改模型代码。

对照你的版本：`Qwen3MultiHeadAttention.__init__` 收 13 个散参数（qwen3_week1.py:22-37），Task 3 装配时由你从 `mlx_model.args` 逐个取出来填。`from_dict` 把「从字典到构造参数」这一步做成了所有模型共用的机制。

`Attention.__init__` 里有一句 `assert args.num_key_value_heads is not None`（qwen3.py:38）。`num_key_value_heads` 在 dataclass 里没有默认值，正常构造必有值，这个 assert 防的是 config.json 里显式写 null 的写法：没有 KV 头数，k_proj 就不知道该多宽。

## Attention: 构造期做什么，前向做什么

### 构造期

`__init__`（qwen3.py:33-57）做三件事，全部与输入无关：

1. 算好 scale 存成属性：`self.scale = head_dim**-0.5`（qwen3.py:42）。1/sqrt(D) 只依赖 head_dim，构造时算一次，前向直接用。
2. 建四个投影（qwen3.py:44-47），`bias=False`。q_proj 的输出宽是 `n_heads * head_dim`，k/v_proj 是 `n_kv_heads * head_dim`，两个数在源码层面的差就是 GQA 省参数的位置（0.6B 是 2048 对 1024，见 docs/model-tree-notes.md 的数字核对表）。
3. 建 q_norm/k_norm 和 RoPE（qwen3.py:49-57）。

`q_norm`、`k_norm` 是 `nn.RMSNorm(head_dim, eps)`：γ 的长度是 head_dim，作用域在头的内部。`self.rope = initialize_rope(...)` 在构造期调用一次，返回一个 RoPE 模块存起来。构造期做与输入无关的工作是这个文件反复出现的模式：RoPE 的频率表只跟 head_dim 和 theta 有关，算一次即可。你的版本同理，`RoPE(dims=head_dim, seq_len=max_seq_len, base=theta)`（qwen3_week1.py:51-53）在构造期预存了整张 cos/sin 表，只是存得更满。

`initialize_rope` 内部按 `scaling_config` 分派（rope_utils.py:235-251）：Qwen3 的 `rope_scaling` 是 None，走 default 分支返回普通 `nn.RoPE`；config 里带 llama3/yarn 缩放的模型会换成长度外推变体。进阶内容，知道有分派即可。

### 前向

`__call__`（qwen3.py:59-89）逐行对照你的实现：

| qwen3.py | 代码（摘） | 在做什么 | 你的对应 |
|---|---|---|---|
| :65 | `B, L, D = x.shape` | 解包形状 | qwen3_week1.py:68 |
| :67 | `q/k/v = q_proj/k_proj/v_proj(x)` | 投影，H*D 个分量拼在一起 | :77-79 |
| :69-74 | `q_norm(reshape(B,L,n_heads,-1)).transpose(0,2,1,3)` | 拆头 → 每头归一化 → 换轴 | :78, :87-88, :106-107 |
| :75 | `values.reshape(...).transpose(0,2,1,3)` | V 只换轴，不归一化不旋转 | :79, :108 |
| :77-83 | cache 分支（见下） | RoPE 带偏移，K/V 拼历史 | （无，Week 2） |
| :85-87 | `scaled_dot_product_attention(...)` | 库函数打分加权 | :115-120 |
| :88-89 | `transpose(...).reshape(B,L,-1)`、`o_proj` | 换回、合头、压回 E | :126, :130 |

三处展开。

一处记号陷阱先排掉：qwen3.py:65 的 `D` 是 x 的最后一维，即 hidden_size；你记号表里的 D 是 head_dim（qwen3_week1.py:17）。同一个字母在两份代码里指不同的数，读的时候以左右文为准。

归一化的时机。`reshape(B, L, n_heads, -1)` 把 H_q*D 拆成两个轴之后，`nn.RMSNorm` 沿最后一维归一化，最后一维恰好是 head_dim：每个 (batch, 位置, 头) 上的 D 维向量各自算均方根，全头共用一组 γ。q_norm 与 k_norm 做同样的事，V 过 norm 的只有 reshape 和 transpose。这与你注释里的判断一致：Q/K 要加工进「内容+位置」，V 是载荷（qwen3_week1.py:66）。

cache 分支（qwen3.py:77-83）。有 cache 时，RoPE 多传一个 `offset=cache.offset`：这批 token 在完整序列里从第几个位置开始，旋转角就从那里取。随后 `cache.update_and_fetch(keys, values)` 把本次的 K/V 拼进缓存的历史，返回拼好的完整 K/V。无 cache 就直接旋转。这就是 KV cache 的接口形态：cache 持有历史 K/V 和当前长度，decode 时 Q 的 L 是 1、K/V 的 S 是历史加一。docs/gqa-notes.md 推导过的「decode 三处差异」在这个分支里全部落了地。Week 2 的主题，这里认出接口即可。

精度处理。你的版本进注意力前 `astype(mx.float32)`、出来再转回（qwen3_week1.py:115-120），防 bfloat16 点积误差累积；工业版直接把张量交给 `scaled_dot_product_attention`，精度策略在 `mx.fast.scaled_dot_product_attention` 的 C++/Metal 内部处理。数学相同，处理位置不同。

`self.scale` 显式传给了库函数（qwen3.py:86）而你不需要传：你的 `scaled_dot_product_attention_grouped` 内部默认 scale = 1/sqrt(D)（docs/gqa-notes.md），工业版把它提到构造期算好。库函数还有一个带 `cache.bits` 判断的量化分支（base.py:117-128），KV cache 可以量化存储，进阶带过。

顺带一处类型注解的松动：`__call__` 的 mask 参数注解是 `Optional[mx.array]`（qwen3.py:62），但 Qwen3Model 传进来的可能是字符串 `"causal"`（下一节）。注解没跟上实际类型，读源码时以实际数据流为准，库函数对字符串和数组都能处理。

## MLP: 一行 SwiGLU

MLP 全部代码（qwen3.py:92-100）：

```python
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))
```

`swiglu` 的定义在 activations.py:9-11，也是一行：

```python
@partial(mx.compile, shapeless=True)
def swiglu(gate, x):
    return nn.silu(gate) * x
```

silu 作用于 gate 支路，乘上线性的 up 支路，与你在主笔记里推过的门控数学一致，也和你的 `gate = silu(linear(x, w_gate)); linear(gate * up, w_down)`（qwen3_week1.py:170-172）逐算子对应。工程差异在装饰器 `@partial(mx.compile, shapeless=True)`：`mx.compile` 把这几个逐元素算子打包成一个编译单元，GPU 上少几次中间结果的读写往返；`shapeless=True` 表示编译结果不锁定输入形状，变长序列不必重编译。性能优化，知道这一行的用途即可。

## TransformerBlock: 四行 Pre-LN

`__init__`（qwen3.py:103-114）建四个子模块：`self_attn`、`mlp`、两个门口 RMSNorm。qwen3.py:106-107 还往 self 上存了 `num_attention_heads` 和 `hidden_size` 两个整数，前向用不到，读到时不必找它们的用途。

前向四行（qwen3.py:122-126）：

```python
r = self.self_attn(self.input_layernorm(x), mask, cache)
h = x + r
r = self.mlp(self.post_attention_layernorm(h))
out = h + r
```

Pre-LN 结构：LN 在子层入口的分支上，残差加法在主干，x 沿 h、out 一路直通。与你的六行展开（qwen3_week1.py:250-257）同构，你给每个中间量起了名字，工业版复用变量 r。mask 和 cache 从 Block 一路透传进 Attention，整个前向的参数表就是 `(x, mask, cache)`，Week 2 加 cache 时改的也是这三个口子。

## Qwen3Model: 主干返回隐藏态

`__init__`（qwen3.py:130-140）三样：`embed_tokens = nn.Embedding(vocab_size, hidden_size)`、`layers = [TransformerBlock(args) for _ in range(N)]`、final `norm = nn.RMSNorm(hidden_size)`。层是普通 Python 列表。

`__call__`（qwen3.py:142-160）五行主干：

```python
if input_embeddings is not None:
    h = input_embeddings
else:
    h = self.embed_tokens(inputs)

if cache is None:
    cache = [None] * len(self.layers)
mask = create_attention_mask(h, cache[0])

for layer, c in zip(self.layers, cache):
    h = layer(h, mask, c)

return self.norm(h)
```

四个点。

mask 整段造一次（qwen3.py:155），交给 `create_attention_mask`（base.py:45-55）。它有四种返回：cache 定制的掩码；`None`（输入只有 1 个 token 时，decode 场景没有未来可遮，docs/gqa-notes.md 推导过这个退化）；显式数组（`return_array` 或滑窗时走 `create_causal_mask`）；字符串 `"causal"`（无 cache 的默认路径）。你手写的 `causal_mask`（src/tiny_llm/attention.py）造的是显式数组，因果语义相同，载体不同：字符串由库函数在底层解析，省一次掩码张量的构造。

cache 为 None 时造 `[None] * N`（qwen3.py:153-154）：cache 列表与层列表一一对齐，`zip(layers, cache)` 给每层发各层的缓存。训练时不传 cache，推理时从外面传进真实的列表。

`input_embeddings` 参数（qwen3.py:146-151）允许调用方跳过查表直接喂嵌入向量，主流程用不到，见到调用时知道这个口子即可。

返回值是 `self.norm(h)`：norm 之后的隐藏态，形状 (B, L, E)，无 logits。出口投影在外壳 Model 里。这个拆分直接关系你 Task 3 的结构选择：Qwen3ModelWeek1 的出口投影放不放、放哪，工业版的答案见下一节。

## Model: 外壳、tie 分支与加载清理

`__init__`（qwen3.py:163-170）建 Qwen3Model，然后看 tie 开关：`tie_word_embeddings` 为 False 才建 `lm_head = nn.Linear(hidden_size, vocab_size, bias=False)`。tie 为 True 时 lm_head 这个模块根本不存在。

`__call__`（qwen3.py:172-183）两步：主干前向拿隐藏态，然后按 tie 分支出 logits：

```python
if self.args.tie_word_embeddings:
    out = self.model.embed_tokens.as_linear(out)
else:
    out = self.lm_head(out)
```

tie 能成立的结构原因在权重形状里：`nn.Embedding(vocab, E)` 和 `nn.Linear(E, vocab)` 的权重都是 `(vocab_size, E)`（qwen3.py:136, 170），同一个张量查表用、转置乘也用，一份参数当两份使。0.6B 是 tie=True，对象树上没有 lm_head，你在 Task 3 里 `hasattr` 检查的就是这个分支。你写过自己的 `as_linear`（src/tiny_llm/embedding.py），这里调的是 `nn.Embedding` 自带的同名方法，语义相同：隐藏态乘嵌入矩阵的转置出 logits。

`sanitize`（qwen3.py:185-188）在加载时清洗权重字典：tie 模型的一些 checkpoint 里仍存着 `lm_head.weight`，而模型上没有这个模块，权重名对不上号会报错，`weights.pop("lm_head.weight", None)` 把它丢弃。加载期一次性的名字清理，与数学无关。

`shard`（qwen3.py:190-219）是多卡切片：把每层的 q/k/v/gate/up 换成 `shard_linear` 分片版（"all-to-sharded"），o_proj 和 down_proj 换成 "sharded-to-all"，头数和 KV 头数整除卡数。单卡推理不会走到，进阶带过。

最后 `@property layers`（qwen3.py:221-223）把 `self.model.layers` 转发到外壳上，调用方写 `model.layers[i]` 就不用 `model.model.layers[i]`。你 print 树用的 `mlx_model.model` 是内层，`mlx_model.model.layers` 与 `mlx_model.layers` 等价。

## 权重从哪进来: load 链路

qwen3.py 全文没有一处给 `q_proj.weight` 赋权重数据——构造函数里的 `nn.Linear`/`nn.Embedding` 是空壳（只有形状），数据从另一个入口进来：

1. `mlx_lm.load()` 读 safetensors，得到 (名字, 数组) 对，名字是模块树路径（`model.layers.0.self_attn.q_proj.weight`）。
2. `model.load_weights(...)`（base.py:123）按点分路径逐级下钻，把数组塞进对应模块的 dict 存储。名字与树严格对号，这也是 sanitize 要先清掉多余 `lm_head.weight` 的原因。
3. 量化模型加载即 freeze（base.py:511，参数进 `_no_grad` 名单），全部无梯度——推理模型的常态。

这解释了 Task 3 的一个现象：`mlx_model` 的模块打印出来形状数据俱全，但那是**加载填进来的**；你的类走另一条路——构造函数直传权重。所以 mlx_lm 的 `__init__` 里没有搬运，你的 `__init__` 里全是搬运，两种写法对应两种供给方式。三件套的内部结构见 docs/model-tree-notes.md 的 QuantizedLinear 一节。

## 与手写版的差异总表

| 维度 | 你的手写版 | 工业版 |
|---|---|---|
| 权重持有 | 裸 `mx.array` 由构造函数传入 | `nn.Linear`/`nn.RMSNorm`/`nn.Embedding` 模块自持，框架按名字加载 |
| scale | SDPA 内部默认 1/sqrt(D) | 构造期 `head_dim**-0.5` 算好，显式传参 |
| RoPE | 构造期预存整张 cos/sin 表 | 模块存频率，`mx.fast.rope` 现算旋转，支持 offset |
| mask | `causal_mask` 造显式数组 | `create_attention_mask` 可返回字符串 `"causal"` 或 None |
| cache | 无参数（Week 1 只做 prefill） | `update_and_fetch` 就地拼历史，RoPE 带 offset |
| 精度 | 进 SDPA 前 `astype(mx.float32)` | 底层融合实现内部处理 |
| 出口 | logits 在模型 `__call__` 里 | 内层返回隐藏态，外壳按 tie 分支出 logits |
| 算子 | 纯 Python 逐算子 | 热路径走 `mx.compile`（swiglu）与 `mx.fast.*` |

Week 1 需要消除的只有三行：权重持有（dequantize 后直取，docs/model-tree-notes.md 的三类搬运）、mask（用你自己的 `causal_mask`）、出口（tie 判断）。cache 一行是 Week 2 的入口，精度与算子两行可以原样保留差异。

## 一代到三代: qwen.py 对照

mlx_lm 里同时躺着 `qwen.py`（Qwen 一代）和 `qwen3.py`。把两个 `Attention` 并排放，几乎每一处差异都能归因到少数几个根因——这比单独读 qwen3.py 的巩固效率高，因为每个"变了"都逼你答"为什么变"。

代真实数字：Qwen-7B 一边是 E=4096, H=32, D=128；Qwen3-8B 一边是 dim=4096, n_heads=32, n_kv_heads=8, head_dim=128。

### 原文

两段都贴全，复习不用再翻源文件。上面 qwen.py（`mlx_lm/models/qwen.py:29-66`），下面 qwen3.py（`mlx_lm/models/qwen3.py:32-89`）：

```python
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_size = args.hidden_size
        self.num_attention_heads = args.num_attention_heads

        hidden_size_per_attention_head = hidden_size // self.num_attention_heads

        self.rotary_emb = nn.RoPE(hidden_size_per_attention_head, traditional=False)

        proj_size = args.kv_channels * self.num_attention_heads

        self.c_attn = nn.Linear(hidden_size, proj_size * 3, bias=True)
        self.c_proj = nn.Linear(hidden_size, proj_size, bias=not args.no_bias)

        self.scale = hidden_size_per_attention_head**-0.5

    def __call__(self, x, mask=None, cache=None):
        qkv = self.c_attn(x)

        q, k, v = mx.split(qkv, 3, axis=-1)

        B, L, _ = q.shape

        queries = q.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
        keys = k.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
        values = v.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rotary_emb(queries, offset=cache.offset)
            keys = self.rotary_emb(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rotary_emb(queries)
            keys = self.rotary_emb(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.c_proj(output)
```

```python
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)
```

### 根因: KV 头数

qwen.py 的 `ModelArgs` 里 `num_key_value_heads = None`，`__post_init__` 默默把它设成等于 Q 头数——32 = 32，标准 MHA，每个 Q 头有自己的 K/V 头，没有拼车。Qwen3 明着声明 `num_key_value_heads=8`，`n_repeats = 32/8 = 4`，每 4 个 Q 头共享一个 K/V 头（gqa-notes.md 的"拼车"一节，n_repeats 就是这个 4）。

一个参数的变化向下推倒了后面所有形状：

```python
# qwen3.py
self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)      # 4096 → 4096
self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)   # 4096 → 1024
self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)   # 4096 → 1024
```

K/V 投影的输出宽度只有 Q 的 1/4。这直接杀死了 Qwen 一代的熔合投影：

```python
# qwen.py
self.c_attn = nn.Linear(hidden_size, proj_size * 3, bias=True)    # 4096 → 12288
q, k, v = mx.split(qkv, 3, axis=-1)
```

熔合是 GPT-2 血统的工程取舍：三个等宽投影打包成一个大矩阵，一次 GEMM 出全部结果，访存效率高，代价是三份必须等宽、`split(qkv, 3)` 才能均匀切。K/V 变窄之后三份不再等宽，熔合的前提没了——这就是 Qwen3 拆回三个独立投影的全部原因，同一个数学，两种打包方式。

### 换轴与合头: 两代相同

```python
# qwen.py
queries = q.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
...
output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
```

`(B,L,E) → (B,L,H,D) → (B,H,L,D)` 进注意力，算完逆向合头 `(B,L,E)`——和 gqa-notes.md 的流程表逐行对应，两代代码一字不差（Qwen3 的 GQA 差异发生在 SDPA 内部：K/V 头先 repeat 到 H 个，qwen3.py 走 `mx.fast` 融合实现，手写版对应你的 `scaled_dot_product_attention_grouped`）。reshape 的 `-1` 就是 D，每个头拿 E 里连续的一段，这是头分割的行业约定。

### 命名: c_proj → o_proj

Qwen 一代把输出投影叫 `c_proj`（convective proj，GPT-2 时代残留），Qwen3 叫 `o_proj`（output）。同一个东西：合头之后 `(B,L,E)` 回投到 E，流程表最后一步。

### 独立的改动

这些和 GQA 无关，是 Qwen3 自己的演进：

**按头 RMSNorm（q_norm/k_norm，Qwen3 新增）**

```python
self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)   # 对 128 维，不是对 4096
```

作用在投影之后、RoPE 之前，每个头独立对它的 D 维做归一化（在换轴后的 `(B,H,L,D)` 上沿最后一维）。Qwen 一代没有。这是 Qwen3 训练稳定性的改进，也是 Week 1 Task 3 装配清单里的额外模块。

为什么归 Q/K 不归 V，把注意力的标量链条摆出来就清楚了。打分公式是 `softmax(q·k / sqrt(D))`，进 softmax 之前的 logit 大小由 |q||k| 共同决定：Q/K 向量的模长不受约束（线性投影的输出，上一层残差流多大它们就多大），模长一大，softmax 的 logit 差距拉大，分布向 one-hot 退化，梯度跟着消失。V 没有这个问题——它只做加权平均的载荷，进的是 softmax 之后的凸组合系数，模长大只意味着输出大，下一层的 RMSNorm（每个 block 入口的 input_layernorm）会把它拉回来。同样过 normalize 的待遇差异，根源在 Q/K 站在 softmax 的上游、V 站在下游。

归一化的位置也有讲究。q_norm 在 RoPE 之前：去掉 γ 看纯 RMS 缩放，缩放和旋转可交换，先后无所谓；但 RMSNorm 带逐维 γ，γR ≠ Rγ（主笔记 RMSNorm 顺序段推过），加上 γ 之后先后是真实的选择，训练时定死、加载时必须一致。γ 的形状是 (head_dim,)，全部头共用一组，按 (batch, 头, 位置) 各自算均方根。

这个做法在 Qwen 之前已有先例：Gemma 2、Chameleon 的 QK-norm 是同一族手段，动机都是压住 attention logit 的尺度。Qwen3 把它带进了主力开源线，你的 Task 3 里它对应 `q_norm`/`k_norm` 两个模块、装配时 `q = self.q_norm(q)` 一行调用。

**bias 全砍**

qwen 的 `c_attn bias=True`；Qwen3 四个投影全部 `bias=False`。现代 LLM 通行做法：bias 对质量没贡献还占参数。

**head_dim 从算出来的变成显式配置**

qwen 的 D = `hidden // heads`，纯推导；Qwen3 的 `head_dim` 是配置直接给的。小尺寸 Qwen3（0.6B）里两者可以不相等：hidden=1024、16 头、head_dim=128，`q_proj` 输出 2048 ≠ 1024。所以 `o_proj` 必须写成 `Linear(n_heads * head_dim, dim)`——两个数不再必然相同，输出投影成了"形状恢复器"。

**RoPE 升级**

```python
# qwen:  nn.RoPE(128, traditional=False)
# qwen3: initialize_rope(head_dim, base=args.rope_theta, scaling_config=..., max_position_embeddings=...)
```

Qwen 一代的 RoPE 用默认 base，Qwen3 显式给 `rope_theta=1000000`，再叠一层 scaling 分派。Day 4 写过 RoPE 本体，这三样都能对上号：

**base（theta）**。Day 4 实现里 base 直接进频率表：第 i 对分量的旋转频率 θ_i = base^(-2i/D)。base 越大，所有频率越小，旋转越慢，相邻位置 q·k 的夹角差越小——远距离位置的区分度衰减得慢。Qwen 从 10000（GPT-2 时代惯例）提到 1000000，这是长上下文的第一层手段：不改任何代码结构，训练时就把位置编码的波长拉长。代价是短距离的分辨率略降（旋转太慢，近处位置挤在一起），所以 base 不是越大越好，是模型尺寸和上下文长度折算出来的数。

**traditional=False**。旋转的配对方式：True 是相邻两维配对（(x0,x1) 一对、(x2,x3) 一对，GPT-J 风格），False 是前后半段配对（x_i 配 x_{i+D/2}，GPT-NeoX/HF 风格）。数学上两种等价（重排一下维度即可），权重排列不同，跨框架加载时对不上号会静默出错。两代都取 False，你的 RoPE 也实现了 traditional 开关，同一个分叉。

**scaling_config**。长上下文的第二层：训练定死之后，推理时改频率映射，让 8k 训练的模型用到 32k 以上。llama3 类型按波段分段重映射频率，yarn 做插值加外推的混合。Qwen3 的 config 里 `rope_scaling` 是 None，走 default 分支返回普通 `nn.RoPE`——分派代码在 rope_utils.py:235-251，配置了缩放的模型（llama 系长上下文版本）才会走别的分支。Qwen3-8B 你加载的这份没有启用，知道有这一层即可。

### 对照总表

| 维度 | qwen.py（一代） | qwen3.py（三代） |
|---|---|---|
| KV 头数 | = Q 头数（MHA，无拼车） | 显式配置（8 vs 32，n_repeats=4） |
| QKV 投影 | 熔合 c_attn(→3E) + split | 三个独立投影，K/V 宽度 E/n_repeats |
| 输出投影 | c_proj | o_proj（同一个东西） |
| 按头归一化 | 无 | q_norm/k_norm（RMSNorm over head_dim） |
| bias | c_attn 有 | 全无 |
| head_dim | hidden // heads 推导 | 显式配置，可与 hidden/H 不等 |
| RoPE | nn.RoPE 裸函数 | initialize_rope 带 base/scaling |
| 位置信息的唯一入口 | RoPE（wte 只查词，无 wpe） | 同 |

### TransformerBlock 与 MLP: 命名换血统

两个 block 的对比结论：结构零差异，差异全在命名和血统。原文对照（qwen.py:76-126 在上，qwen3.py:92-126 在下）：

```python
class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(
            args.hidden_size, args.intermediate_size // 2, bias=not args.no_bias
        )
        self.w2 = nn.Linear(
            args.hidden_size, args.intermediate_size // 2, bias=not args.no_bias
        )
        self.c_proj = nn.Linear(
            args.intermediate_size // 2, args.hidden_size, bias=not args.no_bias
        )

    def __call__(self, x):
        a1 = self.w1(x)
        a2 = self.w2(x)
        return self.c_proj(swiglu(a2, a1))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.ln_1 = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.attn = Attention(args)
        self.ln_2 = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.mlp = MLP(args)

    def __call__(self, x, mask=None, cache=None):
        residual = x
        x = self.ln_1(x)
        x = self.attn(x, mask=mask, cache=cache)
        residual = x + residual
        x = self.ln_2(residual)
        x = self.mlp(x)
        x = x + residual

        return x
```

```python
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(self, x: mx.array, mask: mx.array, cache: Optional[Any]) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out
```

前向逐行同构，一模一样的 Pre-LN：norm 在子层入口的分支上，残差直通主干，注意力出口先并入残差、MLP 再来一轮。一代用 `residual` 一个变量反复覆盖，三代用 `r`/`h` 两个名字，数学逐行相同。

"三代更眼熟"的来源是命名血统：

| 角色 | qwen.py（GPT-2 血统） | qwen3.py（HF Llama 血统） |
|---|---|---|
| 注意力入口 norm | `ln_1` | `input_layernorm` |
| 注意力模块 | `attn` | `self_attn` |
| MLP 入口 norm | `ln_2` | `post_attention_layernorm` |
| MLP 门/上/下 | `w1`/`w2`/`c_proj` | `gate_proj`/`up_proj`/`down_proj` |
| 注意力输出投影 | `c_proj` | `o_proj` |

一代的 `ln_1/ln_2/c_proj` 是 GPT-2 的遗产，三代彻底换到 HF Llama 的命名系——`qwen3_week1.py` 用的就是这套名字。差异有实际后果：checkpoint 里的权重键名跟着模块名走，`layers.0.ln_1.weight` 和 `layers.0.input_layernorm.weight` 是两套格式，跨代加载必须改名，sanitize 那类加载清理处理的就是这种不齐。

MLP 里藏着一处一代的怪东西：`w1`/`w2` 的输出宽是 `intermediate_size // 2`。一代 config 里 `intermediate_size=11008`，实际每个分支只有 5504——这个数在一代的口径里是 gate+up 合起来的宽度（推断口径：与它 c_attn 熔合 QKV 的风格一致，喜欢把配对矩阵存成一个）。另外 `swiglu(a2, a1)` 的参数序把 silu 打在 w2 那条支路上，三代叫 gate_proj，同一份 SwiGLU 数学。复刻三代命名时对照一代，能确认自己没有把"名字"误当成"结构"。

### 供给方式: 18 个参数对一个 args

手写版 `Qwen3TransformerBlock.__init__`（qwen3_week1.py:177-221）收 18 个参数，三代只收一个 `args`，看着手写版复杂得多。拆开数：18 = 6 个标量 + 11 个权重数组 + 1 个默认参数。框架那边的"简洁"是假象——6 个标量躺在 `ModelArgs` 里，11 个权重框架根本不在构造期给：`nn.Linear`/`nn.RMSNorm` 建空壳，权重等 `load_weights` 按名字填。手写版没有框架替填，权重要在构造期一个一个递进来，这就是「权重从哪进来」一节说的两种供给方式：框架把参数表藏进 config 和加载链路，手写版没有框架，藏不了，只能摊开。签名本身也轮不到手写者选——Week 1 的 oracle 按这个参数表构造模块再灌权重对答案，签名是任务的验收合同。总账算全的话：框架文件里还有 sanitize、shard（多卡切片）、tie 分支、RoPE 分派，手写版一样都没有。单看构造函数是手写版复杂，看整个文件是框架复杂；算法复杂度两边相同，手写版多出来的是参数转手的体力，框架多出来的是看不见的基础设施。

### 对 Week 1 的意义

Task 3 要装配的 `Qwen3MultiHeadAttention` 就是 qwen3.py 这份 `__init__` + `__call__` 的手写复刻。Qwen 一代的价值在对照：熔合为什么必须拆、K/V 为什么变窄、o_proj 为什么是 `n_heads*head_dim → dim`——每个字段背后有一个"为什么"，装配时就不是抄形状而是讲道理。RoPE 本体和 cache 接口（offset、update_and_fetch）Day 4 已经写过，qwen.py 前向里那几行现在可以直接对上号：`offset=cache.offset` 是 decode 的绝对位置来源，`update_and_fetch` 是 S 的生长方式。

## 对 Task 3 的直接用处

装配顺序可以照这份源码的字段顺序对：`ModelArgs` 的字段清单就是你 `__init__` 要取的数；`Qwen3Model.__init__` 的三样对应你的 embed / layers / norm；`Model` 的 tie 分支对应你的出口投影。装配中遇到的每个字段名都能在文件地图里找到行号，拿不准某个字段是什么时，回到源码看它的构造参数比猜形状可靠。
