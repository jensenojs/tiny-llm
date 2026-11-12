# 走读: profile_week2_kernels.py

做完 kv-cache 之后要回答的问题是「这一步 decode 的时间花在哪类算子上」。这个脚本用重放回答：把一次 forward 拆成四组，每组按真实形状和真实调用次数单独跑一遍，各自计时，最后算占比。

入口在 `benches/profile_week2_kernels.py`，命令是：

```bash
pdm run profile-week2-kernels --solution tiny_llm --case capacity-cache:decode:128
```

`--case` 的格式是 `checkpoint:phase:tokens`。

## 命令到 main

`main()` 在 502 行。它先解析参数：

- `--solution` 二选一，`tiny_llm` 或 `tiny_llm_ref`
- `--case` 可以重复，不给就用 `DEFAULT_CASES`（26 行）
- `--warmup` 默认 4，`--iterations` 默认 12

`parse_case`（97 行）把字符串拆成 `ProfileCase(checkpoint, phase, tokens)`。

## 加载实现

`load_implementation`（73 行）动态 import 四个模块，把函数抓进 `KernelImplementation` 这个 dataclass（56 行）：

```python
attention = importlib.import_module(f"{name}.attention")
basics    = importlib.import_module(f"{name}.basics")
model     = importlib.import_module(f"{name}.qwen3_week2")
quantize  = importlib.import_module(f"{name}.quantize")
kernels   = importlib.import_module(f"{name}.week2_kernels")
```

抓到的是 `grouped_attention`、`linear`、`quantized_linear`、`decode_attention`、`swiglu` 这些函数本身，后续所有调用都经过这个结构。换 `--solution` 就是换这一层。

## 为这个 case 造模型

`profile_case`（463 行）：

```python
model = implementation.model_type(mlx_model, checkpoint=case.checkpoint)
replay = KernelReplay(implementation, model, case.phase, case.tokens)
```

`model_type` 是 `Qwen3ModelWeek2`，`checkpoint="capacity-cache"` 决定哪些 feature 打开。

## KernelReplay 造输入

`KernelReplay.__init__`（291 行）先定行数与上下文长度：

```python
self.rows = 1 if phase == "decode" else tokens
self.context = tokens
```

decode 时 `rows=1`，`context=128` 表示缓存里已有 128 个位置。prefill 时 `rows=context=128`。

然后从第一层读形状（`hidden_size`、`num_heads`、`num_kv_heads`、`head_dim`、`intermediate_size`、`dtype`），按这些数造随机张量：

```python
self.hidden  = (1, rows, hidden_size)           # 1024
self.query   = (1, num_heads, rows, head_dim)   # 16 头, 128 宽
self.key     = (1, num_kv_heads, context, head_dim)  # 8 组, 128 位置
self.value   = 同 key
self.gate    = (1, rows, intermediate_size)     # 3072
self.up      = 同 gate
self.tokens  = (1, rows) int32
```

最后 `evaluate([...])` 把这些张量全部算出来。正式计时时拿来即用，不含分配开销。

## 四个 builder

`profile_case` 里把四个方法排成 builder：

```python
builders = (
    ("projections", replay.projections),
    ("attention", replay.attention),
    ("normalization, position, and activation", replay.pointwise),
)
if case.phase == "decode":
    builders += (("KV growth", replay.cache),)
```

每个 builder 返回 `list[mx.array]`。返回 list 而不是单个张量，是因为 MLX 惰性求值：没人要的中间结果在求值时会整个被跳过。全部收进 list，`mx.eval(*outputs)` 才会真的算。

**projections**（344 行）。逐层跑：

```python
query = project(hidden, attention.wq)
key   = project(hidden, attention.wk)
value = project(hidden, attention.wv)
attention_output = project(concat(...), attention.wo)
gate = project(mlp_input, layer.mlp.w_gate)
up   = project(mlp_input, layer.mlp.w_up)
mlp_output = project(gate + up, layer.mlp.w_down)
```

最后是输出头。28 层，每层七个矩阵乘加一次输出头。

**attention**（374 行）。逐层跑注意力，按 feature 分派到三条路之一：tiled prefill、decode kernel、可读的 grouped attention。

**pointwise**（418 行）。embedding 查表、两个 layernorm、q_norm/k_norm、rope、swiglu、残差加法，最后是 final norm。

**cache**（445 行）。只在 decode 时加入，模拟一次 append 的 concat，28 层各一次。

## 计时

`benchmark_groups`（249 行）：

```python
orders = [builders[offset:] + builders[:offset] for offset in range(len(builders))]
```

生成四份轮转顺序。warmup 和正式测量每次都换一个起点，否则 projections 永远第一个跑，每次吃冷缓存，后面的组吃热缓存，比出来的比例是假的。

正式测量：

```python
start = perf_counter()
evaluate(build())          # mx.eval(*outputs)
timings[name].append(perf_counter() - start)
```

`perf_counter` 包住 `mx.eval`，测的是算完的时间，不是搭图的时间。取中位数而不是均值。

## 输出

回到 `profile_case`：

```python
total = sum(value for _, value in measured)
categories = [CategoryResult(name, value, value / total) for name, value in measured]
```

占比是「这一组的时间 ÷ 四组时间之和」，不是占一次 decode 的墙钟时间。

实际输出长这样：

```
projections                                16540.6 us  69.5%
attention                                   1174.6 us   4.9%
normalization, position, and activation     5607.4 us  23.6%
KV growth                                    473.9 us   2.0%
```

## 三个性质

**它不是插桩。** 模型里没有一计时器。四组是各自独立重建的调用序列，每个调用都发生在 `KernelReplay` 的方法里。所以它测的是「这类算子在真实形状下反复调用要多久」，不是「模型里这段代码花了多久」。

**每组计时都含一次 `mx.eval`。** 四组就是四次同步，而真实的一次 decode 只有末尾一次。这是组份额被高估的来源，尤其是本身工作量小于一次同步的组。KV growth 那 473.9 µs 里有约 113 µs 是同步开销。

**轮转是为了消掉顺序效应。** 没有它，第一组永远吃冷缓存，比出来的比例就没有意义。
