# Week 2 Day 1 基线: 缓存与测量

这份记录是 Day 2 的对照物。Day 2 改完投影路径之后，拿同一份 workload 重跑，和这里比。

## 环境

```
机器      Apple M4, 10 核 GPU, 16 GB 统一内存
系统      macOS 27.2 arm64
MLX       0.32.0
mlx-lm    0.31.3
Python    3.10.18
模型      Qwen/Qwen3-0.6B-MLX-4bit
```

## Workload

```
input_tokens     128
output_tokens    129
warmup           2
repeats          2 (交替顺序, 减少热漂移)
prefill_logits   all
seed             0
```

`prefill_logits=all` 是因为 Week 1 会对每个 prompt 位置算 logits，只有两边都算全部位置才是算法层面的对比。服务场景要的是 `last`，那是另一份 workload，不能混用。

## Benchmark

`benches/bench_course_progression.py`，三次冷启动各跑两遍取中位数。

| Checkpoint | Prefill tok/s | vs Week 1 | Decode tok/s | vs Week 1 |
|---|---:|---:|---:|---:|
| Week 1 readable | 242.57 | baseline | 1.00 | baseline |
| kv-cache | 1742.20 | 7.18× | 34.20 | 34.38× |
| capacity-cache | 1728.97 | 7.13× | 37.77 | 37.96× |

原始记录：`week2-day1-cache-ladder.json`。

prefill 快 7 倍来自只算一次；decode 快 34 倍来自每步 `L=1` 而不是 `L=S`。capacity 把 decode 再提高 10.4%，因为不再每步 `concatenate` 整段历史。它的 prefill 低 0.8%，两次样本的波动是 ±20%，这个差在噪声内。

## Profiler

`benches/profile_week2_kernels.py`，`capacity-cache:decode:128`，warmup 4、iterations 12。

```
projections                                16540.6 us  69.5%
attention                                   1174.6 us   4.9%
normalization, position, and activation     5607.4 us  23.6%
KV growth                                    473.9 us   2.0%
```

原始记录：`week2-day1-attribution.json`。`decode:128` 表示计时前缓存里已有 127 个 token，计时那一步加第 128 个。

### 它解决什么问题

一次 decode 的墙钟时间只说「慢了」，不说「哪里慢」。要决定下一步改什么，得知道时间落在哪类算子上。直接给整个 `model(...)` 计时做不到这一点，因为 28 层、每层七个矩阵乘、一次注意力、两次归一化全串在一起，测出来只有一个总数。

profiler 换了个做法：把一步 decode 拆成四组，每组单独重放、单独计时。

### 重放怎么构造

`KernelReplay.__init__`（`profile_week2_kernels.py:290-342`）先拿第一层的配置读出形状：

```python
first = model.layers_inner[0]
self.hidden_size        = first.hidden_size
self.num_heads          = first.num_attention_heads
self.num_kv_heads       = first.self_attn.num_kv_heads
self.head_dim           = first.self_attn.head_dim
self.intermediate_size  = first.mlp.hidden_dim
self.dtype              = model.precision
```

然后用这些形状造随机输入：

```python
self.hidden = mx.random.normal((1, rows, hidden_size))
self.query  = mx.random.normal((1, num_heads, rows, head_dim))
self.key    = mx.random.normal((1, num_kv_heads, context, head_dim))
```

`rows` 在 decode 时是 1，prefill 时是 token 数；`context` 是缓存里的总长度。这些张量在 `__init__` 末尾就被 `evaluate(...)` 强制算出来，正式计时时拿来即用，不含分配开销。

权重是真的。`model` 由 `implementation.model_type(mlx_model, checkpoint=case.checkpoint)` 构造，与 benchmark 里的模型同一份。

### 四组分别是什么

**projections**（`344-372`）：逐层跑 `wq/wk/wv/wo` 和 `w_gate/w_up/w_down`，最后是输出头。每层的中间结果 `outputs.extend((key, value))` 收进列表。

**attention**（`374-416`）：用同一组 q/k/v 跑每层注意力。路径按 `use_tiled_prefill_attention` 和 `should_use_decode_attention` 选择，与生产代码的分派条件一致。

**pointwise**（`418-443`）：embedding 查表、两个 layernorm、q_norm/k_norm、rope、swiglu、残差加法，最后是 final norm。

**KV growth**（`445-460`）：只在 decode 时加入，模拟一次 append 的 concat。

四组都返回 list 而不是单个张量。MLX 是惰性的，没人要的中间结果在求值时会被整个跳过。全部收进 list，`mx.eval(*outputs)` 才会真算。

### 计时怎么保证公平

`benchmark_groups`（`249-266`）：

```python
orders = [builders[offset:] + builders[:offset] for offset in range(len(builders))]
```

生成四份轮转顺序，warmup 和正式测量每次换一个起点。否则 projections 永远第一个跑，每次吃冷缓存，后面的组吃热缓存，比出来的比例是假的。

`test_kernel_group_profile_rotates_every_group_through_each_position` 把这16 次调用顺序逐项钉死，就是为了让轮转不被无意改掉。

每组的时间这样取：

```python
start = perf_counter()
evaluate(build())          # mx.eval(*outputs)
timings[name].append(perf_counter() - start)
return {name: median(samples) * 1_000_000 for ...}
```

`perf_counter` 包住 `mx.eval`，测的是算完的时间，不是搭图的时间。取中位数而不是均值，避免单次调度抖动拉偏。

占比由 `profile_case`（`482-485`）算：四组中位数相加当分母。

### 测试在守护什么

`benches/test_profile_week2_kernels.py` 九条，覆盖四类不变量：

- 轮转顺序精确匹配（`test_kernel_group_profile_rotates_every_group_through_each_position`）
- 注意力分派与生产守卫一致，边界值包括 query 长度、context 长度、显式 mask
- tiled prefill 在 `selected` checkpoint 下真的走到 `dense_prefill_attention_mma`
- 决策记录的 `source`、`solution`、`model`、`workload_id` 必须完全一致，否则拒绝合并
- workload 哈希覆盖每个字段：模型、phase、tokens、warmup、iterations、prompt rule、prefill logits
- 输出文件已存在或路径别名时，在加载模型之前就拒绝

最后两条保证的是「记录可信」：两次运行如果 workload 有一项不同，哈希就不同，不能拿来比较。

### 这个方法的边界

四组是分开跑的，没有串行依赖。真实 decode 里 attention 要等 projections 的输出，这里有等待却没有被计入。

每组只测自己那批算子，不含调用开销和 kernel launch 的间隙。

`replay_total_us` 是四个中位数之和，不是一次 decode 的墙钟时间。

所以它回答的是「哪类算子在反复调用时最贵」，不是「一步 decode 要多久」。后者看 benchmark 的 decode tok/s。

## 结论

decode 的主项是 dense projections，占 69.5%。这与 roofline 一致：bf16 权重每字节只对应一次乘加，GPU 大部分时间在等内存。

## 计划

> 这个 replay 里 dense projections 占 69.5%，是 decode 的主项。
>
> 把权重保持 4 bit、只改投影路径，应该让这一项下降，并让整个请求的 decode 变快。
>
> 如果 replay 里的投影时间没降，或者同样 workload 下 decode 变慢，我就放弃这个想法。

第三句是失败判据。没有它，任何结果都能被解释成有进展。
