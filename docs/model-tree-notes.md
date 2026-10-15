# mlx_lm 模型对象与权重树: Task 3 装配地图

Task 3 要把 `mlx_lm` 加载的真模型拆开，用我们自己写的零件重新拼一遍。本文记录那个模型对象长什么样、每个字段是什么、怎么搬进我们的类。所有数字以 Qwen3-0.6B-MLX-4bit 为例。

## 对象从哪来

```python
from mlx_lm import load
mlx_model, tokenizer = load("Qwen/Qwen3-0.6B-MLX-4bit")
print(mlx_model.model)   # 打印权重树
print(mlx_model.args)    # 打印配置字典
```

`load` 返回两个东西: `mlx_model` 是整个模型对象，`tokenizer` 是分词器。模型对象上我们只关心两块:

- `mlx_model.args` —— 配置字典，全是数字（形状参数），决定**怎么构造**；
- `mlx_model.model` —— 权重树，全是张量和子模块，是**搬运的货源**。

## 读树: 每个字段是什么

```
Qwen3Model(
  (embed_tokens): QuantizedEmbedding(151936, 1024, group_size=128, bits=4, mode=affine)
  (layers.0): TransformerBlock(
    (self_attn): Attention(
      (q_proj): QuantizedLinear(input_dims=1024, output_dims=2048, bias=False, group_size=128, bits=4, mode=affine)
      (k_proj): QuantizedLinear(input_dims=1024, output_dims=1024, bias=False, ...)
      (v_proj): QuantizedLinear(input_dims=1024, output_dims=1024, bias=False, ...)
      (o_proj): QuantizedLinear(input_dims=2048, output_dims=1024, bias=False, ...)
      (q_norm): RMSNorm(128, eps=1e-06)
      (k_norm): RMSNorm(128, eps=1e-06)
      (rope): RoPE(128, traditional=False)
    )
    (mlp): MLP(
      (gate_proj): QuantizedLinear(input_dims=1024, output_dims=3072, ...)
      (down_proj): QuantizedLinear(input_dims=3072, output_dims=1024, ...)
      (up_proj): QuantizedLinear(input_dims=1024, output_dims=3072, ...)
    )
    (input_layernorm): RMSNorm(1024, eps=1e-06)
    (post_attention_layernorm): RMSNorm(1024, eps=1e-06)
  )
  (layers.1 ... layers.N 同构)
  (norm): RMSNorm(1024, eps=1e-06)          # 树打印到这里被截断, 往下还有
  (lm_head): ...                             # 绑定时可能不存在, 用 hasattr 检查
)
```

逐层解读:

| 字段 | 是什么 | 证据 |
|---|---|---|
| `embed_tokens` | 词表: 151936 个 token，每个 1024 维向量，4bit 存着 | `QuantizedEmbedding(151936, 1024, ...)` |
| `q_proj` | Q 投影 1024 → 2048 | 2048 = 16 个 Q 头 × 128 维/头 |
| `k_proj` `v_proj` | K/V 投影 1024 → 1024 | 1024 = **8 个** KV 头 × 128——只有 Q 头数的一半，这就是 GQA 的直接证据: `n_repeats = 16 // 8 = 2` |
| `o_proj` | 输出投影 2048 → 1024，把 16 个头拼回 E | 方向和 q_proj 相反 |
| `q_norm` `k_norm` | 每头内部的 Q/K 归一化，γ 长 128 | `RMSNorm(128)`，作用在 head_dim 上，**不是** 1024 |
| `rope` | 位置旋转操作 | 只有配置没有权重，是个算子 |
| `gate_proj` `up_proj` | MLP 升维 1024 → 3072 | 3072 = 3 × 1024（Qwen3 系用 3 倍，有的模型用 4 倍） |
| `down_proj` | MLP 降维 3072 → 1024 | gate/up 的逆操作 |
| `input_layernorm` | 注意力子层门口的 Pre-LN，γ 长 1024 | Pre-LN 结构: LN 在旁路 |
| `post_attention_layernorm` | MLP 子层门口的 Pre-LN | 同上 |
| `norm`（树尾） | 整个模型出口的 final norm | 最后一层的最后一个算子 |
| `lm_head`（树尾） | 出口投影；0.6B 权重绑定，可能不存在 | 用 `hasattr(mlx_model.model, "lm_head")` 检查 |

## QuantizedLinear 内部: 从黑盒到三件套

`print(q_proj)` 只给超参, `vars(q_proj)` 只有 `_no_grad` 和几个标量, 连 `dir(q)` 都列不出 `scales`（实测 False——默认 `__dir__` 合并的是实例 `__dict__` 和类属性, 看不到 dict 基类存储）. 常规的摸索手段在这个框架里全部失灵, 零起点只剩一条可靠的路:

**问出生地, 读定义.** `type(q)` 给出 `mlx.nn.layers.quantized.QuantizedLinear`——`type(q).__module__` 就是定义文件的模块路径. 打开 `.venv/lib/python3.*/site-packages/mlx/nn/layers/quantized.py`, 读 `__init__` 的赋值: 参数清单的权威出处在这里. `weight` `scales` `biases` 三个名字、形状怎么算、还原公式怎么写, 全在这一段里. 定义处永远优先于文档和 repr.

**取值碰壁时, 冲突指向机制.** `q.scales` 拿得到值, `vars(q)` 里却没有——属性能访问与 vars 看得到, 两个事实冲突, 说明存在另一套存储. 顺去看 `base.py`: `Module` 继承自 `dict`（base.py:12）, `__setattr__`（base.py:105）按赋值类型分流, `mx.array`/dict/list 进 dict 存储（`self[key] = val`）, 其余走普通实例 `__dict__`; `.scales` 是 `__getattr__` 先查 dict 存储（base.py:99）. 直接验证: `isinstance(q, dict)` 为 True, 所以 dict 的接口全部可用:

```python
q = m.model.layers[0].self_attn.q_proj
q.keys()      # dict_keys(['weight', 'scales', 'biases'])
q['scales']   # 等价 q.scales: __getattr__ 先查 dict 存储（base.py:99）
```

`_no_grad` 是冻结参数名单（base.py:511）: 量化模型加载即 freeze, 全部参数无梯度. repr 是作者手写的摘要, 只显示超参——摘要与内容是两回事.

三件套（以 q_proj 2048×1024, group_size=128, bits=4 为例）:

| 属性 | 形状 | 内容 |
|---|---|---|
| `weight` | (2048, 16) uint32 | 打包的 4bit 码. 每 128 个权重一组, 组内每档 0..15, 8 档拼进一个 uint32; 1024/128=8 组, 每组 128×4=512bit=16×uint32 |
| `scales` | (2048, 8) bf16 | 每组一个缩放 |
| `biases` | (2048, 8) bf16 | 每组一个偏移 |

`group_size` `bits` 是两个标量参数, 还原时用, 不随矩阵大小变.

数学是一对互逆（有损）运算:

$$w \approx \text{scales} \times q + \text{biases}, \qquad q = \mathrm{round}\!\left(\frac{w - \text{biases}}{\text{scales}}\right)$$

代入实测第一组数: scale=0.0064, bias=−0.0518, q 跨 0..15, 组内覆盖 ≈ [−0.052, 0.045], 步长 0.0064. 损失全部来自四舍五入到档, 最多半档. bias 的作用是把组的起点搬过去——4bit 无符号, 靠它把负区间摆上非负整数轴, 这就是 `mode=affine` 的字面含义.

为什么分组: 整个矩阵共享一对 scale/bias, 步长被全局范围撑大, 小权重全塌进 0 档; 每个权重独立一对, 附属物 2×2 字节/权重, 比本体的 0.5 字节还大 8 倍, 压缩变膨胀. 每 128 个共享一对是折中: 附属开销 4/128 = 3.2%, 总成本 ≈ 4.5bit/权重, 体积比 BF16 小 8 倍. group_size 就是精度与开销的旋钮, 64 更准更肥, 128 是惯例. 硬约束只有两条: 整除 input_dims, 且 bits×group_size 能被 32 整除（打包满 uint32）; 320 也合法（320×4=1280=40×uint32）, 不用它只是没有偏离惯例的理由.

复刻对照: 课程 `dequantize_linear` 在 `__init__` 里一次还原成普通 BF16 数组——下载体积省 8 倍, 运行内存回到 BF16（0.6B 约 0.3GB → 1.2GB）. mlx 自己的 `QuantizedLinear.__call__` 走融合核 `quantized_matmul`, 不拆包, 真 4bit 驻留. 两种算法等价, 内存形态不同.

## 三种字段，三种搬运方式

树上只有三类东西，每类一个固定动作:

1. **量化大张量**（`QuantizedLinear` / `QuantizedEmbedding`）→ 过 `dequantize_linear` 还原成普通数组。所有投影和词表都是这类。
2. **普通小数组**（`RMSNorm` 的 γ: `q_norm` `k_norm` `input_layernorm` `post_attention_layernorm` `norm`）→ 归一化参数太小，不值得量化，直接 `.weight` 拿。
3. **无权重的算子**（`rope`）→ 不搬。它只是配置，我们用自己的 `RoPE(dims=head_dim, seq_len=max_seq_len, base=theta, traditional=False)` 从 `args` 重建。

## 数字对一遍: head_dim 陷阱的实例

把树上读到的数和 `args` 对一遍（跑 `print(mlx_model.args)` 核对，层数和上下文长度以打印为准）:

| 配置项 | 值 | 来源 |
|---|---|---|
| vocab_size | 151936 | 树: embed_tokens 第一个数 |
| hidden_size (E) | 1024 | 树: 几乎所有 1024 |
| intermediate_size | 3072 | 树: gate/up 的 output_dims |
| num_attention_heads | 16 | 2048 / 128 |
| num_key_value_heads | 8 | 1024 / 128 |
| head_dim | **128** | q_norm 的 RMSNorm(128) |
| rms_norm_eps | 1e-06 | 树: 每个 RMSNorm 的 eps |
| rope_theta | 1000000 | args |
| num_hidden_layers | 28（以 args 为准） | 树有 layers.0 ... layers.N |
| tie_word_embeddings | True（0.6B） | args |

注意 head_dim 那一行: **16 个头 × 128 维 = 2048 ≠ hidden_size 1024**。所以 `hidden_size // num_heads` 算出 64 是错的——q_proj 的输出宽度是 2048 不是 1024。这就是书里反复强调 "head_dim 从 config 独立读" 的原因，这棵树就是活例子。顺带: q_proj output(2048) ≠ o_proj input? 不，o_proj input_dims=2048 正好吃下 16 个头拼起来的宽度，再压回 1024。

另一个容易看漏的: `linear` 层的权重形状约定是 **(output_dims, input_dims)**，乘的时候转置——我们 `basics.linear(x, w)` 内部已经做 `x @ w.T`，搬运时不用动。

## `__init__`: 一张翻译表

参数多是好事，说明 `__init__` 只是机械翻译，不需要发明任何东西。分拣原则按上面的三类:

```python
def __init__(self, mlx_model):
    args = mlx_model.args
    # ① 从 args 读形状数字（循环层数、各维度、eps、theta、tie 开关）
    # ② dequantize_linear 搬大张量: embed_tokens + 每层的 7 个投影
    # ③ .weight 直取小数组: q_norm/k_norm（喂给 Block）+ 两层门口 norm + final norm
    # ④ 用 args 重建 RoPE（max_seq_len 取 args.max_position_embeddings）
    # ⑤ 出口: tie=True 用 embed.as_linear；False 才搬 lm_head
    # ⑥ 每层组一个 Qwen3TransformerBlock（你在 Task 1 写的构造函数原样用）, 存进列表
```

每层 7 个投影 = q/k/v/o_proj + gate/up/down_proj。逐层从 `mlx_model.model.layers[i]` 取，名字对照上面的树。

## `__call__`: 五行

```python
def __call__(self, inputs):
    # x = embedding 查表               (N, E)
    # for block in blocks: x = block(x, mask="causal")
    # x = final RMSNorm(x)             (N, E)
    # logits = 出口投影(x)             (N, vocab_size)   未归一化
    return logits
```

## 和 batch-main.py 的连接

`batch-main.py:69` 的 `models.dispatch_model(model_name, mlx_model)` 就是这套装配的批量入口——Task 3 通过后，`dispatch_model` 的 week1 分支会把 `mlx_model` 原样传给你的 `Qwen3ModelWeek1.__init__`。这就是为什么 `__init__` 的签名是收一个 `mlx_model` 对象: 调用方（dispatch）已经替你 load 好了。

## 检查清单

- [ ] `print(mlx_model.args)` 跑过，层数/theta/max_position_embeddings 以打印为准
- [ ] `hasattr(mlx_model.model, "lm_head")` 试过，确认绑定模型的出口走哪条
- [ ] head_dim 从 `args.head_dim` 读，没有出现 `hidden_size // num_heads`
- [ ] 每个投影都过了 `dequantize_linear`；所有 γ 都是 `.weight` 直取
- [ ] Day 3 注意力里的 `mx.fast.rms_norm` 已换成自己的 `RMSNorm`
- [ ] `pdm run test --week 1 --day 5 -- -k task_3` 全绿
