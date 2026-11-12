from typing import Any

import mlx.core as mx


def dequantize_linear(mx_layer: Any) -> mx.array:
    """把量化层还原成稠密 BF16 权重矩阵（Task 3 加载 4bit 模型时每层都要过这里）.

    输入是 mlx_lm 加载的量化层（如 Qwen3-0.6B-MLX-4bit 的 embed_tokens），字段：
      weight:     打包后的位流（uint32 数组），组内每 bits 位存一个整数 q
      scales:     每组一个缩放因子（FP16）
      biases:     每组一个偏移，与 scales 同形
      group_size: 每组多少个权重（如 32）；bits: 每个权重几位（如 4）

    存储方案（分组仿射量化）：每 group_size 个权重为一组，组内共享同一个
    scale 和 bias；组内每个权重只存一个 bits 位整数 q。以 4bit/组 32 为例，
    32 个 16bit 权重压成 32*4 + 16 + 16 bit ≈ 4.5 bit/权重，体积约 8:1。

    数学关系：量化时 q = round(w / scale + bias)，所以还原就是反着算
    w ≈ (q −(+?) bias) × scale。这是有损压缩——原始 BF16 的尾数在打包时已经
    丢掉，每个数带百分之几的相对误差；本函数只做格式转换（位流 → 可计算的
    稠密矩阵），不找回信息，省的是显存和下载体积。

    返回与原权重同形状的 BF16 数组，可直接查表、参与矩阵乘。
    """
    w = mx.dequantize(
        mx_layer.weight,
        mx_layer.scales,
        mx_layer.biases,
        mx_layer.group_size,
        mx_layer.bits,
    )
    return w.astype(mx.bfloat16)


class QuantizedWeights:
    def __init__(
        self,
        scales: mx.array,
        biases: mx.array,
        group_size: int,
        bits: int,
        weight: mx.array,
        use_simdgroup_matmul: bool = False,
        use_simdgroup_matvec: bool = True,
        use_split_k_matmul: bool = False,
        use_mlx_quantized_linear: bool = False,
    ):
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.weight = weight
        self.use_simdgroup_matmul = use_simdgroup_matmul
        self.use_simdgroup_matvec = use_simdgroup_matvec
        self.use_split_k_matmul = use_split_k_matmul
        self.use_mlx_quantized_linear = use_mlx_quantized_linear

    @staticmethod
    def from_mlx_layer(
        mlx_layer: Any,
        use_simdgroup_matmul: bool = False,
        use_simdgroup_matvec: bool = True,
        use_split_k_matmul: bool = False,
        use_mlx_quantized_linear: bool = False,
    ) -> "QuantizedWeights":
        biases = mlx_layer.biases
        return QuantizedWeights(
            scales=mlx_layer.scales.astype(mx.bfloat16),
            biases=None if biases is None else biases.astype(mx.bfloat16),
            group_size=mlx_layer.group_size,
            bits=mlx_layer.bits,
            weight=mlx_layer.weight,
            use_simdgroup_matmul=use_simdgroup_matmul,
            use_simdgroup_matvec=use_simdgroup_matvec,
            use_split_k_matmul=use_split_k_matmul,
            use_mlx_quantized_linear=use_mlx_quantized_linear,
        )


def mlx_quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    pass


def quantized_matmul(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
    use_simdgroup: bool = False,
    use_split_k: bool = False,
) -> mx.array:
    # 扩展的 .so 只在跑过 build-ext 后存在。导入放在函数体里，
    # 模块导入时不要求它已构建。
    from extensions.tiny_llm_ext import _ext

    # 最后一维是点积长度 N，必须整段保留；其余维（batch、序列位置）彼此独立，
    # 乘的是同一张权重，压成一维当 M。二维输入时 lead 只有 M 一项。
    *lead, D = a.shape
    a2 = a.reshape(-1, D)

    """
Metal 只拿到基地址, 按 i*N+j 算位置, 要求行主序。视图不满足这个前提:

x = mx.arange(12).reshape(3, 4)
t = x.T                        # 视图, 与 x 共用一块 buffer

buffer      [0][1][2][3][4][5][6][7][8][9][10][11]

x (3,4)     地址 = i*4 + j
  行0        0  1  2  3
  行1        4  5  6  7
  行2        8  9 10 11

t (4,3)     地址 = j*4 + i
  行0        0  4  8
  行1        1  5  9
  行2        2  6 10
  行3        3  7 11

kernel 按自己假设的 i*3+j 读 t:
  i=0 j=1   读到 1, 而 t[0,1] 是 4
  i=1 j=0   读到 3, 而 t[1,0] 是 1

mx.contiguous(t) 把底层重排成
  [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
之后 i*3+j 才成立。已经连续的数组不复制, 函数直接返回原数组。

四个输入都要过一遍: scales 与 biases 按 k*groups_per_row 取, b 按 k*N/8 取,
它们同样可能来自切片或转置。推导见 docs/mlx-array-lifecycle.md 的
「视图与连续布局」。参数顺序与 bindings.cpp:36-37 的注册顺序一致。
    """
    out = _ext.quantized_matmul(
        mx.contiguous(scales),
        mx.contiguous(biases),
        group_size,
        bits,
        mx.contiguous(a2),
        mx.contiguous(b),
        transpose_b,
        use_simdgroup,
        use_split_k,
    )

    # 按 a 原来的前导维还原，最后一维换成 K = b.shape()[0]。
    return out.reshape(*lead, -1)


def dequantize_weights(
    weight: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    group_size: int,
    bits: int,
) -> mx.array:
    """
    两个数组做逐元素运算时，从最右边一维开始往左对齐，逐维比较：
      - 其中一个为 1 → 可以，结果取另一个的大小（逻辑复制）；
        - 其中一个“不存在” → 视为 1；
      - 相等, 可以, 都大于 1 且不相等, 报错


    weight 是 uint32 数组。每个元素是一个 32 位整数，不是“4 个字节的数组”。
    4bit 量化时，MLX 把 8 个 4bit 的 q 塞进这一个 uint32。低位优先的布局长这样：

    bit 31       24 23       16 15        8 7         0
    [   q7       ][   q6     ][   q5     ][   q4    ][...]

    step1 : 从这一个整数里，把 q0 ~ q7 八个 4bit 值抠出来, 拆包里 weight 缺“移位量”这一维, 在末尾拆 None, 广播出去
    i.e weight(位流) → q(整数序列)

    weight:  (..., M)        M 个 uint32
    ↓ 拆包
    q:       (..., M, n_pack)   每个 uint32 变出 n_pack 个 q
    ↓ 摊平
    q:       (..., M * n_pack)  每个权重一个整数

    核心手法：(x >> shift) & mask
    x      = [ ... q1 的 4 个 bit ... 低 4 位 ... ]
    x >> 4 = [ ... 高位 ... ][ q1 ][ 原来的低 4 位挤掉 ]


    shifts = mx.arange(0, 32, bits)   # [0, 4, 8, 12, 16, 20, 24, 28]，形状 (8,)
    mask = (1 << bits) - 1            # 0b1111，标量
    weight[..., None]                 # 形状 (..., M, 1), reshape 是为了让“组内共享 scale”这件事能用广播表达。
    q = (weight[..., None] >> shifts) & mask

    什么需要广播: 要对每个 uint32 做 n_pack 次不同的右移。shifts 是 (n_pack,)，它要和“每个 uint32”组合。

    广播后：
    - weight[..., None] → (..., M, 1)
    - shifts → (8,)，广播成 (..., M, 8)
    - 每个 uint32 被右移 8 个不同的量，得到 8 个版本, 每个 uint32 独立地、和所有 shift 组合一次。q[..., m, j] = 第 m 个 uint32 右移 shifts[j] 位后取低 bits 位。
    - & mask 每个版本只留低 4 位
    - 结果 q 形状 (..., M, 8)，q[..., m, j] 就是第 m 个 uint32 里的第 j 个 4bit 数

    最后：q = q.reshape(*weight.shape[:-1], -1)  # (..., M, 8) → (..., M*8)

    step2 : 反量化
    q + scales + biases → w(浮点矩阵)
    拆包后有一串整数 q，形状 (..., in_dims)。但 scales 只有 (..., num_groups) 个，一个 scale 管 group_size 个 q。要把“每组一个 scale”应用到“组内每个 q”上。

    q:       (..., in_dims)
    ↓ reshape
    q:       (..., G, S)         G = num_groups, S = group_size
    scales:  (..., G)
    ↓ 插 None
    scales:  (..., G, 1)
    ↓ 广播
    scales:  (..., G, S)         每组 scale 复制 S 份
    ↓ 公式
    w:       (..., G, S) → 摊平 → (..., in_dims)
    """

    # step1
    shifts = mx.arange(0, 32, bits)
    mask = (1 << bits) - 1
    q = (weight[..., None] >> shifts) & mask  #  q 形状 (K, M, 32/bits), uint32

    q = q.reshape(
        *weight.shape[:-1], -1
    )  # (K, M, 8) -> (K, N) 差别在「reshape 是拆分已有轴，None 是插入新轴」。
    q = q.reshape(
        *q.shape[:-1], -1, group_size
    )  # (K, N) -> (K, G, S), 把「位置」换成「组号 × 组内偏移」

    # step2
    w = q * scales[..., None]
    if biases is not None:
        # (K, G, S)
        w = w + biases[..., None]

    w = w.reshape(*weight.shape[:-1], -1)  # (K, N)
    return w.astype(mx.bfloat16)


def quantized_matvec_custom(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    pass


def quantized_matmul_vanilla(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    pass


def quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    pass
