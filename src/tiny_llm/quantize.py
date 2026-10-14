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
    w ≈ (q − bias) × scale。这是有损压缩——原始 BF16 的尾数在打包时已经
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
    pass


def dequantize_weights(
    weight: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    group_size: int,
    bits: int,
) -> mx.array:
    pass


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
