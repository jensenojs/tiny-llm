import mlx.core as mx
from .quantize import QuantizedWeights, dequantize_linear


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        assert weight.shape == (vocab_size, embedding_dim)
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        # 查表: 每个整数换成一整行
        #   x      (B, L)       int32, 行号
        #   weight (V, E)       bf16, 词表矩阵
        #   返回   (B, L, E)    每个位置取 weight 的对应行
        return self.weight[x]

    def as_linear(self, x: mx.array) -> mx.array:
        # 出口: 每个位置的 E 维向量和整张词表点积
        #   x (B, L, E) @ weight.T (E, V) -> (B, L, V)
        return x @ self.weight.T


class QuantizedEmbedding:
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        weight: QuantizedWeights,
        use_custom_kernel: bool = False,
    ):
        self.use_custom_kernel = use_custom_kernel
        # 4bit 是有损压缩: 每组一个 scale, 组内只记 16 级整数, 原始 BF16 的尾数信息在打包时就丢掉了.
        # 这里的还原不找回信息, 只是把打包格式换回可计算的 BF16 —— 省的是显存和下载体积, 计算照旧走 BF16.
        self.weight = dequantize_linear(weight)
        assert self.weight.shape == (vocab_size, embedding_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight[x]

    def as_linear(self, x: mx.array) -> mx.array:
        return x @ self.weight.T
