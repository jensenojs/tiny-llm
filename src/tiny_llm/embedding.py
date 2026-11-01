import mlx.core as mx
from .quantize import QuantizedWeights, dequantize_linear


class Embedding:
    """渲染与对账入口, 对应框架那一侧 mlx_model.model.embed_tokens.

.venv/bin/python debug_repl.py
print(mlx_model.model.embed_tokens)
print(tuple(mlx_model.model.embed_tokens.weight.shape),
      mlx_model.model.embed_tokens.weight.dtype)

渲染:
QuantizedEmbedding(151936, 1024, group_size=128, bits=4, mode=affine)
(151936, 128) mlx.core.uint32

判据: 磁盘上是打包表 (151936,128) uint32, 一个 uint32 装 8 个 4bit 权重;
dequantize_linear 还原成 (151936,1024) bf16 后才进本类 (见 Qwen3ModelWeek1.__init__).
查表 weight[x] 与出口 as_linear (x @ weight.T) 共用同一张表; tie=True 时没有第二张,
出口打分就是"这个位置的向量与每个词条的向量做点积".
__call__ 的形状流注释见方法内. 接缝: 查表输出 (B,L,E) 进 28 层 block;
出口 as_linear 的 (B,L,V) 在 generate.py 取最后位置.
"""

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
