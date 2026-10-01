import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # 误差管理: x**2 与 mean 是归约点 —— D 个输入的舍入误差汇入同一个结果,
        # 累积量约 √D 倍; bfloat16 下 x**2 还会下溢(最小正规数 ~1e-38).
        # 所以统计量(mean/sqrt/除法)应全程在 float32 下计算, 归一化完
        # astype 回 x.dtype 再乘 weight —— 与 MLX fast kernel 的低精度路径一致.
        # 除法和乘 weight 是逐元素运算, 误差不被放大, 留在模型精度即可.
        x_f32 = x.astype(mx.float32)
        norm = mx.sqrt(mx.mean(x_f32**2, axis=-1, keepdims=True) + self.eps)
        normalize = (x_f32 / norm).astype(x.dtype)
        return self.weight * normalize
