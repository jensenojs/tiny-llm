import mlx.core as mx


class RMSNorm:
    """渲染与对账入口, 对应框架那一侧 layers[0].input_layernorm.

.venv/bin/python debug_repl.py
from tiny_llm.layer_norm import RMSNorm
ln0 = mlx_model.model.layers[0].input_layernorm
print(ln0, tuple(ln0.weight.shape), ln0.weight.dtype)
norm = RMSNorm(1024, ln0.weight)
x = mlx_model.model.embed_tokens(mx.array([[1, 2, 3]], dtype=mx.int32))
a = norm(x).astype(mx.float32)
b = mx.fast.rms_norm(x, ln0.weight, eps=1e-6).astype(mx.float32)
print(float(mx.abs(a - b).max()))

渲染:
RMSNorm(1024, eps=1e-06) (1024,) mlx.core.bfloat16
0.0078125

判据: γ 长度是 E=1024 (整条表示, 不是单头 D=128), 输入输出同为 (B,L,E);
eps 必须传 args.rms_norm_eps=1e-6, 构造默认 1e-5 正是 Qwen3ModelWeek1 对账 0.34375 的来源;
对账差 0.0078125 是两边把中间量铸回 bf16 的位置不同, 落在最后一位舍入 (float32 输入时为 6.7e-06).
__call__ 内的误差管理注释 (统计量全程 float32) 见方法内. 接缝:
block 里两条支路各一道 (input / post_attention), 全部 28 层跑完还有最后一道 self.norm;
norm 出错会同时污染残差主干两侧的输入, 症状是整句逐层放大.
"""

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
