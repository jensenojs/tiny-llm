import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # Supplied for Day 1; a manual implementation is an optional bonus exercise.
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    if bias is not None:
        return mx.addmm(bias, x, w.T)
    return mx.matmul(x, w.T)


def silu(x: mx.array) -> mx.array:
    # z = exp(-|x|) <= 1, 指数恒非正, 任何输入都不溢出.
    # x < 0 时 1/(1+exp(-x)) = z/(1+z), 避免先算 exp(大正数) 再溢出;
    # 也避免 1-(1/(1+z)) 变形: 1+z 的舍入会先把 z 抹掉, 减法再放大丢失.
    z = mx.exp(-mx.abs(x))
    sigmoid = mx.where(x < 0, z / (1 + z), 1 / (1 + z))
    return x * sigmoid
