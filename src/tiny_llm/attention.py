import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    k_t = mx.swapaxes(key, -1, -2)
    qk = mx.matmul(query, k_t)
    if scale is None:
        DIM_D = query.shape[-1]
        scale = 1.0 / (DIM_D**0.5)
    qk = qk * scale
    if mask is not None:
        qk = qk + mask
    return mx.matmul(softmax(qk, -1), value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.num_heads = num_heads
        self.D = hidden_size // num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        # step 1 ① 投影: E → H*D
        q = linear(query, self.wq)
        k = linear(key, self.wk)
        v = linear(value, self.wv)
        # step 2 ② 拆头
        q_split = mx.reshape(q, (*q.shape[:-1], self.num_heads, self.D))
        k_split = mx.reshape(k, (*k.shape[:-1], self.num_heads, self.D))
        v_split = mx.reshape(v, (*v.shape[:-1], self.num_heads, self.D))
        # ③ 换轴: L 和 H 交换
        q_swap = mx.swapaxes(q_split, -3, -2)
        k_swap = mx.swapaxes(k_split, -3, -2)
        v_swap = mx.swapaxes(v_split, -3, -2)
        # 4 attention
        a = scaled_dot_product_attention_simple(q_swap, k_swap, v_swap, mask=mask)
        # 5 换回去
        a_swap = mx.swapaxes(a, -3, -2)
        # 6 拼头
        a_reshape = mx.reshape(a_swap, (*a_swap.shape[:-2], self.num_heads * self.D))
        # 7 混合
        return linear(a_reshape, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    if L > S:
        raise ValueError("L should be <= S")

    i = mx.arange(L).reshape(L, 1)
    j = mx.arange(S).reshape(1, S)

    allowed = j <= i + (S - L)
    mask = mx.where(allowed, 0.0, float("-inf"))
    return mask.astype(dtype)


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    # 1. 头数检查
    H_q = query.shape[-3]
    H = key.shape[-3]
    n_repeats = H_q // H
    if H_q % H != 0:
        raise ValueError("xxx")

    # 2. 拼车
    L, D = query.shape[-2], query.shape[-1]
    S = key.shape[-2]
    q = query.reshape(*query.shape[:-3], H, n_repeats, L, D)
    k = key.reshape(*key.shape[:-3], H, 1, S, D)
    v = value.reshape(*value.shape[:-3], H, 1, S, D)

    # 3. 打分
    scores = mx.matmul(q, mx.swapaxes(k, -1, -2))
    if scale is None:
        scale = 1.0 / (D**0.5)
    scores = scores * scale

    # 4. 掩码
    if mask == "causal":
        mask = causal_mask(L, S, query.dtype)
    elif mask is not None:
        # 什么情况下需要reshape?
        # 第三种要 reshape 的原因：它按 Q 头数 H_q 组织——每个 Q 头带一张 (L, S) 表。但我们的 scores 头维已经拆成了两层 (H, n_repeats)。要相加，mask 的头维也得跟着拆：
        # (..., H_q, L, S)  →  (..., H, n_repeats, L, S)
        # 而 "causal" 现算出来的 (L, S) 不用 reshape——它没有头维，广播时自动扩到 (..., H, n_repeats)（从右往左对齐，缺的维按 1 处理）。
        mask = mask.reshape(*mask.shape[:-3], H, n_repeats, L, S)

    if mask is not None:
        scores = scores + mask

    #
    weights = mx.softmax(scores, -1)  # (..., H, n_repeats, L, S)
    out = mx.matmul(weights, v)  # (..., H, n_repeats, L, D)

    return out.reshape(*out.shape[:-4], H_q, L, D)


def paged_attention(
    query: mx.array,
    key_pages: mx.array,
    value_pages: mx.array,
    block_table: mx.array,
    context_lens: mx.array,
    page_size: int,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    """Attend to paged K/V storage without reconstructing a dense cache.

    Week 3 Day 4 owns the correctness-first decode and long-prefill paths for
    both float32 and BF16. Day 5 optimizes the same public boundary.
    """
    pass
