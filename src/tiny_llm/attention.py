import mlx.core as mx
from .basics import softmax, linear
from .trace import stage_once


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
    # 掩码加到 scores 上, 形状 (L, S):
    #   M[i][j] = 0     不加任何东西, 放行
    #   M[i][j] = -inf  加进去后 exp 得 0, 禁止
    # 禁止只能用 -inf 说. 写 0 等于放行.
    #
    # 列是全文 S 条 token, 全部进 softmax 的分母: 多一列就改输出, KV 缓存切到
    # offset 就是为了不把没写过的槽送进来 (见 kv_cache.py 的 logical/physical).
    # 行是本次新来的 L 条, 位于全文最后 L 个位置, 行 i 的绝对位置是 (S-L)+i.
    # 只能在尾部: token 只能追加, 插队破坏因果; 重排还得重算 RoPE —— 每个 K
    # 绑死在写入时的绝对角度上.
    # 判据一行: j <= i + (S-L).  L=S 时退化成下三角 j <= i.
    #
    # prefill  offset=0, L=2, S=2   行列同号, 普通下三角
    #   位置 0  [ .    -inf]
    #   位置 1  [ .    .   ]
    #
    # decode   offset=2, L=1, S=3   只有一行, 位置 2, 历史全放行
    #   位置 2  [ .    .    .   ]
    # L=1 时全 0, 传 None 结果相同; 留着是让三种情形走同一段代码.
    #
    # 分块     offset=4, L=2, S=6   已缓存 4 条, 这次同时送 2 条
    #   位置 4  [ .    .    .    .    .    -inf]
    #   位置 5  [ .    .    .    .    .    .   ]
    #
    # 宽度由 S 决定, -inf 只由 L>1 决定: 这 2 条分两次送是两个 (1,5)、(1,6) 矩阵,
    # 都不带 -inf; 一次送完是 (2,6), 多出的 -inf 管这一批内部的先后.
    # 每次调用只算新来的 L 行: 老行的输出早已算完, 不会再变. 一次一次送时每次拿到
    # 一行, 列数逐次 +1; 把整句画成方阵, 第 i 行正是第 i 步拿到的那一行, 累计就是
    # 下三角.
    #
    # 需要 L>1 的有两处, 动机不同.
    # 分块 prefill 是为了内存. 一次算完时 scores 是 (1, 16, S, S) 的 float32,
    # S 取表高 32768: 1 × 16 × 32768 × 32768 × 4 B = 68.7 GB = 64 GiB, 一个中间
    # 数组就这么大. 按 query 切成 2048 行的块, 每块 (1, 16, 2048, 32768):
    # 1 × 16 × 2048 × 32768 × 4 B = 4.3 GB = 4 GiB. 列宽仍是 S —— 每块都要看完
    # 全部历史, 所以峰值随 S 线性增长, S² 没有被消掉. 再往下是不物化整个矩阵,
    # 只留 (block_q, block_k) 的 tile: softmax 的分母靠 running max 与 running sum
    # 在线累加, 每读一块用新的最大值修正先前的和, 完整 (L, S) 从不出场.
    # 前缀缓存是同一形状: 已算过的历史不重算, 只送新增部分, offset>0.
    # 投机解码是为了带宽: decode 每 token 只出一个, 却要读全部权重
    # 596.0M × 2 B = 1.19 GB, 只做 2 × 596.0M FLOP, 算术强度 1.0 FLOP/byte;
    # 200 GB/s 时下界 1.19 ÷ 200 = 5.96 ms 一个 token, 算力空着. 一次送 k+1 个
    # 位置, 权重仍只读一遍, 多出的乘加由空转的算力吃掉. batch 大或 prefill 阶段
    # 算力不再空闲, 收益随之消失.
    #
    # 已接受 6 个、草稿 4 个就是 k+1=5 个位置: L=5, S=11, 左边 6 格是位置 0..5,
    # 右边 5 格是这一批的位置 6..10:
    #   位置 6    [ .    .    .    .    .    .    .    -inf -inf -inf -inf]   ← 看到 0..6
    #   位置 7    [ .    .    .    .    .    .    .    .    -inf -inf -inf]   ← 看到 0..7
    #   位置 8    [ .    .    .    .    .    .    .    .    .    -inf -inf]   ← 看到 0..8
    #   位置 9    [ .    .    .    .    .    .    .    .    .    .    -inf]   ← 看到 0..9
    #   位置 10   [ .    .    .    .    .    .    .    .    .    .    .   ]   ← 看到 0..10
    # 右上角 5×4 的严格上三角全封. 少了这块, 行 6 就能看见 7..10 的草稿, 它的
    # logits 是在"已经知道答案"的条件下算出来的: 校验失效, 接受率虚高, 输出分布偏.
    #
    # 校验本身不改输出: 接受的都是大模型自己的 argmax, 第一个不吻合处直接用大模型
    # 的 token, 所以整条输出与普通贪心逐位相同. 单位置接受率 a 时, k 个草稿每次
    # 验证的期望产出是 1 + a + … + a^k 个 token (a=0.8、k=4 得 3.36).
    #
    # dtype 必须和 scores 一致: -inf 只在浮点域成立. 借错类型不报错, 只静默失效:
    # GPU 上 int64、uint32 把 -inf 铸成 0, 而 0 正是放行; int32 铸成 INT32_MIN,
    # exp 直接下溢成 0, 勉强还算屏蔽. 这个形参存在就是为了让调用方明确传 scores
    # 的类型 —— 当时错传 tokens.dtype, 就是 Bug2 的现场.
    #
    # offset 不在签名里, 三处用途必须相等:
    #   RoPE            q/k 转到 @offset .. @offset+L-1
    #   S = offset + L  决定列数与右对齐
    #   写入槽位        storage[:, :, offset:offset+L, :]
    # 一处错, 形状照样对, 结果全错. 守卫放在 Qwen3ModelWeek2.__call__ 入口, 比一次
    # offset 与 cache.offset, 之后三处都从这一个数派生.
    # L > S 不可能成立 (S = offset + L >= L); 出现了就是那个 offset 不一致.
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
    """打分对账入口: 与 mx.fast.scaled_dot_product_attention 同输入对账 (GQA 广播语义同款).

.venv/bin/python debug_repl.py
from tiny_llm.attention import scaled_dot_product_attention_grouped
mx.random.seed(0)
q = mx.random.normal((1, 16, 5, 128))
k = mx.random.normal((1, 8, 5, 128))
v = mx.random.normal((1, 8, 5, 128))
mask = mx.tril(mx.ones((5, 5)))
a = scaled_dot_product_attention_grouped(q, k, v, mask=mask)
b = mx.fast.scaled_dot_product_attention(q, k, v, scale=128**-0.5, mask=mask)
print(float(mx.abs(a - b).max()))

渲染:
7.748603820800781e-07

判据: float32 输入下差 1e-6 量级 (两边求和顺序不同); 输出还原 Q 头布局 (…,H_q,L,D);
H_q % H 必须为 0, 查询头 i 坐 kv 头 i // (H_q // H) (拼车语义见方法内全景图);
"causal" 掩码在本函数内现造, dtype 取 query 的 (Bug2 的教训, 见下方掩码段注释).
接缝: 本函数只管打分. 上游 (投影/拆头/归一化/旋转) 在 Qwen3MultiHeadAttention.__call__
①-④, 下游 (换轴/合头/wo) 在 ⑥-⑦.
"""
    # ── GQA 全景: 拼车 → 打分 → 掩码 → 加权 ──────────────────────
    #
    #   query (…,H_q,L,D)          key (…,H,S,D)      value (…,H,S,D)
    #        │                          │                   │
    #        ▼ 拆成 n_repeats 组        ▼ 插 1              ▼ 插 1
    #   (…,H,n_rep,L,D)           (…,H,1,S,D)         (…,H,1,S,D)
    #        │                          │                   │
    #        └────────┐ 广播: 1 → n_rep │                   │
    #                 ▼                 ▼                   │
    #        scores = q @ k^T  (…,H,n_rep,L,S)              │
    #                 │ + mask (广播到同形)                 │
    #                 ▼                                     │
    #        weights = softmax(scores, 最后维)  (…,H,n_rep,L,S)
    #                 │ @ value (广播: 1 → n_rep)           ▼
    #                 └──────────────────────────────► (…,H,n_rep,L,D)
    #                 ▼ 还原
    #        out (…,H_q,L,D)
    #
    #   拼车语义: Q 头 i 坐 KV 头 i // n_repeats 的车
    #   (例 H_q=4, H=2: Q0,Q1 → KV0;  Q2,Q3 → KV1)

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
    #
    #   dtype 取 query 的: 掩码要加进 scores, 而 scores 由 query 和 key 打分得到,
    #   这里两者同类型 (Qwen3 的 q/k/v 在进本函数前已被铸成 float32).
    #   借错一次不会报错, 只会静默失效: -inf 只在浮点域成立, 铸进整数域由目标
    #   类型决定 —— GPU 上 int64 / uint32 给 0, 而 0 在加性掩码里正是"放行";
    #   int32 给 INT32_MIN, 勉强还算屏蔽. 这几种结果都是合法值, 下游一路放行.
    if mask == "causal":
        mask = causal_mask(L, S, query.dtype)
        # 一行看整张表: 逐行数"允许"的列数 (值为 0 的个数).
        #   期望由形状推出, 不写死: 第 i 行放行 i + (S - L) + 1 列.
        #   L = S 时是 [1,2,3,…]; L=1, S=34 时只有一行, 值是 34.
        #   掩码全零时每行都等于 S, 而第 0 行期望放行 1 列, 第一位就不一样.
        _zeros = (mask == 0).sum(axis=1)
        _want = [i + (S - L) + 1 for i in range(min(6, L))]
        stage_once(
            3,
            "mask",
            f"{mask.dtype}  zeros/row={_zeros[: len(_want)].tolist()}  want={_want}",
        )
    elif mask is not None and mask.ndim >= 3 and mask.shape[-3] == H_q:
        # 只有 (…, H_q, L, S) 这种按 Q 头组织的掩码才需要拆头
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
