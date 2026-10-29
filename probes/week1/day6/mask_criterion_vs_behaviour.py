# 掩码 dtype 的三种取值, 对比两件事: zeros/row 判据怎么说, 屏蔽位实际拿到多少注意力.
# 不加载模型, 打分矩阵是手写的常量, 所以每个数都能手算复核.
import mlx.core as mx
from tiny_llm.attention import causal_mask

L = S = 4
# scores[i, j] = j: 越靠后的 key 原始分越高, 所以"该不该屏蔽"直接决定结果.
scores = mx.broadcast_to(mx.arange(S, dtype=mx.float32).reshape(1, S), (L, S))
want = [i + 1 for i in range(L)]

print(f"L={L} S={S}  scores[i,j]=j  期望 zeros/row={want}", flush=True)
for dt in (mx.float32, mx.int64, mx.bool_):
    m = causal_mask(L, S, dt)
    zeros = (m == 0).sum(axis=1).tolist()
    w = mx.softmax(scores + m, -1)
    # 屏蔽位 = 上三角 (j > i), 按因果结构本该拿到 0 权重
    j = mx.arange(S).reshape(1, S)
    i = mx.arange(L).reshape(L, 1)
    blocked = mx.where(j > i, w, mx.zeros_like(w))
    print(
        f"{str(dt):14s} zeros/row={zeros}  mask[0,1]={m[0, 1].item()!r:>12}"
        f"  屏蔽位权重 max={mx.max(blocked).item():.4e}",
        flush=True,
    )
