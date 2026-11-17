"""同一个权重矩阵, 改变激活的行数 M, 看吞吐怎么从带宽限制爬到算力限制."""

import time

import mlx.core as mx

E = 4096
W = mx.random.normal((E, E)).astype(mx.bfloat16)
mx.eval(W)
print(f"权重 {W.nbytes / 2**20:.0f} MiB, 元素 {E * E:,}")
print("M = 同时处理的激活行数, 权重矩阵始终只有一份")
print("  decode         M = 1")
print("  prefill        M = 序列长度")
print("  批量 decode    M = 批大小")


def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


# 先量这台机器的两个常数
a = mx.random.normal((64 * 1024 * 1024,)).astype(mx.float32)
b = mx.random.normal((64 * 1024 * 1024,)).astype(mx.float32)
mx.eval(a, b)
BW = 3 * a.nbytes / bench(lambda: a + b)  # a + b 一次运算读三块内存, bandwidth，带宽

big = mx.random.normal((2048, E)).astype(mx.bfloat16)
mx.eval(big)
PEAK = (
    2 * 2048 * E * E / bench(lambda: big @ W, warmup=3, iters=10)
)  # ：2 × 2048 × 4096 × 4096 是 big @ W 的浮点运算次数，除以秒数，就是峰值。

print(f"带宽 {BW / 1e9:.1f} GB/s (两个 256 MiB 数组相加, 读 512 写 256)")
print(f"峰值 {PEAK / 1e12:.2f} TFLOPS (M=2048)")
print(f"比值 {PEAK / BW:.1f} FLOP/字节")
print()

rows = []
for M in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
    x = mx.random.normal((M, E)).astype(mx.bfloat16)
    mx.eval(x)
    rows.append((M, bench(lambda: x @ W)))

mem_ms = W.nbytes / BW * 1e3
print(f"{'M':>4} {'mem_ms':>8} {'cmp_ms':>8} {'real_ms':>8} {'GFLOPS':>8}  谁主导")
for M, t in rows:
    flops = 2 * M * E * E  # 这一次矩阵乘要做多少次浮点运算
    cmp_ms = flops / PEAK * 1e3  # 这些运算按峰值算力要多少毫秒
    real_ms = t * 1e3  # bench 测到的中位耗时, 秒换成毫秒
    gflops = flops / t / 1e9  # 实测吞吐, 每秒多少十亿次浮点运算
    side = "搬运" if cmp_ms < mem_ms else "计算"
    print(
        f"{M:>4} {mem_ms:>8.3f} {cmp_ms:>8.3f} {real_ms:>8.3f} {gflops:>8.1f}  {side}"
    )


# | 列 | 变量 | 来源 |
# |---|---|---|
# | mem_ms | `W.nbytes / BW * 1e3` | 权重字节数 ÷ 带宽，换成毫秒。**与 M 无关**，所以每行一样 |
# | cmp_ms | `2*M*E*E / PEAK * 1e3` | 运算次数 ÷ 峰值算力 |
# | real_ms | `t * 1e3` | bench 实测 |
# | GFLOPS | `flops / t / 1e9` | 运算次数 ÷ 实测耗时 |
# | 谁主导 | `cmp_ms < mem_ms` | 计算下界比搬运下界小，就是搬运主导 |

print(f"两个下界相交于 M = {mem_ms / (2 * E * E / PEAK * 1e3):.0f}")
print()

t1, t_big = rows[0][1], rows[-1][1]
g1, g_big = 2 * 1 * E * E / 1e9, 2 * 512 * E * E / 1e9
print(f"同一份 {W.nbytes / 2**20:.0f} MiB 权重:")
print(f"  M=1    {t1 * 1e3:6.3f} ms   {g1:5.3f} GFLOP")
print(f"  M=512  {t_big * 1e3:6.3f} ms   {g_big:5.2f} GFLOP")
print(f"  时间涨 {t_big / t1:.1f} 倍, 运算量涨 512 倍")
print(f"  每多一倍运算, 只多花 {(t_big / t1 - 1) / 511 * 100:.2f}% 的时间")
