# probes/week2/day1/4_axpby_bandwidth.py
#
# 问题：axpby 的两条 kernel（contiguous / general）在下面三组不同布局上耗时多少。
#
# 三组输入。走哪条 kernel 由 axpby.cpp 里那条判据决定：
#   contiguous_kernel = (x 行连续 && y 行连续) || (x 列连续 && y 列连续)
#
#   (x, y)              两块都行连续 -> 判据为真 -> contiguous kernel
#   (x, y.T)            y 是转置视图，strides=(1,N)：行连续为假、列连续为真，
#                       与 x 的"行连续"凑不成一对 -> general kernel，warp 内
#                       32 次访存落在相距 16 KB 的地址上（分散）
#   (x[:N/2], y[::2])   y 的 strides=(2N,1)：行连续与列连续都为假 -> general
#                       kernel，但最后一维步长仍是 1，warp 内访存仍然合并
#
# 后两组都走 general，但访存位置不同；第一组走 contiguous。
# 第三组的元素数只有前两组的一半，输入形状、缓存行为、分配和派发开销也可能不同。
# 因此三组耗时差是这些条件共同变化的结果，不能直接相减来分别归因整数寻址
# 或访存分散。若要隔离其中一项，须固定工作量和其他条件，单独改动这一项。
#
# 此秒表脚本从 C++ 判据推导所选路径。配套 4_axpby_capture.py 将预热后的
# 单次求值保存为 .gputrace，可用 gpudebug 在命令树直接核对实际 kernel；
# general 的 buffer 6、7 还可导出并按 int64 解码两份 strides。
# 捕获及指标读取 SOP 见 docs/metal-counter-investigation.md。
#
# 跑法（仓库根目录）：
#   PYTHONPATH=src/extensions pdm run python probes/week2/day1/4_axpby_bandwidth.py
#
# 读法：us 是整次调用的墙钟时间；ps/elem 用输出元素数归一化，但不能消除
# 不同规模下固定开销和缓存状态的差异。跨运行的漂移需要重复测量，不能假定固定比例。

import time

import mlx.core as mx
import tiny_llm_ext as ext

N = 4096
ITERS = 30
x = mx.random.normal((N, N)).astype(mx.float32)
y = mx.random.normal((N, N)).astype(mx.float32)


def bench(xx, yy, label, iters=ITERS):
    # 预热：第一次调用要建 pipeline state、分配输出缓冲，成本不该算进稳态。
    for _ in range(3):
        mx.eval(ext.axpby(xx, yy, 1.0, 1.0))
    # mx 的算子只建图，mx.eval 才真正下发并等它跑完。计时区间必须把 eval
    # 包在里面；只调 ext.axpby 而不 eval，量的就只是建节点的时间。
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(ext.axpby(xx, yy, 1.0, 1.0))
    dt = (time.perf_counter() - t0) / iters
    # 分子按"三块数组各过一遍"算：x 读、y 读、out 写，每元素 4 字节。
    # 这是名义读写字节数，不是实测内存流量；缓存命中、事务放大不在分子里。
    gbs = xx.size * 4 * 3 / dt / 1e9
    # 每元素耗时的单位是皮秒：dt 秒 / 元素数，再乘 1e12。
    ps = dt / xx.size * 1e12
    print(f"{label:34s} {dt*1e6:9.1f} us  {gbs:6.1f} GB/s  {ps:6.1f} ps/elem")


bench(x, y, "contiguous  (x, y)")
bench(x, y.T, "general     (x, y.T)    strides=(1,N)")
bench(x[: N // 2], y[::2], "general     (x, y[::2]) strides=(2N,1)")
