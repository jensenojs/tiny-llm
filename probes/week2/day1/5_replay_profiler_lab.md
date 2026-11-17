# 重放式 profiler 的可信区间

`benches/profile_week2_kernels.py` 不执行模型。它按一步 decode 的形状，把四组算子重建在合成输入上，每组一次 `mx.eval`，`perf_counter` 包在 eval 外面，取中位数，再按组求和算占比。这个脚本在同一进程里把四个偏差源分别打开，量出各自的大小，再用尺度扫描定出比例可信的边界。

结论的形状：重放总和约等于真实一步加上每次 `mx.eval` 的固定开销。固定开销远小于真实步时长时，总和与比例都可用；固定开销把某组压住时，该组的份额在任何尺度上都不可用。

## 被测模型

2 层、E=64、H=4、G=2、D=16、I=128、词表 512，bf16 随机权重（种子 20261109），139648 个参数，272.8 KiB。forward 是 embedding → 每层（rms_norm → q/k/v 投影 → q/k norm → rope → 拼接 cache → `mx.fast.scaled_dot_product_attention` → wo → 残差 → rms_norm → gate/up → silu → down → 残差）→ final norm → lm_head（只算最后一行）。prefill 送 8 个 token 填满 cache，decode 送 1 个 token 读这 8 个位置，attention 的 context 是 9。

`KernelReplay` 照 bench 的结构写：读模型形状，造随机输入，`mx.eval` 一次把输入固定住；四个 builder 分别重放 projections、attention、pointwise、cache，算子顺序与真实 forward 一致，其中 projections 组把 `gate + up` 送进 `w_down`（与 bench 相同），silu 归入 pointwise。`benchmark_groups` 用旋转顺序、warmup 后逐轮 `perf_counter` 包住 `mx.eval`、最后取中位数。

## 跑法

```bash
PYTHONPATH=$PWD:$PWD/src .venv/bin/python probes/week2/day1/5_replay_profiler_lab.py
```

只依赖 mlx，不读任何仓库文件，整块跑完约 7 秒。

## 原始输出

```
mlx=0.32.0 dtype=bfloat16 device=Device(gpu, 0) one-eval floor=112.83us process-first-eval=33765.75us
model layers=2 E=64 H=4 G=2 D=16 I=128 vocab=512
parameters=139648 weights=272.8 KiB

[1] rotated replay, one mx.eval per builder (warmup=5, iterations=101)
group        median_us  share   op sites  site share  median/floor
------------------------------------------------------------------
projections  282.46     32.9%   15        42.9%       2.50x
attention    209.46     24.4%   2         5.7%        1.86x
pointwise    215.29     25.1%   16        45.7%       1.91x
cache        151.92     17.7%   2         5.7%        1.35x
sum          859.12     100.0%  35        100.0%      7.61x

[2] same groups, build() only, no mx.eval
group        build_us  build share  eval_us  eval share  eval/build
-------------------------------------------------------------------
projections  9.87      42.7%        282.46   32.9%       28.60x
attention    3.37      14.6%        209.46   24.4%       62.06x
pointwise    5.96      25.8%        215.29   25.1%       36.13x
cache        3.92      16.9%        151.92   17.7%       38.78x
sum          23.12     100.0%       859.12   100.0%      37.15x

[3b] median_us by execution position within the round (rotated run)
group        pos 0  pos 1  pos 2  pos 3  max/min
------------------------------------------------
projections  278.5  321.4  279.4  321.4  1.15x
attention    210.8  209.3  220.5  209.0  1.06x
pointwise    212.7  227.1  232.1  206.2  1.13x
cache        134.8  151.9  167.6  150.3  1.24x

[3] fixed order (projections first) vs rotated, against run-to-run drift
group        rotated_us  rotated repeat_us  repeat/rotated  fixed_us  fixed/rotated
-----------------------------------------------------------------------------------
projections  282.46      289.25             1.02x           278.25    0.99x
attention    209.46      216.08             1.03x           208.21    0.99x
pointwise    215.29      223.17             1.04x           217.67    1.01x
cache        151.92      158.37             1.04x           162.46    1.07x
sum          859.12      886.87             1.03x           866.58    1.01x

[4] mean vs median, rotated run's samples
group        median_us  mean_us  mean/median  max_us   max/median  mean_wo_max
------------------------------------------------------------------------------
projections  282.46     320.84   1.14x        839.62   2.97x       315.66
attention    209.46     221.19   1.06x        595.62   2.84x       217.45
pointwise    215.29     225.53   1.05x        426.29   1.98x       223.52
cache        151.92     172.23   1.13x        1000.08  6.58x       163.95
sum          859.12     939.80   1.09x

[5] real decode step vs replay sum
quantity               us       vs real median
----------------------------------------------
real step, median      342.54   1.00x
real step, mean        411.37   1.20x
replay sum of medians  859.12   2.51x
replay sum of means    939.80   2.74x
replay build-only sum  23.12    0.07x
4 x one-eval floor     451.33   1.32x

[6] real forward, sync after every stage (35 syncs/step)
group        staged_us  staged share  replay_us  replay share  staged/replay
----------------------------------------------------------------------------
projections  1749.42    42.4%         282.46     32.9%         6.19x
attention    329.00     8.0%          209.46     24.4%         1.57x
pointwise    1833.17    44.4%         215.29     25.1%         8.51x
cache        216.33     5.2%          151.92     17.7%         1.42x
sum          4127.91    100.0%        859.12     100.0%        4.80x
staged total / real single-eval median = 12.05x; mean cost per stage = 117.94us vs one-eval floor 112.83us (1.05x)

[7] prefill (L=8) replay vs real single-eval step
group        replay_us  replay share
------------------------------------
projections  290.08     39.2%
attention    223.92     30.3%
pointwise    226.17     30.6%
sum          740.17     100.0%
prefill real median = 346.54 us, replay sum / real = 2.14x

[8] same harness across hidden sizes (decode, warmup=3, iterations=31)
E     MiB     replay_us  real_us  replay/real  4xfloor/real  proj   attn   point  cache
---------------------------------------------------------------------------------------
64    0.27    739.21     334.83   2.21x        134.8%        34.7%  22.2%  26.1%  17.0%
256   2.75    716.21     330.54   2.17x        136.5%        37.2%  17.5%  26.0%  19.2%
1024  38.01   1166.42    789.33   1.48x        57.2%         61.3%  11.4%  16.1%  11.2%
2048  148.02  2642.88    2245.58  1.18x        20.1%         78.3%  8.2%   7.6%   5.9%
```

## 每次 eval 的固定开销是失真项

本机上一次 `mx.eval`（图里只有一个 64 元素加法）的中位数是 112.83 µs。进程里第一次 eval 是 33765.75 µs，是真实 decode 步 342.54 µs 的 98.6 倍，warmup 之后不再出现。

四组中位数之和 859.12 µs，同一个 decode 步单次 eval 的墙钟 342.54 µs，前者是后者的 2.51 倍。差值落在固定开销上：`342.54 + 4 × 112.83 = 793.87`，实测比这个和多 8.2%。尺度扫描的四行与 prefill 行给出同一条关系，残差落在 −8.4% 到 +8.2% 之间。

按组看，cache 只有两个 concat，测到 151.92 µs，是固定开销的 1.35 倍；attention 两个 sdpa，209.46 µs，1.86 倍。这两组在总和中占 42.1%，其中 62.4% 是两次 eval 的往返。projections 有 15 个算子，282.46 µs，固定开销占它自报时间的 40%。

## 不计 eval 时，测到的是派发次数

`build()` 不触发执行，只把算子挂进图。四组 build 中位数之和 23.12 µs，是真实步的 0.07 倍，比 eval 后的总和少 37.15 倍。此时占比与各组的算子个数逐项对齐：projections 42.7% 对 42.9%，attention 14.6% 对 14.3%，pointwise 25.8% 对 26.0%，cache 16.9% 对 16.9%，最大偏差 0.3 个百分点。这样得到的数字描述的是「这一步派发了几个算子」，与算子做什么无关：15 个矩阵乘和 16 个归一化、rope、激活在同一个量级上。计时器在这里得到的是 Python 侧把算子写进图的耗时，与 GPU 上要算多少无关。

## 轮转与固定顺序的差别落在漂移之下

固定顺序把 projections 放在第一位，总和 866.58 µs，比旋转版的 859.12 µs 高 1.01 倍；逐组看，projections 0.99×、attention 0.99×、pointwise 1.01×、cache 1.07×。同一次运行里再跑一遍旋转版，总和比是 1.03×，逐组 1.02×–1.04×。组在各个位置上的中位数最大差 1.24×（cache），最小 1.06×（attention）。

五次独立运行里，固定/旋转的总和比是 0.97、0.97、1.01、1.11、1.37，同进程重复一次旋转是 0.97、0.98、1.00、1.02、1.21。两组分布重叠，固定顺序的偏差没有超过运行间的漂移，也没有稳定符号；位置上最大差落在哪一组同样逐次变化，取值范围 1.02×–1.24×。第一条本来要防的那笔成本是 33.8 ms 的进程首次 eval，warmup 的 5 轮已经吃掉；轮转只能把「第一个」的位置在组间轮流，替不了 warmup。warmup 为 0 时这笔成本全部记在测得的第一组上，而中位数要 101 个样本才把它稀释到看不见。

## 均值带着尾部的偏移

同批样本里，均值比中位数高 1.05×–1.14×，总和 1.09×；max/median 从 1.98×（pointwise）到 6.58×（cache）。去掉每组最大的一个样本后，均值仍比中位数高 3.8%–11.8%，抬高的来源是一批几百微秒的样本，不是一次毫秒级抖动。均值把这个偏移带进占比，中位数不带。用均值算份额，份额反映的是这一步被系统抖动打断了多久。

## 真实执行的组间比例没有被独立观测到

把真实 forward 切成 35 个阶段、每个阶段同步一次，得到一份「真实执行」的组间归属：总和 4127.91 µs，是真实步的 12.05 倍；每阶段平均 117.94 µs，与一次 eval 的固定开销 112.83 µs 相当（1.05×）。这份数据测到的仍是 eval 往返，不是各组的工作量，占比里 pointwise 44.4% 只是因为它有 16 个阶段、projections 42.4% 因为它有 15 个。在 272.8 KiB 的模型上，单段的工作量无法从往返里分出来，组间比例在原理上不可观测。

可观测的量只有两个：整步的单次 eval 墙钟，以及重放给出的各组中位数。重放法用四次固定开销换来了可分解，代价是它给出的比例成为「各组工作量 + 四份固定开销」的混合物，而真实比例不可独立校验。

## 尺度扫描

同一个 harness 换隐藏维度重跑，权重从 0.27 MiB 到 148.02 MiB。这一节用 warmup=3、31 次迭代、每个尺寸新建模型，与主表（warmup=5、101 次迭代）的绝对值和前面几节不能逐位对齐；同一进程内各尺寸之间的趋势才是它的结论。

| E | 权重 | 重放总和 | 真实步 | 比值 | 4×floor/真实步 | projections | attention | pointwise | cache |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 64 | 0.27 MiB | 739.21 µs | 334.83 µs | 2.21× | 134.8% | 34.7% | 22.2% | 26.1% | 17.0% |
| 256 | 2.75 MiB | 716.21 µs | 330.54 µs | 2.17× | 136.5% | 37.2% | 17.5% | 26.0% | 19.2% |
| 1024 | 38.01 MiB | 1166.42 µs | 789.33 µs | 1.48× | 57.2% | 61.3% | 11.4% | 16.1% | 11.2% |
| 2048 | 148.02 MiB | 2642.88 µs | 2245.58 µs | 1.18× | 20.1% | 78.3% | 8.2% | 7.6% | 5.9% |

真实步从 334.83 µs 涨到 2245.58 µs，涨了 6.71 倍；重放总和从 739.21 µs 涨到 2642.88 µs，涨了 3.58 倍。差在四份固定开销上：它不随尺寸变化，权重越大，它在比值里的权重越小。E=2048 时 projections 的份额从 34.7% 走到 78.3%，其余三组从 65.3% 压到 21.7%。这个方向对应 decode 一步的权重搬运：一个 token 要把全部权重读一遍，权重几乎都在 projections 里。

轻组的份额仍然不可信。E=2048 时 cache 的 5.9% 落在 155.9 µs（2642.88 × 5.9%），而固定开销是 112.83 µs，这一组自报的时间里有 72% 是 eval 往返。attention 的 8.2% 落在 216.7 µs，同样有 52% 是往返。

## 结论

总和的可信条件可以写成一个等式：`重放总和 ≈ 真实一步 + builder 数 × 单次 eval 往返`。本机单次往返 112.83 µs，decode 四个 builder 合计 451.33 µs。真实一步 342.54 µs 时（E=64），这笔开销是真实步的 1.32 倍，重放/真实 = 2.51；真实一步 2245.58 µs 时（E=2048），同一笔开销降到 20.1%，比值降到 1.18。判据因此可以直接测：四份开销占真实一步的比例落到百分之几，总和可用；停在几十个百分点，总和只剩「模型规模」这一条信息。

各组份额另有条件，看的是该组自己的工作量。projections 有 15 个算子，E=2048 时固定开销只占 5.5%，份额可用。cache 只有两个 concat，在各尺寸上都是固定开销的 1.1–1.4 倍，份额不可用。用比例做决策时，projections 的上升可以信，attention、pointwise、cache 之间的此消彼长不能信。

还有一处失真与测量开销无关，来自 builder 与真实 forward 的对应关系。projections 组把 `gate + up` 送进 `w_down`，真实路径送的是 `silu(gate) * up`。两者形状相同；代价由算术量决定时，这个替换不改结果，代价由算子个数决定时，silu 记在 pointwise、`+` 记在 projections，两组份额跟着平移。哪一侧主导，用 build-only 的占比与 eval 后的占比对照即可判断：本机两者相差 0.3 个百分点以内，说明此刻是算子个数在决定代价。
