# Axpby 单 kernel 捕获证据

2026-11-17，Apple M4、macOS 27.2.1、MLX 0.32.4、gpudebug v1.0。连续与转置输入均为两个 `4096×4096` float32 数组，系数均为 1，种子为 0。各进程在输入求值与三次预热后捕获一个 Axpby，捕获外比较 `out` 与 `x+yy`，两组均正确。每份 trace 各保存一次串行重放的 profiling session。

`contiguous.capture.json` 与 `transpose.capture.json` 记录输入、环境、捕获边界、预期名称和源码 SHA256。两个 `.report.jsonl` 是重新打开 trace 并 `profile load` 后直接读取的原始输出，包含实际 kernel 名称、dispatch、timeline、shader 概览和计数器定义。`summary.json` 汇集这些字段，供对照原文使用。

每个 `.trace-manifest.json` 记录 trace 内普通文件的长度与 SHA256，以及符号链接的目标。带嵌入 profile 的 trace 本体各约 0.5 GB，留在 metadata 指定的本机临时目录；清理后可使用 `probes/week2/day1/4_axpby_capture.py` 重新捕获。trace 的 profiling 嵌入会修改包，清单在嵌入完成后生成。

四个 `transpose.buffer*.bin` 来自实际 dispatch 的内联参数：槽 5 为两个 int32 的 shape，槽 6 和 7 为两个 int64 的 x/y strides，槽 8 为一个 int32 的 ndim。按小端解码：

```sh
pdm run python - <<'PY'
from pathlib import Path
import struct
p = Path("docs/evidence/axpby-kernels")
for slot, fmt in [(5, "<2i"), (6, "<2q"), (7, "<2q"), (8, "<i")]:
    print(slot, struct.unpack(fmt, (p / f"transpose.buffer{slot}.bin").read_bytes()))
PY
```

结果依次为 `(4096,4096)`、`(4096,1)`、`(1,4096)`、`(2,)`。strides 单位为元素。命令树实际验证的名称为连续 `axpby_contiguous_float32`、转置 `axpby_general_float32`，两者网格 `16777216×1×1`、单组 `1024×1×1`，前三个资源各 64 MiB。

本次连续与转置 shader Instructions 分别为 7 与 118，Integer and Complex Limiter 为 0.00% 与 63.85%，Last Level Cache Limiter 为 31.17% 与 65.56%。Limiter 描述相应资源相对于峰值能力的压力；指令字段描述 shader 概览，完整矩阵的动态执行和内存事务还需采用各自对应的指标。

GPU time 分别为 2.2706 ms 与 3.5630 ms，比例约 1.57。这是各一次 profiling session 的重放结果，报告采样 3 / 10 个核心。它与 `docs/evidence/axpby-counters/` 的 Instruments 多次全 GPU 时间采样、普通 bench 的 CPU 墙钟分别属于不同测量。完整命令、捕获边界问题与加载步骤见 `docs/metal-counter-investigation.md`。
