# 用 Metal 计数器核对 Axpby 转置访存

## 从局部分段估算到实测流量

Axpby 的输入 x、y 为 `4096×4096` 的 float32 数组，输出计算 `x+y`。连续输入下，两块输入按缓冲下标直接读取；将 y 换成转置视图后，输出位置 `(r,c)` 分别读取 x 的 `r*4096+c` 与 y 原缓冲的 `c*4096+r`。

原估算假定一次获取以对齐的 32 字节单位进行。输出第一行的 32 个相邻线程，对连续 y 覆盖 4 个单位，对转置 y 覆盖 32 个单位，数量比为 8。处理后续输出行的线程又会读取这些单位内的其他元素：固定 c、取 `r=0..7` 时，八次读取恰好消费同一个单位中的八个 float。因此完整计算的实际传输量取决于这些访问由哪层缓存服务以及获取发生了几次。

这一假设模型中，完整 y 包含 64 MiB 有效数据，每段获取一次为 64 MiB，每个元素都重新获取整段为 512 MiB。再假定 x 读取与 out 写入各为 64 MiB，两种总量为 192 MiB 和 640 MiB。32 字节是该算式的前提；M4 的实际事务规则需要硬件依据。采集的目的，是检查实际可观测访存层级的服务量，再与整体耗时比较。

## 本次取得的定量证据

2026-11-16，在 Apple M4、macOS 27.2.1、Instruments 27.1 上，对 `probes/week2/day1/4_axpby_bandwidth.py` 进行两次录制。第一次使用 Metal System Trace 默认配置，第二次将 Metal Application 的 Counter Set 设为 Performance Limiters。下面的数据来自第二次录制的 Run 2。

脚本按连续、转置、跳行的顺序执行三组，每组预热 3 次并测量 30 次。trace 中恰好有 99 个目标进程 encoder。按创建时间分为每组 33 个，去掉各组前 3 个，再按 encoder ID 与 GPU 执行区间关联。一个 encoder 可以被记录为多个执行片段，汇总时先对片段取并集。

| 观测量 | 连续输入，30 次 | 转置输入，30 次 |
| --- | ---: | ---: |
| 每次 GPU 活跃区间长度 | 3.7059 ms | 4.0887 ms |
| Buffer L1 Miss Rate，按覆盖时间加权 | 61.53% | 66.77% |
| GPU Read Bandwidth 积分，摊到每次 | 155.70 MB | 152.98 MB |
| GPU Write Bandwidth 积分，摊到每次 | 67.63 MB | 68.06 MB |
| Last Level Cache Bandwidth 积分，摊到每次 | 250.16 MB | 1094.75 MB |
| Integer and Conditional Limiter，按时间加权 | 2.30% | 23.49% |
| Integer and Complex Limiter，按时间加权 | 2.34% | 39.28% |

两组中上述计数器的采样区间均覆盖目标 GPU 执行区间的 100%。MB 按十进制计算，`1 MB=10⁶ B`；前面模型中的 MiB 按二进制计算，`1 MiB=2²⁰ B`。

转置组的末级缓存服务量积分约为连续组的 **4.38 倍**，GPU 外部读取量积分约为连续组的 **0.98 倍**，GPU 活跃时间约为连续组的 **1.10 倍**。这次录制将变化区分到两个访存层级：转置访问增加了末级缓存服务量，GPU 外部读取量仍接近连续输入。缓存层级吸收访问分散产生的额外服务需求，与这组观测相符。具体哪一个缓存块命中、M4 的事务粒度，以及各项成本独立贡献多少，仍需更细的实验。

这些计数器覆盖整个 GPU。表格表示“目标进程 GPU 活跃期间的全 GPU 采样积分”，包括同期其他进程的流量；数据归属范围应随数值一并保留。GPU Read Bandwidth 的定义是 GPU 从其外部内存读取的带宽，描述中使用 “potentially device memory”；将其进一步等同为某个 DRAM 通道的物理字节数，需要额外依据。

第三组跳行输入的带宽计数器只覆盖其执行区间的约 24.97%，L1 miss 指标覆盖约 90.12%。汇总文件保留实际覆盖比例；本报告比较流量时选用完整覆盖的前两组。部分覆盖的数据只积分已覆盖片段。

脚本在普通运行中测得连续与转置的墙钟时间为 2602.0 与 3046.3 µs，比值为 1.171。计数器采集会改变运行条件，Run 2 的 GPU 活跃时间是另一组测量，应在同一录制内比较。Limiter 指标描述硬件管线接近峰值能力的程度；上表的整数指标变化支持“general 路径增加整数工作”的判断，独立耗时贡献仍需控制变量实验。

## 实际走通的采集路径

本机首先枚举 `MTLDevice.counterSets`，仅得到 `timestamp/GPUTimestamp`。该结果说明通过这次公共 API 查询可取得时间戳，Instruments 的 Performance Limiters 则另外提供了缓存和带宽指标。

以下命令在仓库根目录运行，将原脚本作为目标进程启动：

```sh
TRACE_DIR="$(mktemp -d "${TMPDIR%/}/axpby-counters.XXXXXX")"
PYTHON_BIN="$(pdm run python -c 'import sys; print(sys.executable)')"
xcrun xctrace record \
  --template 'Metal System Trace' \
  --time-limit 12s \
  --output "$TRACE_DIR/axpby.trace" \
  --env "PYTHONPATH=$PWD/src/extensions" \
  --target-stdout "$TRACE_DIR/bench.txt" \
  --launch -- "$PYTHON_BIN" \
  "$PWD/probes/week2/day1/4_axpby_bandwidth.py"
```

默认模板的录制元数据写着 `Counter Set: (null)`。导出的表目录虽然列有 `gpu-counter-info` 等 schema，判断采集内容时仍应检查计数器定义与数据行。默认录制仅出现 RT Unit Active 一项定义，且无法回答本次缓存问题。

打开 trace 后，选择 `View → Show Next Recording Options`，在 Metal Application 下将 `Counter Set` 从 `None` 改为 `Performance Limiters`，确认启动目标仍为 Python、参数仍指向原脚本、环境中仍有上述 PYTHONPATH，再点击 Record。目标脚本自行退出后结束录制并保存。这次 Run 2 的元数据确认 `Counter Set: Performance Limiters`，定义表中实际导出了 84 个计数器。

首次启动 Instruments GUI 时，本机提示 Install Required，要求安装系统级组件。用户完成安装授权后，界面才可进入。前台操作也由用户明确授权。命令行路径在 GUI 初始化前已能录制默认 trace。

本机尝试 `--instrument 'GPU Counters'` 时返回 `Instrument with name 'GPU Counters' cannot be found`；可用的选择入口位于 Metal Application 的录制选项。`--show-recording-options` 的 JSON 输出此次只包含 Hangs 和 Time Profiler 配置，计数集通过 GUI 选择完成。复跑应以实际菜单和录制元数据核对所选集合。

## 按表导出并关联执行区间

先导出目录，确认 run 编号、目标 PID、退出码与计数集：

```sh
xcrun xctrace export --input "$TRACE_DIR/axpby.trace" --toc \
  --output "$TRACE_DIR/toc.xml"
```

第二次录制使用 Run 2。以下导出四张表，名称对应 `4_axpby_trace_summary.py` 的四项输入：

```sh
for spec in \
  'info:gpu-counter-info' \
  'app:metal-application-encoders-list' \
  'gpu:metal-gpu-intervals' \
  'counters:metal-gpu-counter-intervals'
do
  output="${spec%%:*}"
  schema="${spec#*:}"
  xcrun xctrace export --input "$TRACE_DIR/axpby.trace" \
    --xpath "/trace-toc/run[@number=\"2\"]/data/table[@schema=\"$schema\"]" \
    --output "$TRACE_DIR/$output.xml"
done

pdm run python probes/week2/day1/4_axpby_trace_summary.py \
  --info "$TRACE_DIR/info.xml" \
  --app "$TRACE_DIR/app.xml" \
  --gpu "$TRACE_DIR/gpu.xml" \
  --counters "$TRACE_DIR/counters.xml" \
  --samples "$TRACE_DIR/overlapping-samples.csv" \
  --output "$TRACE_DIR/summary.json"
```

本次 `metal-gpu-counter-intervals` XML 为约 1.34 GB，导出耗时明显高于录制。一次同时导出多张高频表的命令在 120 秒超时后被终止，产生了不完整 XML；后续改为逐表导出，并用完整 XML 解析验证。复跑时应给高频表导出足够时间，再检查文件完整性。

汇总脚本使用标准库流式解析 XML，解析 `id/ref` 引用，保留需要的标量。xctrace 的同一个数值可能首次以带 id 的元素出现，之后用 ref 引用；直接逐行读取元素文本会遗漏这些值。

脚本验证 encoder 总数为 99、encoder ID 唯一且属于同一目标进程，每个计时 encoder 都有 GPU 执行记录。分组依赖当前脚本固定的三组顺序和每组 `3+30` 次调用；修改基准工作量后应同步修改分组规则，并重新核对 trace。

## 采样积分与指标含义

`gpu-counter-info` 给出指标名、ID、单位描述和定义。Run 2 中关键 ID 如下：

| ID | 名称 | trace 中的定义 |
| --- | --- | --- |
| 46 | Buffer L1 Miss Rate | shader 的 buffer L1 cache accesses 中 miss 的百分比 |
| 71 | GPU Read Bandwidth | GPU 从外部内存读取的 GB/s |
| 72 | GPU Write Bandwidth | GPU 向外部内存写入的 GB/s |
| 83 | Last Level Cache Bandwidth | GPU 末级缓存读写的 GB/s |
| 17 | Integer and Conditional Limiter | 整数及条件管线相对于峰值能力的限制指标 |
| 21 | Integer and Complex Limiter | 整数及复杂指令管线相对于峰值能力的限制指标 |

采样值覆盖一个时间区间，目标 encoder 的实际 GPU 执行也覆盖一个或多个区间。对采样区间与目标执行区间求交集，得到每个样本实际计入的长度 `overlap_ns`。带宽按如下公式积分：

```text
某布局的平均 MB/次
  = Σ(带宽 GB/s × overlap_ns × 10⁻⁹ s/ns × 10³ MB/GB) / 30
  = Σ(带宽值 × overlap_ns) / 10⁶ / 30
```

采样区间内使用常值近似；落在边界两侧的样本按时间交集裁剪。计数器提供的是时间采样，这项积分是按该近似获得的流量估计。

百分比指标按交集时间加权：`Σ(value×overlap_ns)/Σ(overlap_ns)`。因此表中的 L1 miss 是 miss 百分比的时间加权均值；要获得所有访问合并后的 miss/总访问比，还需要每个区间的访问计数作为权重。覆盖率为计入的区间长度除以目标执行区间总长度。

带宽积分可以直接用两段采样复核单位：100 GB/s 持续 1 ms 对应 100 MB，200 GB/s 持续 2 ms 对应 400 MB，合计为 500 MB。这里每段带宽描述传输速率，乘持续时间后才成为这段服务的数据量。比较实验时，先对每组各段求和，再除以实际调用数。

寻址也会消耗 GPU 指令处理能力。Axpby 的 general kernel 在 `elem_to_loc` 中执行整数除法、取余及 strides 的乘加；连续 kernel 直接使用 index。Integer and Conditional Limiter 与 Integer and Complex Limiter 观察对应执行管线相对于峰值能力的压力，导出定义将 100% 描述为该管线达到满容量。这个分母使 Limiter 可以提示资源是否接近上限；将时间加权值读成“整次调用中有多少时间花在整数运算”会改变指标含义。各类指令与内存服务可以重叠，独立耗时贡献需要逐 kernel 分析与控制变量。

末级缓存服务量与外部读取量观察不同接口。一块数据在末级缓存驻留时，可以被多次提供给更近的请求方，同时复用从外部取回的那份数据。其内部服务量可以增长，外部取数仍接近原量。本次全 GPU 采样确定了这两个指标的变化；具体各次请求的命中层级和 M4 的完整缓存容量需另外核对。

`overlapping-samples.csv` 保留布局、counter ID、样本起点、原样本长度、原样本值和交集长度，`summary.json` 保留执行片段与汇总。两者可独立重算积分：

```sh
pdm run python - "$TRACE_DIR" <<'PY'
import collections, csv, json, math, pathlib, sys
p = pathlib.Path(sys.argv[1])
summary = json.loads((p / "summary.json").read_text())
totals = collections.defaultdict(lambda: [0.0, 0])
with (p / "overlapping-samples.csv").open() as f:
    for row in csv.DictReader(f):
        key = row["layout"], row["counter_id"]
        ns = int(row["overlap_ns"])
        totals[key][0] += float(row["value"]) * ns
        totals[key][1] += ns
for (layout, cid), (weighted, ns) in totals.items():
    metric = summary["groups"][layout]["counters"][cid]
    assert math.isclose(weighted/ns, metric["weighted_mean"], rel_tol=1e-12)
    if "integrated_MB_per_call" in metric:
        assert math.isclose(weighted/1e6/30,
                            metric["integrated_MB_per_call"], rel_tol=1e-12)
print("CSV integration matches summary")
PY
```

## 捕获单个 Axpby 并读取 kernel 指标

系统时间线可以显示工作开始前的延迟，并在执行区间内积分全 GPU 计数器。要检查 `elem_to_loc` 带来的指令和资源压力，还需要把记录缩到一个具体 dispatch。本节采用 Apple 的 GPU 命令捕获与重放：`.gputrace` 保存命令、管线与绑定资源，`gpudebug` 重放保存的工作，取得该次重放的 shader 与硬件数据。

本次环境为 Apple M4、macOS 27.2.1、MLX 0.32.4、`/usr/bin/gpudebug` v1.0。捕获入口为 `probes/week2/day1/4_axpby_capture.py`。它使用 `4096×4096` float32 的 x 和 y、固定种子 0，系数均为 1；连续组传入 `(x,y)`，转置组传入 `(x,y.T)`，两个进程分别执行。输入创建和三次预热发生在捕获前，捕获范围包含一次求值，`x+yy` 正确性比较在捕获结束后执行。

### 准备文件并捕获

以下命令均在仓库根目录执行。先检查工具和磁盘空间，再建立独立输出目录；trace 会保存缓冲内容，本次两份带 profile 的包各约 0.5 GB。

```sh
test -x /usr/bin/gpudebug
/usr/bin/gpudebug --version
df -h .
CAPTURE_DIR="$(mktemp -d "${TMPDIR%/}/axpby-kernels.XXXXXX")"
printf '%s\n' "$CAPTURE_DIR"

for layout in contiguous transpose; do
  MTL_CAPTURE_ENABLED=1 PYTHONPATH=src/extensions pdm run python \
    probes/week2/day1/4_axpby_capture.py \
    --layout "$layout" \
    --trace "$CAPTURE_DIR/$layout.gputrace"
done
```

`MTL_CAPTURE_ENABLED=1` 在进程启动前启用 Metal 捕获；`PYTHONPATH` 使 Python 导入仓库中的扩展。脚本拒绝覆盖已有 trace 及其同名 `.capture.json`。metadata 记录输入规模、布局、预热与捕获次数、正确性结果、系统版本和源码哈希。`expected_kernel` 保存从 C++ 选路推导的名称，随后应由实际命令记录验证。

捕获边界采用：

```text
输入求值与三次预热
→ start_capture
→ synchronize
→ 构造一次 Axpby 并 eval
→ stop_capture
→ 比较结果与 x+yy
```

`start_capture` 后的 `synchronize` 有具体作用。MLX 0.32.4 在提交 command buffer 后会立即创建下一只，预热结束时已有一只在捕获前创建的空 buffer。直接将 Axpby 编进它，本机 trace 只留下后续同步命令。先在捕获内同步，可以提交旧 buffer，并让后续计算进入捕获期间新建的 buffer。最小对照中，内置加法和显式指定 stream 的扩展都复现了空捕获；加入该边界后，trace 才出现一个 encoder 和一个 dispatch。正确的数值输出、存在 trace 文件以及 profiler 的成功通知，分别验证各自阶段；实际计算覆盖需要继续核对命令树。

### 在命令树核对输入与派发

```sh
gpudebug --json --oneshot \
  -t "$CAPTURE_DIR/transpose.gputrace" --timeout 120 \
  -c 'go /commands' \
  -c 'go cb0' \
  -c 'go ce0' \
  -c 'info disp0' \
  -c 'go disp0/compute'
```

`commands` 下是 command buffer，`cb0` 中的 `ce0` 是计算 encoder，`disp0` 是一次派发。每次 `go` 会列出当前位置的子节点。本次两个 trace 各有一个计算 encoder、一个 dispatch，另有一个空的同步 command buffer。应沿实际列出的节点名导航；`cb0/ce0/disp0` 是本次命名。

连续组实际名称为 `axpby_contiguous_float32`，转置组为 `axpby_general_float32`，两者 `threadsPerGrid=16777216x1x1`、`threadsPerThreadgroup=1024x1x1`，正好覆盖 `4096²` 个输出。连续路径绑定槽 0 至 4，general 还绑定 shape、两份 strides 与 ndim。compute 子树中的 `buf[0]`、`buf[1]`、`buf[2]` 分别对应 x、y 和 out，本次均为 64 MiB。

还可以读取 general 的内联参数，直接核对真实布局。下面的 fetch 输出会给出写出的文件路径：

```sh
gpudebug --json --oneshot \
  -t "$CAPTURE_DIR/transpose.gputrace" --timeout 120 \
  -c 'fetch /commands/cb0/ce0/disp0/compute/buf[5]' \
  -c 'fetch /commands/cb0/ce0/disp0/compute/buf[6]' \
  -c 'fetch /commands/cb0/ce0/disp0/compute/buf[7]' \
  -c 'fetch /commands/cb0/ce0/disp0/compute/buf[8]'
```

slot 5 按两个小端 int32 解码，slot 6、7 按两个小端 int64，slot 8 按一个小端 int32。本次解码得到 shape `(4096,4096)`、x strides `(4096,1)`、y strides `(1,4096)`、ndim `2`。Python 中可用 `struct.unpack("<2q", Path(实际文件路径).read_bytes())` 读取 strides。这里 strides 的单位是元素，字节偏移还需乘以 float32 的 4 字节。

### 重放并保存 profile

```sh
for layout in contiguous transpose; do
  gpudebug --json --oneshot \
    -t "$CAPTURE_DIR/$layout.gputrace" --timeout 1800 \
    -c 'profile run --gpu-state default --exec serial --embed' \
    -c 'wait' \
    > "$CAPTURE_DIR/$layout.collect.jsonl" \
    2> "$CAPTURE_DIR/$layout.collect.stderr"
done
```

`profile run` 用保存的命令和资源在设备上采集；`--exec serial` 请求串行重放，`--gpu-state default` 保留默认 GPU 状态，`--embed` 将采集结果存入 trace，`wait` 等待当前操作。它会改变 trace 包的内容，因此包校验信息应在嵌入完成后生成。普通 bench 的墙钟时间仍由原脚本单独测量。

本机采集命令返回成功后，同一会话立即访问性能子树曾出现空内容。重新打开 trace 并显式 `profile load` 后，嵌入的数据可读。因此复核分为采集、加载、读取三个动作，按以下路径执行：

```sh
for layout in contiguous transpose; do
  gpudebug --json --oneshot \
    -t "$CAPTURE_DIR/$layout.gputrace" --timeout 120 \
    -c 'profile list' \
    -c 'profile load' \
    -c 'info /performance/timeline' \
    -c 'go /performance/shaders' \
    -c 'info comp0' \
    -c 'info /performance/timeline/counters/integer_and_complex/integer_and_complex_limiter' \
    -c 'info /performance/timeline/counters/integer_and_conditional/integer_and_conditional_limiter' \
    -c 'info /performance/timeline/counters/last_level_cache/last_level_cache_limiter' \
    -c 'info /performance/timeline/counters/last_level_cache/last_level_cache_utilization' \
    > "$CAPTURE_DIR/$layout.report.jsonl" \
    2> "$CAPTURE_DIR/$layout.report.stderr"
done
```

先看 timeline 中的 Commands、Encoders 和 GPU time，再看 shaders 中实际列出的节点。本次只有 `comp0` 一个 shader，Cost 为 100% 表示它占该捕获的 shader 成本；捕获本来就只含一个计算 kernel。`Instructions`、INT32、FP32、Device loads 等字段来自编译后的 shader 概览，适合对照实现形状。它们与完整矩阵的动态执行次数、实际缓存事务数分别属于不同统计对象；例如 general 的 Device loads 还可能包含 shape 和 strides 元数据加载。

计数器 `info` 同时返回 Name、Scope、Value、Description。Integer Limiter 的定义将 100% 描述为对应整数执行管线达到峰值能力；Last Level Cache Limiter 描述末级缓存尝试服务读写请求的压力，Utilization 描述实际完成服务的利用情况。指标的分母是对应资源峰值，判断时应同时读取定义、数值和工作范围。

### 本次单 kernel 结果与含义

2026-11-17 的两份捕获各重放并保存一次 profiling session，报告均为 Consistent state=yes、3 sampled / 10 total cores。请求选项为 default、serial；`profile list` 将嵌入 session 的 perf-state 显示为 medium，timeline 的 Profiled state 显示为 default。保留这两处原始字段，复跑时核对。

| 字段 | 连续 kernel | general kernel |
| --- | ---: | ---: |
| GPU time | 2.2706 ms | 3.5630 ms |
| shader Instructions | 7 | 118 |
| shader INT32 | 0 | 24 |
| shader INT16 | 0 | 23 |
| shader Branch | 0 | 4 |
| shader Device loads | 2 | 6 |
| shader Temp registers | 3 | 16 |
| Integer and Conditional Limiter | 0.55% | 37.92% |
| Integer and Complex Limiter | 0.00% | 63.85% |
| Last Level Cache Limiter | 31.17% | 65.56% |
| Last Level Cache Utilization | 17.06% | 59.37% |

这些结果将源码中的额外工作接到了具体 kernel：general 的逐维寻址对应更多整数与分支指令、更高的临时寄存器数量，并伴随整数资源压力增加；访问布局变化也伴随末级缓存压力增加。两类需求同时变化。当前结果支持继续研究这两类资源，独立延时贡献仍需固定访问地址或固定寻址实现的对照。

本次重放的 GPU time 比为 `3.5630/2.2706≈1.57`。它属于两个单次捕获在 GPU profiler 下的重放，运行环境、缓存初态和观测过程均与前一天的 30 次 Instruments 采样及普通运行不同。前一天的 1.10 是 Instruments 中的平均 GPU 活跃时间比，1.17 是普通 bench 的平均墙钟时间比；分别在自己的测量条件下解释。当前单 session 的重放尚未提供稳态变异范围。

计数器若读取 bandwidth，当前 gpudebug 的 Value 字段以 GiB / Second 显示，而 Description 文字使用 gigabytes per second。原始单位应随数值保留，跨工具换算先核对定义。本节表格采用百分比与时间，流量比较仍使用前面明确单位的 Instruments 采样积分。

### 保留证据并用于教学复核

本次可携带产物在 `docs/evidence/axpby-kernels/`：两份 capture metadata、无错误的 report 原文、汇总 `summary.json`、trace 包的逐文件清单，以及转置组槽 5 至 8 的原始二进制。trace 本体留在本机临时目录，确切路径写在 summary 中；临时文件被清理后可按上述命令重新捕获。清单分别记录普通文件的 SHA256 与符号链接的目标，本次两个 trace 各有一个包内链接，保持其原始结构。

教学复核可以先只读取连续组的 dispatch、三个 64 MiB 缓冲与七条 shader 指令，再读取转置组的 `(1,4096)` strides、118 条指令以及两项整数 Limiter。这样每个新指标都有具体代码与输入作为参照。改变 `--size` 可重新捕获其他规模；比较时同时保留输入规模、实际路径、正确性结果和原始指标。源码行成本还需要另外的采样和解码，本节验收覆盖命令、绑定资源、shader 概览与硬件指标。

## 固定布局与固定实现的三组对照

此前两份捕获同时改变了输入布局和 kernel。连续路径有七条指令，转置路径有 118 条指令，同时转置路径承担不同的内存请求。两组总时长只能描述这两项变化共同发生的结果。为判断下一步先研究哪项，2026-11-17 增加连续输入执行 general 的条件：

| 条件 | x strides | y strides | kernel |
| --- | --- | --- | --- |
| A | `(4096,1)` | `(4096,1)` | `axpby_contiguous_float32` |
| B | `(4096,1)` | `(4096,1)` | `axpby_general_float32` |
| C | `(4096,1)` | `(1,4096)` | `axpby_general_float32` |

A 与 B 保持地址布局一致，比较完整实现策略。B 与 C 保持 general 实现一致，比较 y 的布局。三组都计算 `x+y` 或 `x+y.T`，输出为行优先的 `4096×4096 float32`，grid 为 `(16777216,1,1)`，group 为 `(1024,1,1)`，系数均为 1。C 用不同 strides 解释同一份 y 缓冲，输入转置视图直接交给 kernel。

### 实验入口与测量范围

`probes/week2/day1/4_axpby_control.mm` 直接加载现有 `src/extensions/tiny_llm_ext/tiny_llm_ext.metallib`，分别选择其中已经编译的两条函数。课程实现与 MLX 自动选路保持原样。新入口让 B 能够执行 general，并让三组共用同一套主机派发代码。

三个条件共用 x、y、out 三块各 64 MiB 的 shared buffer。输入为固定公式生成、可由 float32 精确表示的有界数值，公式保存在脚本中；它与此前 Python bench 的随机输入不同，全部数值比较在本次三组条件内部进行。general 的 shape、strides、ndim 按原 C++ 实现的槽 5 至 8 绑定，捕获后又从 B、C 的资源中读回并用 `<2i`、`<2q`、`<2q`、`<i` 解码，确认实际参数与表格一致。

每组先预热三次，再测六十次。每六轮遍历 A/B/C 的全部六种排列，使每组在三个位置各出现二十次。每个计时样本只包含一次 dispatch；提交 command buffer、等待完成后，分别记录：

```text
wall_ms = 创建命令对象、编码、commit、wait 的主机墙钟
gpu_ms  = (GPUEndTime − GPUStartTime) × 1000
```

输入、输出缓冲分配与数据准备发生在计时外。B、C 比 A 多四次 `setBytes`，主机墙钟包含这些编码差异；比较设备执行时采用 GPU 时间。该入口测量直接 Metal 派发，原 Python benchmark 中的 MLX 建图、分配与调度具有另一组边界。

六种排列平衡了位置，前驱条件的次数仍有差异，缓存与运行状态也会延续。六十个样本属于一个进程中的连续执行。为检查排序是否只来自少数时段，原始样本同时按六轮区块和前驱条件汇总，结果保存在 `docs/evidence/axpby-controls/summary.json`。

### 普通派发与校验运行

Apple M4、macOS 27.2.1 下，两个普通进程分别得到以下统计。第一、第二普通进程在证据中命名为 `run1` 和 `run3`；夹在其间的 `run2-validation` 专门开启 Metal API Validation。

| 普通进程 | 条件 | GPU 中位数 | GPU 均值 | GPU P10 至 P90 | 墙钟中位数 |
| --- | --- | ---: | ---: | ---: | ---: |
| run1 | A | 2.1767 ms | 2.2396 ms | 2.0542 至 2.4455 ms | 2.4674 ms |
| run1 | B | 2.2156 ms | 2.4313 ms | 2.0607 至 3.2241 ms | 2.4927 ms |
| run1 | C | 2.7268 ms | 2.9004 ms | 2.6340 至 3.4396 ms | 3.0258 ms |
| run3 | A | 2.1459 ms | 2.2772 ms | 2.0300 至 2.8788 ms | 2.4149 ms |
| run3 | B | 2.1415 ms | 2.2291 ms | 2.0180 至 2.6541 ms | 2.4081 ms |
| run3 | C | 2.6907 ms | 2.8539 ms | 2.6078 至 3.7898 ms | 2.9799 ms |

GPU 中位数的 B/A 分别为 1.0179 和 0.9979；C/B 分别为 1.2307 和 1.2565。两个普通进程共二十个六轮区块，各区块的 C 中位数都大于 B；B/A 的区块排序则有变化。当前信号支持优先调查布局及内存访问。A 与 B 的接近说明额外实现需求在这些条件下对完成时间影响较小，仍需用相应的指令、依赖和重叠解释其机制。

`run2-validation` 的 A/B/C GPU 中位数为 2.6698、2.5894、3.0043 ms。标准错误记录了 `Metal API Validation Enabled`，运行完成。校验模式具有不同开销，单独保存为运行正确性的证据。三个进程合计保存 540 个计时样本。

全元素正确性检查发生在各组最后一次预热后，以及各组捕获派发后：先将 out 填为 NaN，再执行并逐项比较参考值。计时中的每个样本未单独做全元素校验。原始 `correct_all_elements` 字段表达这些检查通过，汇总中的 `correctness_scope` 写明了检查范围。

### 三份捕获的实际资源与重放结果

普通计时结束后，分别捕获 A、B、C 的一次 dispatch。实际命令树确认 A 使用 contiguous，B、C 使用 general，三者的 grid、group 都与表格一致；B 的 y strides 为 `(4096,1)`，C 为 `(1,4096)`。三份 trace 分别使用 `default`、`serial`、`embed` 选项重放一次，然后在新会话加载结果：

| 单次 profiler 重放字段 | A | B | C |
| --- | ---: | ---: | ---: |
| GPU time | 2.3132 ms | 2.6910 ms | 3.4527 ms |
| Instructions | 7 | 118 | 118 |
| INT32 | 0 | 24 | 24 |
| INT16 | 0 | 23 | 23 |
| Temp registers | 3 | 16 | 16 |
| Uniform registers | 16 | 28 | 28 |
| Integer and Complex Limiter | 0.00% | 85.16% | 66.06% |
| Integer and Conditional Limiter | 0.54% | 51.10% | 39.26% |
| Last Level Cache Limiter | 30.73% | 14.05% | 66.94% |
| Last Level Cache Utilization | 16.34% | 14.05% | 61.15% |
| Address Generation Limiter | 3.23% | 10.36% | 20.19% |
| sampled / total cores | 6 / 10 | 4 / 10 | 5 / 10 |

三份报告均有一个 command、一个 encoder，Consistent state 为 yes。`profile list` 的 perf-state 为 medium，timeline 的 Profiled state 为 default，两个原始字段都保留。这里的核心数描述采样覆盖，实际各时刻活跃核心还应读相应计数器。

B、C 的 shader 概览一致，C 的缓存压力与完成时间更高；这与地址布局增加内存系统工作相容。B 的整数 Limiter 高于 C，但 B 完成得更快，说明该百分比衡量的资源压力与整次完成时间具有不同分母。指令概览、资源需求及完成时间需要分别读取，再由受控变化建立联系。

profiler 中 B/A 约为 1.16，与普通派发中接近 1 的比值不同。采样覆盖、重放环境和缓存初态都需要保留：本脚本在捕获派发前由 CPU 将 out 填成 NaN，计时阶段则直接复用 out。因此，单次 profile 用于核对实际 shader 和资源线索，普通重复计时用于比较当前派发条件下的稳定时间信号。两张表一起暴露了观测条件会改变结果这一事实。

A/B 比较覆盖完整 kernel 的差异，包括指令、寄存器与内存请求的组织方式；它尚未把整数寻址的独占时间单独量出。B/C 比较已经为布局调查提供优先级，进一步优化应以匹配的普通运行检验收益，并把准备布局的成本计入完整工作负载。

### 复跑与直接读取原始报告

在仓库根目录编译并执行：

```sh
EXPERIMENT_ROOT="$(mktemp -d "${TMPDIR%/}/opencode/axpby-controls.XXXXXX")"
clang++ -O2 -fobjc-arc -std=c++17 -Wall -Wextra \
  -framework Metal -framework Foundation \
  probes/week2/day1/4_axpby_control.mm -o "$EXPERIMENT_ROOT/control"
MTL_CAPTURE_ENABLED=1 "$EXPERIMENT_ROOT/control" \
  "$PWD/src/extensions/tiny_llm_ext/tiny_llm_ext.metallib" \
  "$EXPERIMENT_ROOT/run1" 60

for CASE in A_contiguous B_general_contiguous C_general_transpose; do
  TRACE="$EXPERIMENT_ROOT/run1/$CASE.gputrace"
  gpudebug --json --oneshot -t "$TRACE" --timeout 1800 \
    -c 'profile run --gpu-state default --exec serial --embed' -c wait
  gpudebug --json --oneshot -t "$TRACE" --timeout 120 \
    -c 'profile load' \
    -c 'info /performance/timeline' \
    -c 'info /performance/shaders/comp0' \
    -c 'info /performance/timeline/counters/integer_and_complex/integer_and_complex_limiter' \
    -c 'info /performance/timeline/counters/last_level_cache/last_level_cache_limiter' \
    -c 'info /performance/timeline/counters/last_level_cache/last_level_cache_utilization'
done
```

每个运行目录都要求为新路径。再次执行程序、改用 `run3` 可取得另一次普通进程；使用 `MTL_DEBUG_LAYER=1 MTL_CAPTURE_ENABLED=1` 并改用 `run2-validation` 可单独检查 API 校验模式。普通进程与校验进程分别汇总。

本次 trace 根目录为 `/private/var/folders/fn/dkhyrt214c7_ksxbck36j49h0000gn/T/opencode/axpby-controls.Bynlt3/run1`。持久证据位于 `docs/evidence/axpby-controls/`：三个进程的全部样本、三份原始采集与读取报告、B/C 的四个参数槽二进制、源码与 metallib 哈希、统计汇总及文件校验清单。已有报告可以直接读取，无需重新进行 GPU 采集：

```sh
pdm run python -m json.tool docs/evidence/axpby-controls/summary.json
```

### 继续调查所增加的知识

本轮普通运行的 C/B 为 1.2307 和 1.2565，已为 `4096×4096` 输入的布局调查提供优先级。若课程中的下一个任务要求做 kernel 选型，可以先带着这个判断继续，并在新形状上重测。

继续研究指令与内存服务怎样共同决定完成时间，可在保持 A/B/C 定义的基础上扫描规模。若 A、B 的时间差随工作规模变化，需进一步区分启动成本、缓存驻留和执行资源的重叠。这个方向训练的是从工作量、资源能力和依赖推导时长，并用尺度变化检验预测；一次扫描先建立相关现象，机制再由后续对照检验。

若当前决定是为转置输入选择实现，更直接的候选是比较“general 直接读取”与“先整理为连续输入，再执行 contiguous”。计时应包含整理和额外存储，并改变后续复用次数，找出准备成本何时能被摊薄。这个方向训练完整成本核算，只有出现实际布局选择需求时才值得实施。

精确解释 Limiter 的内部计算公式、M4 的事务粒度与缓存替换则进入硬件资料和微基准研究。它能扩展硬件模型，对本轮选型提供的即时增量较小，且成本取决于公开资料与可观测接口。当前课程推进可以停在本轮证据，保留这些问题作为后续遇到实际瓶颈时的调查入口。

## 原始流量采样证据

仓库内的 `docs/evidence/axpby-counters/` 保存本次 `summary.json`、重叠采样 CSV、84 项计数器定义、99 个 encoder 的记录、录制目录与来源文件校验信息。大体积 trace 和完整高频 XML 保留在本机临时目录；确切路径及 SHA256 见 `source-manifest.json`。临时目录会被清理，后续机器应使用上述命令重新采集；仓库内的采样 CSV 用于复核本次积分。

这些采样回答布局变化与不同访存层级服务量的关系。三组对照另行保持地址布局或实现，以检查完成时间。若进一步研究每次 Axpby 独占的物理读取量，应减少其他 GPU 负载，增加按布局标记的执行区间，并重复录制检验变异。
