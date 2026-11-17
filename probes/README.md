# probes

观测脚本，按 `week<N>/day<M>/` 放。每个脚本只回答一个问题，打印可复核的数值，不做断言、不改被观测的对象。

跑法（在仓库根目录）：

```bash
PYTHONPATH=$PWD:$PWD/src .venv/bin/python probes/week1/day6/zero_mask_two_hop.py
```

`$PWD` 要进 `PYTHONPATH`，因为脚本从根目录的 `model_names.py` 取模型短名；`$PWD/src` 供 `tiny_llm`。脚本放在子目录里，Python 只把脚本自己所在的目录加进 `sys.path`，所以这两段都得给。

命名按"它回答什么"，不按发现顺序。新脚本进对应的 `week/day` 目录，并在下面的表里加一行。

## week1/day6

流式解码器：

| 脚本 | 回答的问题 |
| :--- | :--- |
| `detokenizer_minimal_model.py` | 纯 Python 复刻的 `last_segment`：连读两次，第二次拿到什么 |
| `detokenizer_consuming_read.py` | 真货 `tokenizer.detokenizer` 上同一个问题 |
| `prompt_and_decode_snapshot.py` | prompt 渲染结果、token 数、top-8、8 步贪心、解码器逐步输出 |

## 因果掩码的 dtype

| 脚本 | 回答的问题 |
| :--- | :--- |
| `mask_dtype_values.py` | `causal_mask` 在 int32/int64/uint32 下的实际内容，以及 `scores + mask` 的 dtype |
| `input_dtype_propagation.py` | 输入 dtype 换三种时的 top5；`argmax` 的返回类型；`concatenate` 的提升规则 |
| `mask_criterion_vs_behaviour.py` | `zeros/row` 判据与屏蔽位实际权重在 float32/int64/bool 下各是多少（不加载模型，数可手算） |
| `mask_forced_float32.py` | 把 `causal_mask` 猴补成恒用 float32，输入 dtype 还影响结果吗 |
| `activation_delta_by_stage.py` | int32 与 int64 输入逐级比对激活值 max\|Δ\|，定位差异出现在哪一级 |
| `zero_mask_two_hop.py` | 两边都 float32、只换掩码内容，逐层比对末尾位置与前序位置 |

## week2/day1

| 脚本 | 回答的问题 |
| :--- | :--- |
| `5_replay_profiler_lab.py` | 重放式 profiler 的四组中位数之和、各组占比何时与真实执行相符，失真来自哪一项；配套的 `5_replay_profiler_lab.md` 记录原始输出与结论 |
| `4_axpby_bandwidth.py` | axpby 在三种布局下的耗时与名义带宽；这些对照不能单独归因整数寻址或访存分散 |
| `4_axpby_trace_summary.py` | 将 Instruments 导出的计数器区间与三组各 30 次 GPU 执行关联，积分带宽并报告覆盖率；采集、导出和复核见 `docs/metal-counter-investigation.md` |
| `4_axpby_capture.py` | 各捕获一个预热后的连续或转置 Axpby；核对真实 kernel、绑定 strides，再用 gpudebug 读取指令与资源指标 |

第四组材料区分三个计时范围：`4_axpby_bandwidth.py` 用 CPU 秒表测完整调用，`4_axpby_trace_summary.py` 将 Instruments 的带宽速率乘目标执行时长得到服务量，`4_axpby_capture.py` 保存一个具体 dispatch 供 GPU profiler 重放。后者显示单 kernel 的指令和资源压力，重放时间按自身条件解释。

捕获命令在仓库根目录运行，输出应使用新路径：

```sh
CAPTURE_DIR="$(mktemp -d "${TMPDIR%/}/axpby-kernels.XXXXXX")"
MTL_CAPTURE_ENABLED=1 PYTHONPATH=src/extensions pdm run python \
  probes/week2/day1/4_axpby_capture.py --layout transpose \
  --trace "$CAPTURE_DIR/transpose.gputrace"
```

该入口在捕获前求值输入并预热三次，捕获内先同步以建立 command buffer 边界，再执行一次 Axpby，结束后验证结果。伴随的 `.capture.json` 记录规模、环境、源码哈希与预期路径；实际路径需要在 trace 中核对。`docs/metal-counter-investigation.md` 就地解释捕获、命令树、profile 的采集与加载、指标定义和本次结果，作为完整 SOP。

### Metal 派发探针

`.mm` 文件是 Objective-C++，不经过 Python。跑法是把源码直接编译成可执行文件，在仓库根目录：

```bash
clang++ -fobjc-arc -std=c++17 -framework Metal -framework Foundation \
  probes/week2/day1/<name>.mm -o /tmp/<name>
/tmp/<name>
```

| 脚本 | 回答的问题 |
| :--- | :--- |
| `0_mtl_limits.mm` | 一个线程组最多多少线程、一条指令推进多少线程；这两个数与 `maxTotalThreadsPerThreadgroup()` 的关系 |
| `1_mtl_dispatch_probe.mm` | `dispatchThreads` 的第一个参数是线程数还是组数；末组不满、组大于线程总数、组超过硬件上限三种边界各是什么行为 |
| `2_mtl_grid_fold.mm` | 把三维线程坐标映射到一维缓冲下标；二维线程组为何未必对应连续下标 |
| `3_mtl_submit_batching.mm` | 把 K 个 kernel 装进一次提交与逐个提交，每个 kernel 的固定开销差多少 |
