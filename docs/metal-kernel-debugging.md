# Metal kernel 的读法与调试

一个 kernel 出错时，错误只有三种形态：启动失败、读到垃圾、数值算错。区分它们的成本远低于逐个怀疑。这篇给读 kernel 需要的四个词、一个最小的完整算子、以及一套能在写 Metal 之前就跑的检查。

## 一次启动里有四个词

```
thread          一个执行流，有自己的寄存器
threadgroup     一组 thread，调度到同一个 GPU 核心上
grid            本次启动全部线程组成的坐标范围
buffer(n)       第 n 个参数槽
```

`buffer(n)` 不是内存，是参数表里的位置。host 侧 `set_input_array(scales, 0)` 把指针放进第 0 槽，kernel 侧 `[[buffer(0)]]` 从第 0 槽取。两边靠数字配对，放错槽位编译器不会报错，运行时读到的是别的数组。

`dispatch_threadgroups(组数, 线程组大小)` 按各方向的组数启动完整线程组；`dispatch_threads(线程数, 线程组大小)` 指定各方向的线程总数，边缘可以使用未满线程组。使用前者时，各方向的线程总数等于组数乘组尺寸，计算覆盖范围需采用这一乘积。

拿一层真实的 prefill 为例，`M=512, N=1024, K=2048`，线程组开 `(16, 64)`：

```
grid = (ceil(512/16), ceil(2048/64)) = (32, 32) 个组
每组 16 * 64 = 1024 个线程
线程总数 = 32 * 32 * 1024 = 1,048,576 = M * K
```

线程数刚好等于输出元素数，一个线程一个输出。再看 decode 前的小形状 `M=3, K=5`，线程组仍是 `(16,16)`：

```
grid = (1, 1)，256 个线程
真正干活的 15 个，其余 241 个空转
```

所以 kernel 里的 `if (i < M && k < K)` 不是形式，是常态。

## 一个最小算子：axpby 的两半

仓库里 `axpby` 是最小的完整例子，`.metal` 与 `.cpp` 各一半。

kernel 侧（`src/extensions/src/axpby.metal:9-25`）：

```metal
template <typename T>
[[kernel]] void axpby_contiguous(
    device const T* x [[buffer(0)]],
    device const T* y [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const float& alpha [[buffer(3)]],
    constant const float& beta [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  out[index] = static_cast<T>(alpha) * x[index] + static_cast<T>(beta) * y[index];
}
```

host 侧（`src/extensions/src/axpby.cpp:150-194`）：

```cpp
auto library = d.get_library("tiny_llm_ext");
auto kernel = d.get_kernel(kname.str(), library);
auto &compute_encoder = mx::metal::get_command_encoder(s);
compute_encoder.set_compute_pipeline_state(kernel);
compute_encoder.set_input_array(x, 0);
compute_encoder.set_input_array(y, 1);
compute_encoder.set_output_array(out, 2);
compute_encoder.set_bytes(alpha_, 3);
compute_encoder.set_bytes(beta_, 4);

size_t tgp_size = std::min(nelem, kernel->maxTotalThreadsPerThreadgroup());
compute_encoder.dispatch_threads(MTL::Size(nelem, 1, 1), MTL::Size(tgp_size, 1, 1));
```

对照着看，host 侧只有五件事：取库、按名字取管线、绑管线、按槽位放参数、启动。kernel 里的 `[[buffer(n)]]` 与 host 的 `set_*` 一一对应，`[[thread_position_in_grid]]` 对应 `dispatch_threads` 里那个一维网格。

模板 kernel 的名字在实例化时定死。`axpby.metal:38-46` 用宏展开成 `axpby_contiguous_float32`、`axpby_contiguous_bfloat16` 这些名字，host 侧拼的字符串必须与它们逐字相同：

```cpp
std::ostringstream kname;
kname << "axpby_";
kname << (contiguous_kernel ? "contiguous_" : "general_");
kname << type_to_name(out);
```

## 先在 Python 里把 kernel 跑一遍

kernel 的每一次读都可以在 Python 里重放。在写 Metal 之前，用同一套下标算一遍，和稠密结果对账，能提前排除索引错误。

下面这段把 vanilla kernel 的循环原样搬进 Python。`B` 是打包后的字，`S`、`BI` 是每组一对的参数：

```python
import mlx.core as mx

mx.random.seed(0)
M, N, K = 3, 128, 5
a = mx.random.normal((M, N)).astype(mx.bfloat16)
w = mx.random.normal((K, N)).astype(mx.bfloat16)
packed, scales, biases = mx.quantize(w, group_size=128, bits=4)
mx.eval(a, packed, scales, biases)

A, B, S, BI = a.tolist(), packed.tolist(), scales.tolist(), biases.tolist()
mask = 0xF
out = [[0.0] * K for _ in range(M)]
for i in range(M):
    for k in range(K):
        s = 0.0
        for g in range(N // 128):
            sc, bi = S[k][g], BI[k][g]
            for word_idx in range(16):
                word = B[k][g * 16 + word_idx]
                for off in range(8):
                    q = (word >> (4 * off)) & mask
                    s += (q * sc + bi) * A[i][g * 128 + word_idx * 8 + off]
        out[i][k] = s

dense = mx.dequantize(packed, scales, biases, 128, 4).astype(mx.float32)
ref = a.astype(mx.float32) @ dense.T
mx.eval(ref)
print("max|Δ|", mx.abs(mx.array(out) - ref).max().item())
```

输出 `max|Δ| 0.029`，数值量级 15 左右，这是 float32 累加与 bf16 输出的差，索引是对的。换成别的公式再跑，差会立刻变大一个数量级，一眼就能看出来。

这段的价值在于：它把「下标算错」和「GPU 执行错」分开。Python 版本对了，Metal 若还错，问题在启动参数或槽位；Python 版本就错，说明公式理解错了，不用去碰 GPU。

## 上机之后怎么对账

固定种子，先算期望值，再调算子，两步都 `mx.eval`：

```python
mine = quantized_matmul(scales, biases, 128, 4, a, packed, True)
expected = a @ mx.dequantize(packed, scales, biases, 128, 4).astype(mx.bfloat16).T
mx.eval(mine, expected)
print(mx.abs(mine - expected).max().item())
```

形状要覆盖三类：正好填满线程组的、留尾巴的、极小的。

```
M = 1, 3, 8, 9        K = 5, 64, 65
N 固定 128 的倍数
```

`K=65` 与 `M=9` 会分别踩到最后一列的越界与最后一行，这两个数比随机大矩阵更能暴露边界问题。

## 把中间量拿出来

数值错但形状对时，下一步是把 kernel 内部的几个值写到一个额外缓冲区。Metal 里加一个槽：

```metal
[[kernel]] void quantized_matmul_vanilla_debug(
    /* ... 原有的 buffer(0..7) ... */
    device float* dbg [[buffer(8)]]) {
    if (i < M && k < K && i == 0 && k == 0) {
        dbg[0] = static_cast<float>(b_val_packed);
        dbg[1] = scale;
        dbg[2] = static_cast<float>(a[a_loc]);
        dbg[3] = sum;
    }
}
```

只让一个线程写，否则几千个线程会覆盖同一格。host 侧分配 `mx::array dbg({4}, mx::float32)`，`set_output_array(dbg, 8)`，并且把它加进 `outputs`，这样 `mx.eval` 之后能读到。这一步需要 `output_shapes` 返回两个形状，调试版可以临时改，正式版删掉。

更省事的替代是把 `sum` 在循环中途分段写进 `out`，跑一次看前几个组的部分和是否与 Python 模拟一致。分段比单点更能定位到哪个 group 开始偏。

## 数值正确后检查实际执行成本

Axpby 的连续与 general 路径可以计算相同规模的正确结果，但 general 还要执行逐维地址换算。检查这类成本时，先捕获一个预热后的实际 dispatch，再由 GPU profiler 重放。`probes/week2/day1/4_axpby_capture.py` 在捕获前准备输入并预热三次，捕获中求值一次，结束后比较 `out` 与 `x+yy`：

```sh
CAPTURE_DIR="$(mktemp -d "${TMPDIR%/}/axpby-kernels.XXXXXX")"
MTL_CAPTURE_ENABLED=1 PYTHONPATH=src/extensions pdm run python \
  probes/week2/day1/4_axpby_capture.py --layout transpose \
  --trace "$CAPTURE_DIR/transpose.gputrace"
gpudebug --json --oneshot -t "$CAPTURE_DIR/transpose.gputrace" --timeout 1800 \
  -c 'profile run --gpu-state default --exec serial --embed' -c 'wait'
gpudebug --json --oneshot -t "$CAPTURE_DIR/transpose.gputrace" --timeout 120 \
  -c 'profile load' \
  -c 'go /commands/cb0/ce0' \
  -c 'info disp0' \
  -c 'info /performance/shaders/comp0' \
  -c 'info /performance/timeline/counters/integer_and_complex/integer_and_complex_limiter'
```

先核对命令树实际列出的 kernel 名称与 dispatch，再读取 shader 指令和 Limiter。前者显示该实现编译后的指令形状，后者描述某个执行资源相对于峰值能力的压力。整数 Limiter 的 100% 对应整数相关管线满容量，百分比的分母是管线能力。资源压力需要结合缓存、其他执行管线与依赖共同解释，重放的时间也按重放条件读取。

本机 MLX 0.32.4 在捕获开始后需要先同步以切换到捕获内创建的 command buffer；入口已经执行这一步。省略时曾得到数值正确但没有计算 encoder 的 trace。另一次采集成功后，重新打开并 `profile load` 才读到嵌入的数据。这两个检查分别保护捕获覆盖与 profile 加载。完整 SOP、实际指令与硬件结果保存在 `docs/metal-counter-investigation.md`，原始小体积证据位于 `docs/evidence/axpby-kernels/`。

2026-11-17 的实测在 Apple M4、macOS 27.2.1、MLX 0.32.4、gpudebug v1.0 上执行，两份 trace 各捕获一个 `4096×4096` float32 Axpby，各保存一次 profile：

| 检查项 | 连续路径 | general 转置路径 |
| --- | ---: | ---: |
| 实际 dispatch 名称 | axpby_contiguous_float32 | axpby_general_float32 |
| GPU time | 2.2706 ms | 3.5630 ms |
| shader Instructions | 7 | 118 |
| shader INT32 | 0 | 24 |
| shader Temp registers | 3 | 16 |
| Integer and Complex Limiter | 0.00% | 63.85% |
| Last Level Cache Limiter | 31.17% | 65.56% |

general 在读取 x、y 前分别调用 `elem_to_loc`，需要整除、取余和 strides 乘加；指令数量与整数压力的增加对应这一实现差异。更分散的 y 访问也对应更高的末级缓存压力。指令概览反映编译实现，Limiter 反映资源压力；独立耗时贡献还需要固定其他条件的实验。

末级缓存服务多次提供驻留的数据时，外部内存可以只取回一次。前一天另一次 Instruments 录制测得末级缓存服务量约 4.38 倍、外部读取约 0.98 倍、GPU 平均活跃时间约 1.10 倍；服务量来自带宽采样乘目标执行时长的积分，覆盖全 GPU。普通 bench 另测端到端时间比约 1.17。本节单次重放比约 1.57，按重放条件解释，各次测量的时间范围与运行环境分别保留。

## 失败模式

```
pipeline 创建失败   名字拼错, 或模板实例化没写, 或 Metal 源码编译不过
读到垃圾或 GPU 报错 槽位错, 网格覆盖不全, 下标越界, 少 check 了边界
数值有限但不对      解包公式, 累加顺序, scale 用错 group, 写回位置错
```

第一种发生在 `get_kernel` 之后；第二种往往在 `mx.eval` 时崩或读数明显异常；第三种形状、dtype 全对，只有数值错。三者需要的排查手段不同，先分清再动手。

## 本项目 Task 3 的检查顺序

```
1  Python 模拟对上稠密结果        排除公式与下标
2  kernel 只在 (M,K)=(1,1) 跑通   排除网格与边界
3  跑 (3,5) 与 (9,65)             排除尾巴
4  跑真实形状 (512,2048)          排除线程组尺寸与寄存器压力
5  跑 task_3 测试                 与 mx.quantized_matmul 全量对账
```

每一步只改一个变量，错了就退回上一步。第 4 步出问题而前三步正常时，通常是线程组尺寸超过了 `maxTotalThreadsPerThreadgroup()`，或者每线程的寄存器数让占用率骤降。
