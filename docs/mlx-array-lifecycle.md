# 一个 array 从构造到可读

`mx.array` 是 MLX 的基本数据类型，和 numpy 的 `ndarray` 属于同一类东西：一块按某种形状排列的数值，加一个元素类型。numpy 的数组一旦存在就有数值；MLX 的数组可以在没有数值的情况下存在。`d = mx.matmul(b, b)` 返回时，`d.shape` 与 `d.dtype` 都已确定，显存里一个字节都没有。

同一个对象因此有两种身份：有数值时它是一块数据，没有数值时它是图上的一个待算节点，记着「将来按什么算」和「从哪些数里算」。

「求值模型」指的就是这两件事之间的规则：数值什么时候产生、由谁产生、放在哪里、什么时候可以读。本文从一行 Python 开始，一路跟到 GPU 写回结果，中间经过的每个对象都在它必须出现的地方介绍。

读者预设：知道 `mx.array` 传进 C++ 时不拷贝数据，知道 `(m,k) @ (k,n)` 的形状规则，Task 1 里写过 W4 的解包。

标「推断」的地方，是头文件里只有声明、实现体在 `.so` 里，只能从接口与可观测行为反推。

## 构造完成时手里是什么

```python
import mlx.core as mx
MiB = 2 ** 20
mx.clear_cache()

a = mx.zeros((4096, 4096), dtype=mx.float32)
b = a + 1.0
d = mx.matmul(b, b)
print(d.shape, mx.get_active_memory() / MiB)   # (4096, 4096) 0.0

mx.eval(d)
print(mx.get_active_memory() / MiB)            # 192.0
```

三次调用做完，`d` 的形状与 dtype 已经确定，显存一块没分配；`mx.eval(d)` 之后计数跳到 192。192 是三块 4096×4096 的 float32：`a` 与 `b` 各 64 MiB，`d` 自己的结果 64 MiB。`a`、`b` 之所以还在，是因为 Python 侧的名字仍引用着它们，求值完成不会把它们丢掉。

构造完成时的数组已经知道自己的形状，还不知道自己的数值。要把它算出来，节点得记住两件形状回答不了的事。

## 节点记住的两件事

第一件是怎么算，它需要一个能执行的对象。`ArrayDesc` 的 `primitive` 字段装这个对象的指针（`include/mlx/array.h:464`）。

第二件是从哪些数里算，`inputs` 字段装上上游数组（`array.h:488`）。它是一串 `array`，每个 `array` 只是指向节点的引用，所以记依赖不拷贝数据，上游也不会被提前回收。上面例子里 `b` 出现了两次，`inputs` 里就是同一个节点出现两次，执行时按节点身份去重一次。

`ArrayDesc` 是节点本身，形状、strides、dtype、依赖、状态、数据指针都在里面（`array.h:459-493`）。`mx::array` 只有一个成员 `array_desc_`（`array.h:514`），复制它只增加一次引用计数，`id()` 返回的就是节点地址（`array.h:227-229`）。切片与转置靠 `data` 加 `offset`、`strides` 共享同一段缓冲，不搬字节。

其余字段各管一件事。`offset`、`strides`、`data_size` 决定视图读哪一段：转置改 strides，切片改 offset 与 shape，广播把被扩展维度的 stride 记作 0。`flags` 里的 `row_contiguous` 是 kernel 选路径用的，参考实现的 `eval_gpu` 开头就检查它（`quantized_matmul.cpp:118-123`）。`siblings` 与 `position` 服务于多输出算子，`eval_gpu` 要拿到完整的输出列表，靠 `position` 把自己插回正确位置。`event` 记跨 stream 的依赖，`is_tracer` 在自动微分与向量化期间阻止节点被提前摘出图。

整张图因此是一棵由 `shared_ptr` 连起来的 DAG，节点没人引用时整条链一起释放。

## 为什么算子对象必须是 C++

`primitive` 的类型是 `std::shared_ptr<Primitive>`，`Primitive` 的两个求值方法是纯虚函数（`include/mlx/primitives.h:70-75`）：

```cpp
virtual void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) = 0;
virtual void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) = 0;
```

一个对象要成为图上的节点，就必须能响应这两个调用。Python 类实现不了 C++ 的纯虚接口，所以新算子的产生方式只能是 C++ 里的一个子类。

这条推理只走到「节点的实现必须是 C++」，还没回答节点为什么必须是一个 C++ 对象。后者没有语言层面的必然性。JAX 的 primitive 在 Python 层注册，执行交给 XLA 的 C++ 后端；PyTorch 也允许从 Python 注册自定义算子。MLX 选了另一条路，产物里有三条线索。

`Primitive` 的虚函数不止求值。同一个类里还有 `jvp`、`vjp`、`vmap`（`primitives.h:80-102`），对应自动微分的前向、反向与向量化。每个算子都要为这三种变换各给一份实现，分派同样走虚函数。

C++ 接口随包发布，不只在内部使用。`.venv/lib/python3.10/site-packages/mlx/include/mlx/` 下的 `array.h`、`primitives.h`、`scheduler.h`、`compile.h`、`backend/metal/device.h`，与 `.so` 里的实现一一对应。

官方还给了编译扩展的通道。`mlx/extension.py` 用 setuptools 与 cmake 把 `.cc`、`.cpp`、`.cxx`、`.metal` 源文件编成扩展（`_HOST_SOURCE_SUFFIXES` 与 `_METAL_SOURCE_SUFFIX`）。课程要加的算子 MLX 没有，只能走这条通道。

立项时的决策记录我没有，下面是从产物反推的解释（推断）：核心不依赖解释器，图遍历、变换分派、kernel 发射都在 C++ 里完成，Python 只是绑定。代价是新增算子必须编译一次，收益是热路径上不出现解释器。

顺带回答一个容易卡住的问题。`import mlx.core` 加载的是 `.venv/lib/python3.10/site-packages/mlx/core.cpython-310-darwin.so`（1.6 MB），Python 侧没有另一份实现，`mx.matmul`、`mx.eval` 都是这个扩展导出的函数。你写的每一行 MLX Python，已经在直接调 C++。

## 节点对象里存什么

这一节回答签名为什么只有三个参数。Task 2 的代码不需要这层推理，Task 3 只需要知道 `M`、`N`、`K` 从哪里来。

参考实现的构造函数只有三个参数（`src/extensions_ref/src/tiny_llm_ext.h:23-26`）：

```cpp
QuantizedMatmul(mx::Stream stream, bool use_simdgroup, bool use_split_k)
    : mx::Primitive(stream), use_simdgroup_(use_simdgroup), use_split_k_(use_split_k) {};
```

入口收的另外三个参数 `group_size`、`bits`、`transpose_b` 不进对象。它们在入口被检查一遍就丢掉：`bits` 必须是 4、`group_size` 必须是 128、`transpose_b` 必须是 true（`quantized_matmul.cpp:42-56`），kernel 里用 `constexpr` 写死同样的值（`quantized_matmul.metal:121-122`）。这个实现只支持一种权重格式，格式参数没有第二个可能值，存下来只是重复。

`use_simdgroup_` 与 `use_split_k_` 是执行策略，`eval_gpu` 按它们选 kernel（`quantized_matmul.cpp:153-166`）：

```
use_matvec      quantized_matvec_x4_fast_w4a16_g128_{f16,bf16}
use_split_k     quantized_matmul_simdgroup_splitk_w4a16_g128_{f16,bf16}
use_simdgroup_  quantized_matmul_simdgroup_w4a16_g128_{f16,bf16}
其余            quantized_matmul_vanilla_w4a16_g128_{f16,bf16}
```

四条路的输出形状完全相同，`out_shape[1] = b.shape()[0]` 在入口就算定（`quantized_matmul.cpp:53`），与选哪个 kernel 无关。差别只在 kernel 与 dispatch 几何：split_k 分支先算 partials 再做一次 reduce（`cpp:186-217`），matvec 分支按 simdgroup 摊输出（`cpp:225-233`）。

有一类值两个地方都不存。`M`、`N`、`K` 从 `inputs` 的形状现算（`cpp:125-127`），再作为标量参数传进 kernel（`set_bytes(M, 5)`，`cpp:221-223`）。`use_matvec` 同理，它由 `M <= 8` 当场算出（`cpp:137`）；切分份数 `split_k` 也按 threadgroup 数现算，对象里只存 `use_split_k_` 这个开关（`cpp:138-151`）。

三类值的分界因此是：结果在 `ArrayDesc`，推导不出来的产生方式在 `Primitive`，推导得出来的（维度、是否走 matvec、切几份）哪里都不存。原因不是省内存，是避免同一个事实有两份副本：存进对象的 `M` 在构造时算一次，之后 `inputs` 的形状一变，两份里就有一份是旧的。

`Reshape` 的构造参数收目标形状（`primitives.h:1812`），`Softmax` 收一个 `precise` 布尔（`primitives.h:2217`），与 `QuantizedMatmul` 存执行策略是同一个分工。

## 什么时候算

如果每次算子调用都立刻执行，`d` 在 `mx.matmul` 返回时就该有数值。MLX 没有这么做。执行方需要先看到完整的图，才能对共享节点去重，也才能把连续节点攒进同一次提交。后一件的收益可以量出来，见下面提交一节。

代价是数值的时点被推迟，任何要读它的地方都得先触发求值。

`mx.eval(d)` 从 `d` 出发，沿 `inputs` 反向走，收集 `status == unscheduled` 的节点，去重，排成拓扑序，再逐个编码。四步的实现（遍历顺序、visited 的容器、屏障时机）在 `.so` 里，以下是从接口与行为反推（推断）：

```
收集   从根出发沿 inputs 反向 DFS, 遇到已求值的节点就停
去重   按 array_desc_ 的指针身份判同一节点
排序   后序得到拓扑序, 依赖在前
编码   每个节点先 output_shapes(inputs) 定形状, 再 eval_gpu(inputs, outputs)
```

## 三个耗时尺度

构造、编码、等待三段各量一次（4096 个 float32 的 elementwise 乘法链，50 次取平均）：

```
构造 1 个节点            0.78 us   含一次 Python 调用
构造 100 个节点         51.38 us   0.51 us 一个
编码并提交 100 个节点  375.35 us   3.75 us 一个
编码提交并等待 100 个  669.03 us   6.69 us 一个
```

三次运行取中位数，单项的最大值与中位数相差 5% 到 30%，随机器负载浮动。脚本在文末。

构造一个节点只写几个指针，0.51 µs。给它编码一次 dispatch 约 3.8 µs。剩下的约 2.9 µs 是等 GPU 把 100 个 kernel 跑完，摊到每个节点上。

三段相差一个数量级。三千个节点的图按 0.51 µs 单价外推是 1.5 ms，建图在毫秒量级，不是微秒。逐个 eval 的代价则不在 GPU 上，每个节点都要重新走一遍编码与提交的固定开销，下一节给出对比。

## 编码做完不等于算完

`Status` 有三个值，`array.h:375-388` 的原始注释：

```cpp
enum Status {
  // The output of a computation which has not been scheduled.
  unscheduled,
  // The array's `eval_*` function has been run, but the computation is not
  // necessarily complete. The array will have memory allocated and if it is
  // not a tracer then it will be detached from the graph.
  evaluated,
  // If the array is the output of a computation then the computation
  // is complete. Constant arrays are always available.
  available,
};
```

`evaluated` 的注释写明：`eval_*` 调过了，显存分配了，计算不一定完成。`available` 才能读。两个状态之间的间隔，就是命令已提交、GPU 还在跑的那段时间。

Python 侧能量出这段间隔。同一张单节点图，`mx.async_eval` 提交后立即返回，`mx.eval` 要等它结束：

```
1 op async_eval    20.5 us
1 op eval         137.0 us
```

（mlx 0.32.4，Apple silicon，4096 个 float32 乘一次，各跑 50 次取中位数。）

`mx.eval` 返回时，被求值的输出都在 `available`。

## 同一个图可以 eval 多次

`mx.eval` 处理的是还缺数值的那部分，不是整张图。收集阶段遇到 `evaluated` 或 `available` 的节点就停，这些节点是边界，不重算。

共享节点就是这么处理的。`d = mx.matmul(b, b)` 里 `b` 出现两次，第二次碰到它时 `visited` 里已经有了，只编码一次。

分步求值。先 eval 中间节点，再 eval 最终节点：

```python
import mlx.core as mx
MiB = 2 ** 20
mx.clear_cache()

a = mx.zeros((4096, 4096), dtype=mx.float32)
b = a + 1.0
mx.eval(b)                            # 只算到 b
print(mx.get_active_memory() / MiB)   # 128.0: a 与 b

d = mx.matmul(b, b)
mx.eval(d)                            # b 已经 available, 不重算
print(mx.get_active_memory() / MiB)   # 192.0
```

这个例子只用来验证边界规则。单看显存与总耗时，直接 eval `d` 就够，中间那次没有收益。真正需要分步 eval 的是另外三种情形。

控制流依赖数值。自回归生成每步都要拿到采样出的 token，才能建下一步的图。`logits → sample → next token → 下一次 forward` 这条链里，数值是下一步的输入，不 eval 就动不了。

定位错误。写完一个算子立刻 eval 它，错误停在被测算子这一层；攒到最后一起 eval，栈上看到的是 `mx.eval`，几十层之外的算子出错也长一个样。

分离计时。profiler 要把建图与求值分开测，建完立刻 eval 是最直接的切法。

第二次 eval 覆盖的只有 `d`，`b` 是它的输入边界。对同一个 root 再 eval 一次，收集阶段在根上就停，什么也不做。

状态因此是单向的：节点第一次被收集到时进入编码，`eval_gpu` 跑完变 `evaluated`，提交的命令执行完变 `available`，之后不再回到 `unscheduled`。这也解释了为什么「早 eval」既不会让计算白做，也不会把峰值降下来：省下的只是重复编码，已经算过的结果仍然被引用着。

## 输出的显存由 eval_gpu 申请

传给 `eval_gpu` 的 `outputs` 有形状、没有显存。分配责任在评估函数这一侧，`primitives.h:63-68` 的注释原文：

```
To avoid unnecessary allocations, the evaluation function
is responsible for allocating space for the array.
```

`output_shapes` 只返回形状，不实现时直接抛异常（`primitives.h:112-114`）。本项目的参考实现照这个分工写，`QuantizedMatmul::eval_gpu` 在检查完连续性、算出 `M`、`N`、`K` 之后立刻分配（`src/extensions_ref/src/quantized_matmul.cpp:135`）：

```cpp
out.set_data(mx::allocator::malloc(out.nbytes()));
```

分配之后才绑 buffer、设 dispatch。

这一条决定 Task 3 的第一行代码。忘了它，后面往里写的是空指针。

## 一批 kernel 怎么到 GPU

单个 kernel 从编码完成到 GPU 跑完有一段固定开销，这段开销与一次提交里装了多少 kernel 无关。实测（同一台机器，4096 个 float32 的 elementwise 乘法链，各跑 30 次取平均）：

```
1 个 kernel, 一次 eval        194.3 us
100 个 kernel, 一次 eval      538.9 us
100 个 kernel, 每个单独 eval  14664.4 us
```

100 个 kernel 攒成一次提交，比逐个提交快约 27 倍。固定开销只付一遍，随 kernel 数增长的部分约 3.5 µs 一个。

Metal 先为要运行的 GPU 函数准备可绑定的执行对象，再记录派发命令。两个过程都以同一台设备为起点：

```
GPU 函数源码 → library → 按函数名取出 function
                          └─ 与 Device 一起创建 ComputePipelineState（ps）

Device → CommandQueue → CommandBuffer（cb）→ ComputeCommandEncoder
                                                 │
                                                 ├─ 绑定 ps、输入/输出数据缓冲
                                                 └─ dispatch：记录本次 grid、threadgroup
结束编码 → commit cb → GPU 执行记录的派发 → 等待完成后读取输出
```

`ps` 指定派发哪一个已编译的 GPU 函数；它在派发前创建，可以供同一函数的多次派发使用。grid 和 threadgroup 属于每次派发，决定启动的线程范围及分组。数据缓冲存放数组，例如 `x`、`y`、`out`；绑定槽位固定时，各次派发仍可绑定不同的数据缓冲。command buffer 存放命令，一次提交后不能再拿来记录下一批命令。encoder 负责记录绑定与派发，调用 `dispatchThreads` 时尚未执行 GPU 函数；结束编码并 `commit` 后，GPU 才能执行。一个 command buffer 可以记录多次派发，两个不同的 `ps` 也可以分别用于同一次提交中的两条派发。计算图中节点的输入依赖决定哪些结果需要先产生，`ps` 本身不保存节点之间的依赖关系。

MLX 另有一个 `metal::CommandEncoder`（`include/mlx/backend/metal/device.h:24`），与 Metal 的编码器同名不同物。它自己持有 `queue_` 与 `buffer_`（`device.h:115-116`），内部再懒建一个 Metal 的 encoder。编码一次 kernel 的动作是固定的几个调用：装内核、绑 buffer、传标量、dispatch（`device.h:46-57`）。

提交涉及三个动作（`device.h:97-102`）：`end_encoding()` 结束当前计算 encoder；同一 command buffer 在提交前仍可以记录其他 encoder 的命令。`commit()` 把当前 buffer 交给队列，并为后续命令创建下一只 buffer。MLX 的 `synchronize()` 会结束当前编码、提交并等待完成。`evaluated` 表示后端已编码，`available` 表示结果计算完成。

`Stream` 是 `(index, device)` 二元组（`include/mlx/stream.h:14-17`）。一个 stream 对应一个 `CommandEncoder`，也就对应一条队列与一个当前 buffer，取用入口是 `get_command_encoder(Stream)`（`device.h:257`）。编码器按 stream 的下标存在一张表里（`device.h:259-260`），所以多条 stream 就是多条队列，可以同时提交。

并行提交本身不产生等待。依赖来自图：某个节点的输入，是另一条 stream 上产生的数组。这时消费方不能直接读，因为生产方可能还没跑完。让整台设备同步代价太大，MLX 用事件只串起这一条依赖：生产方的 encoder 在算完之后 `signal_event`（`device.h:99`），消费方的 encoder 在开始之前 `wait_event`（`device.h:98`）。数组上的 `event` 字段（`array.h:468-469`）记的就是这个还没兑现的信号，`attach_event` 挂上，`detach_event` 在把数组读走之前先 `check_error()` 再清空（`array.h:411-419`）。没有事件，跨流依赖只剩同步整个设备这一条路。

`set_input_array(const array&, int, int64_t offset)` 绑定缓冲及偏移（`device.h:48`），strides 需要另外传递，或者由 kernel 的布局约定确定。要求行连续的量化 kernel 由调用边界准备连续输入；Axpby 的 general kernel 则在槽 5 至 8 传 shape、两份 strides 与 ndim，直接读取转置或切片视图。布局约定由 primitive 与 kernel 共同保证。

## 数值什么时候没了

节点没人引用时释放。Python 侧删掉最后一个引用，`active` 归零，同时 `cache` 上涨同样的数量：

```
分配后    active 64.00  cache  0.00
删引用    active  0.00  cache 64.00
clear后   active  0.00  cache  0.00
```

（4096×4096 float32 一块，读 `mx.get_active_memory()` 与 `mx.get_cache_memory()`。）

缓冲回到分配器手里，没有还给系统，下一块同尺寸的申请直接复用。`mx.clear_cache()` 清空这个池子。连续 decode 不会反复向系统要内存，原因在这里；做内存实验之前要先 `clear_cache()`，否则上一轮的分配被复用，读数看不出区别。

## 融合不发生在 eval

`mx.eval` 只决定每个节点怎么执行，不改图的结构。把多个节点换成一个，发生在 `mx.compile` 里。

`compile` 的签名在 `include/mlx/compile.h:13-15`：

```cpp
std::function<std::vector<array>(const std::vector<array>&)> compile(
    std::function<std::vector<array>(const std::vector<array>&)> fun,
    bool shapeless = false);
```

它收一个函数，返回一个函数。第一次调用返回的函数时，先把传入的函数跑一遍，记下它调了哪些算子，得到一张图；此后同形状的调用直接复用这张图，不再进 Python（trace 的机制从头文件看不到，从签名与 `shapeless` 参数反推）。编译期在这张图上做两种改写，档位写在 `compile.h:10`：

```cpp
enum class CompileMode { disabled, no_simplify, no_fuse, enabled };
```

`no_simplify` 关掉化简，`no_fuse` 关掉融合。这里的「编译」是 MLX 自己的图编译器，产物仍是对 Metal kernel 的调用，不是机器码。

所以「跨节点改图」的准确含义是：`eval` 逐个节点编码，节点之间的边界不变；`compile` 重写整张图，把相邻节点替换成一个，或者消掉等价表达式。融合规则本身没有公开文档（推断），能确定的是这两个档位各自可控。

Day 4 手写 fused kernel 的理由也在这里：`silu(gate) * up` 在 Python 侧是三个算子，编译期能否合并取决于规则覆盖到什么程度，手写 Metal kernel 直接保证一次显存往返。

## 落到 Task 2 和 Task 3

Task 2 写两个函数。入口检查参数、构造 `QuantizedMatmul`、构造 `ArrayDesc` 并返回，状态停在 `unscheduled`，不碰显存。`eval_cpu` 抛异常：`mx.default_device()` 在当前环境返回 `Device(gpu, 0)`，这个数组不会被要求走 CPU，抛异常是把这个前提写成可执行断言。

Task 3 写 `eval_gpu`：申请输出显存、绑 buffer、设 grid、dispatch。这几步各自依赖前面讲过的一件事：`outputs` 有形状没有数据，因为分配责任在评估函数这一侧（`primitives.h:63-68`）；`inputs` 里的数组已经算好、在显存里，因为执行方保证依赖先算完；grid 的大小由 `M`、`N`、`K` 算出来，这三个数从 `inputs` 的形状取（`quantized_matmul.cpp:125-127`）。

## 对象地图

| 对象 | 语言 | 职责 | 持有者 |
|---|---|---|---|
| `mx.core.array` | Python | 用户手里的值 | Python 变量 |
| `mx::array` | C++ | 句柄，一个 `shared_ptr` | Python 对象 |
| `ArrayDesc` | C++ | 节点：形状、依赖、状态、数据指针 | `mx::array` |
| `Primitive` 子类 | C++ | 这个节点怎么算，以及本次调用的参数 | `ArrayDesc::primitive` |
| `Data` | C++ | 一块缓冲及其释放责任 | `ArrayDesc::data` |
| `MTL::ComputePipelineState` | Metal | 针对设备与 GPU 函数编译的可绑定对象；报告该函数的线程组上限 | 设备构造，编码派发时引用 |
| `metal::CommandEncoder` | C++ | 往当前 buffer 追加命令 | 按 stream 存放 |
| `MTL::CommandBuffer` | Metal | 一次提交的单位 | `CommandEncoder::buffer_` |
| `MTL::CommandQueue` | Metal | 提交端口 | `CommandEncoder::queue_` |

结果一侧的持有链：

```
Python 变量
  └─ mx.core.array
       └─ mx::array                array.h:514
            └─ ArrayDesc            array.h:459
                 ├─ primitive ──> QuantizedMatmul 实例
                 ├─ inputs ─────> 其他 array, 构成 DAG
                 └─ data ───────> Data, eval 前为空
```

执行一侧不在这条链上，它跟着 stream 走：`Stream` 对应 `CommandEncoder`，后者持有 `CommandQueue` 与 `CommandBuffer`。

一次调用的控制流，全部在 CPU 上，微秒级：

```
quantized_linear(x, w)
  → 绑定层把 Python 数组转成 mx::array, 不拷贝
  → 入口构造 primitive 与 ArrayDesc, 返回
  → Python 拿到一个 unscheduled 的 array
```

`mx.eval` 时的数据流：

```
mx.eval(root)
  → 反向收集 unscheduled 节点, 去重, 拓扑排序
  → 逐节点: output_shapes 定形状, eval_gpu 申请显存并编码
  → end_encoding, commit
  → synchronize 之后 status 变 available
```

## 视图与连续布局

一个数组可以没有自己的缓冲区。`x.T`、`x[1:3]`、`mx.broadcast_to(x, ...)` 都只是换一套 shape、offset、strides 去解释同一块内存，数据不动。`ArrayDesc` 里的 `data` 是 `shared_ptr<Data>`，视图与它的来源共享这个指针（`array.h:475-477`）。

取一个 3×4 的数组：

```python
x = mx.arange(12).reshape(3, 4)
t = x.T
```

底层只有一份 12 个数的缓冲区，`x` 与 `t` 各自记一套读取规则：

```
buffer      [0][1][2][3][4][5][6][7][8][9][10][11]

x (3,4)     地址 = i*4 + j
  行0        0  1  2  3
  行1        4  5  6  7
  行2        8  9 10 11

t = x.T (4,3)   地址 = j*4 + i
  行0        0  4  8
  行1        1  5  9
  行2        2  6 10
  行3        3  7 11
```

`t` 的第一行在缓冲区里并不相邻，它取的是第 0、4、8 个元素。

Metal kernel 拿到的是基地址加一个偏移，没有任何 strides 信息。它在代码里按 `i * N + j` 算位置，这个公式假定数组行主序存放。把 `t` 直接传进去，读出来是：

```
i=0 j=1   i*3+j = 1   读到 1, 而 t[0,1] 是 4
i=1 j=0   i*3+j = 3   读到 3, 而 t[1,0] 是 1
```

`mx.contiguous(t)` 在需要时复制出一份行主序的缓冲区：

```
[0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
```

此后 `i*3+j` 才指向正确的元素。已经连续的数组不复制，函数直接返回原数组。

让 kernel 自己按 `j*4+i` 读取，可以直接处理这个视图。它改变寻址指令和相邻线程的请求地址；实际指令数量取决于实现与编译结果，实际传输量还取决于事务粒度和缓存驻留。另一种做法是先复制一次，换取后续算子的连续访问；两者的取舍需要计入复制与所有使用者的成本。

对这个项目，这条规则落在 `quantize.py` 的边界上：四个输入都过一遍 `mx.contiguous`。模型里的激活与量化权重通常已经连续，所以多数调用是空操作；它防的是有人把切片或转置的视图直接传进来。

## 把一次 array 求值核对到 GPU 命令

前面的对象关系可以通过 Axpby 捕获直接检查。`4_axpby_capture.py` 创建两个 `4096×4096` float32 输入，各占 64 MiB；连续布局传入 `(x,y)`，转置布局传入 `(x,y.T)`。两组输出各占 64 MiB，转置视图仍共享 y 原缓冲。输入先求值并预热三次，再捕获一次 `ext.axpby` 和 `mx.eval`，捕获结束后比较结果与 `x+yy`。

```sh
CAPTURE_DIR="$(mktemp -d "${TMPDIR%/}/axpby-lifecycle.XXXXXX")"
MTL_CAPTURE_ENABLED=1 PYTHONPATH=src/extensions pdm run python \
  probes/week2/day1/4_axpby_capture.py --layout transpose \
  --trace "$CAPTURE_DIR/transpose.gputrace"
gpudebug --json --oneshot -t "$CAPTURE_DIR/transpose.gputrace" --timeout 120 \
  -c 'go /commands' -c 'go cb0' -c 'go ce0' \
  -c 'info disp0' -c 'go disp0/compute'
```

`commands` 下的 `cb0` 是一次提交的 command buffer，`ce0` 是其计算 encoder，`disp0` 是一条派发。`go` 列出子节点；导航名称以本次 trace 实际列出的内容为准。compute 子树列出绑定槽，前三槽是输入 x、输入 y、输出 out。转置组另有内联的 shape、strides、ndim。2026-11-17 本机捕获实际包含一个计算 dispatch，名称为 `axpby_general_float32`，线程网格为 `16777216×1×1`、单组 `1024×1×1`；导出槽 6、7 按 int64 解码，分别得到 `(4096,1)` 与 `(1,4096)`。这将 array 的视图元数据接到了 GPU 读取规则。

捕获边界还暴露了 command buffer 的生命周期。MLX 0.32.4 提交后预先创建下一只 buffer，因而预热完成时已有一只在捕获外创建的 buffer。最初只在 `start_capture` 和 `stop_capture` 之间求值，结果正确，trace 却只有同步命令。捕获开始后先 `mx.synchronize()`，提交旧 buffer 并创建捕获内的新 buffer，随后的 Axpby 才进入实际记录。入口已经包含这个动作，原计算代码保持原样。

同一份 trace 还能由 `gpudebug` 重放并读取编译后 shader 数据：

```sh
gpudebug --json --oneshot -t "$CAPTURE_DIR/transpose.gputrace" --timeout 1800 \
  -c 'profile run --gpu-state default --exec serial --embed' -c 'wait'
gpudebug --json --oneshot -t "$CAPTURE_DIR/transpose.gputrace" --timeout 120 \
  -c 'profile load' \
  -c 'info /performance/timeline' \
  -c 'info /performance/shaders/comp0'
```

`--embed` 保存分析到 trace，重新打开后的 `profile load` 加载该数据；本机实测中，采集成功后立即访问性能子树曾返回空内容，重新加载后读数完整。timeline 的 GPU time 描述重放范围，shader 的 Instructions 和寄存器字段描述编译后的实现。连续路径实际报告 7 条指令、3 个临时寄存器，general 报告 118 条指令、16 个临时寄存器。general 的整除、取余与 strides 换算给出了额外指令的源码来源；这些字段分别于完整矩阵的动态执行总数和物理访存量解释。

本次单次重放 GPU time 为连续 2.2706 ms、转置 3.5630 ms；普通 bench 另一次运行的平均墙钟为 2.6020 ms 与 3.0463 ms。捕获重放与普通调用处于不同测量条件，求值生命周期相同，计时范围与资源状态各自记录。原始命令报告、参数二进制与 metadata 保存在 `docs/evidence/axpby-kernels/`；完整捕获、加载、计数器解释见 `docs/metal-counter-investigation.md`。

## 与别的框架的差别

只留一条对读代码有用的差别。MLX 的语义更接近 JAX：函数是纯的，算子被 trace 成图，执行前有一张完整的图。PyTorch 默认 eager，调用即算，图只在反向传播时由 autograd 记下来。这条差别决定读 MLX 代码时的预期：`mx.matmul` 返回的不是数值。

真正的框架对照要拿各自的源码与文档做锚点，那是另一篇的范围，Day 2 的实验也用不到，这里不展开。

## 边界

四步的实现细节、屏障插入条件、buffer 切分阈值都在 `.so` 里，头文件只有声明。要确认某一条，手段是改一个条件再看内存或耗时是否随之变化。

`mx.compile` 的融合规则同样没有公开文档，本文只用到两条：融合在编译期，`eval` 不做融合。

## 复现脚本

三段脚本，在仓库根目录用 `.venv/bin/python` 直接跑。mlx 0.32.4，Apple silicon。

内存与求值：

```python
import mlx.core as mx

MiB = 2 ** 20
mx.clear_cache()
mx.reset_peak_memory()

a = mx.zeros((4096, 4096), dtype=mx.float32)
b = a + 1.0
c = b * 2.0
d = mx.matmul(b, b)
print("建图", mx.get_active_memory() / MiB)          # 0.0

mx.eval(d)
print("eval(d)", mx.get_active_memory() / MiB,
      mx.get_peak_memory() / MiB)                    # 192.0 192.0

print("读 c[0,0]", c[0, 0].item())
print("之后", mx.get_active_memory() / MiB)          # 256.0
```

三段耗时，跑三轮取中位数：

```python
import time
import mlx.core as mx

x0 = mx.ones((4096,), dtype=mx.float32)


def chain(n):
    x = x0
    for _ in range(n):
        x = x * 1.0000001
    return x


def t(fn, reps):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e6


rows = {"build 1": [], "build 100": [], "async 100": [],
        "eval 100": [], "async 1": [], "eval 1": []}
for _ in range(3):
    rows["build 1"].append(t(lambda: x0 + 1.0, 50))
    rows["build 100"].append(t(lambda: chain(100), 50))
    rows["async 100"].append(t(lambda: mx.async_eval(chain(100)), 20))
    rows["eval 100"].append(t(lambda: mx.eval(chain(100)), 20))
    rows["async 1"].append(t(lambda: mx.async_eval(chain(1)), 50))
    rows["eval 1"].append(t(lambda: mx.eval(chain(1)), 50))

for name, values in rows.items():
    print("%-10s min %7.2f  med %7.2f  max %7.2f"
          % (name, min(values), sorted(values)[1], max(values)))
```

引用计数与缓冲池：

```python
import gc
import mlx.core as mx

MiB = 2 ** 20
mx.clear_cache()

a = mx.zeros((4096, 4096), dtype=mx.float32)
mx.eval(a)
print("分配后", mx.get_active_memory() / MiB, mx.get_cache_memory() / MiB)

del a
gc.collect()
print("删引用", mx.get_active_memory() / MiB, mx.get_cache_memory() / MiB)

mx.clear_cache()
print("clear 后", mx.get_active_memory() / MiB, mx.get_cache_memory() / MiB)
```
