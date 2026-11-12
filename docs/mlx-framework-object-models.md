# 形状、节点、算子、显存：一个未算出的矩阵在 MLX、JAX、PyTorch、CUDA 里的位置

先看一个现象。执行 `b = mx.ones((2, 2)) @ mx.ones((2, 2))` 之后，`b.shape` 立刻是 `(2, 2)`，但此时还没有输出缓冲区被分配，也没有任何 GPU 命令被提交。`b` 能回答形状问题，不能回答数值问题。

如果让一个 Python 对象同时保存形状、计算图和显存，每次算子调用都要在 Python 里走一遍分派，每次复制数组都要决定复制哪一层，每个后端也要各自发明一套节点表示。MLX 把这个对象拆成四层，让每一层只回答一个问题。

第一层是 Python 手里的句柄。`mx.array` 是 Python 侧的值对象，在公开头文件里只持有一个指向节点描述的共享指针（`mlx/array.h:514`），复制这个句柄不会复制图，也不会复制显存。

第二层是图节点。`ArrayDesc` 是图节点的描述，同时保存 shape、dtype、输入、算子、状态和缓冲区指针（`mlx/array.h:459-493`），它才是图上的节点。

第三层是算子对象。`Primitive` 是算子对象的基类，实现保存这次调用的参数，基类保存目标 stream，并声明 CPU、GPU、梯度、批量四种行为，它回答“这个节点怎么算”。

第四层是显存。`Data` 是缓冲区的持有者，保存真正的设备缓冲区，不同数组可以共享同一个 `Data`，这一层回答“数值放在哪里”。

这个拆法带来的直接后果，是 Python 侧只负责构图和持有句柄，图遍历、去重、拓扑排序和编码都在 C++ 里完成。这是从头文件的成员布局推断出来的，官方文档没有用一句话总结这套分层。

所依据的版本：MLX 0.32.4（`mlx/version.h:7-9`，文档页也标 0.32.4）；PyTorch 2.7（本机安装 2.7.1，文档取 `docs.pytorch.org/docs/2.7`）；JAX 取 PyPI 上的 0.12.0.1 与 `docs.jax.dev/en/latest`，文档页没有统一标版本号；CUDA 取 Programming Guide v13.5；Metal 取 Apple 当前文档，Apple 没有在页面上给出版本号。MLX 的锚点根目录是 `/Users/oujinsai/Projects/tiny-llm/.venv/lib/python3.10/site-packages/mlx/include/`。

## 一个 2×2 的矩阵，为什么先得到形状

这一节要回答的是，形状已经能读、数值还没有的时候，这个东西在别的框架里归谁所有。先看 MLX 的最小例子：

```python
a = mx.ones((2, 2))
b = a @ a
print(b.shape)   # (2, 2)
```

创建时只确定了 shape、dtype 和依赖关系，输出缓冲区还没有分配。`ArrayDesc` 的 status 字段负责表达这件事，它有三个取值：unscheduled 表示计算还没有排进调度，evaluated 表示内存已经分配但计算不一定完成，available 表示可以安全读取（`mlx/array.h:375-389`）。

真正把数值写进去的是求值。`mx.eval` 会收集所有 unscheduled 的节点，按节点身份去重，排成拓扑序，再按 stream 编码。

非 const 的 `item()` 会直接调用 eval（`mlx/array.h:562-568`）。const 的 item() 遇到 unscheduled 会抛异常（`mlx/array.h:576-579`）。这两处把“读之前必须先算”写进了接口。

后端入口在头文件里只有一行声明，`gpu::eval(array&)`（`mlx/backend/gpu/eval.h:15`）。收集、去重和拓扑排序的实现体在 `.so` 里，发布头文件看不到。

JAX 在编译路径上把“没有数值的值”交给另一个对象。`jax.jit` 在真正编译之前，必须先把 Python 函数跑一遍，才能知道里面有哪些算子、形状怎么变。这一次运行没有真实数值，只有抽象值。

为了让这次运行不碰数据，JAX 用 tracer 代替真实数组。tracer 是一个记录操作的占位对象，它只带 shape 和 dtype，每次参与运算就把对应的 primitive 记进当前 trace。官方文档里的追踪输出写成 `JitTracer(~float32[])`，并说明 `ShapedArray` 只记录类型和形状、不记录数值（https://docs.jax.dev/en/latest/601/jax-primitives.html）。

trace 结束之后，编译器需要一份稳定的结构来读，JAX 把记录下来的方程写成 jaxpr。jaxpr 是一个中间表示，里面是一串方程，每个方程记录 primitive、输入、输出和参数。官方 API 文档把 `JaxprEqn` 的字段列为 `invars`、`outvars`、`primitive`、`params`（https://docs.jax.dev/en/latest/jax.extend.core.html）。

jaxpr 是 JAX 自己的表示，XLA 不认识，所以还要 lowering 成 StableHLO。lowering 是把高层算子翻译成目标编译器能接受的低层表示，StableHLO 是 XLA 前端使用的一种可移植中间表示。XLA 架构文档把这一层描述为 frontend 与 compiler 之间的移植层（https://openxla.org/xla/architecture）。

编译完成后得到 XLA executable。executable 是针对一组输入签名编译好的可执行对象，`jax.jit` 在首次调用时做 trace、编译、执行，后续调用直接复用（https://docs.jax.dev/en/latest/jit-compilation.html）。AOT 文档把这条链拆成 trace、lower、compile、execute 四步（https://docs.jax.dev/en/latest/aot.html）。

dispatch 之后，用户拿到的是 `jax.Array`。官方异步调度文档把它称为 future，“a value that will be produced in the future on an accelerator device but isn’t necessarily available immediately”，shape 和 dtype 可以查，读值要 `block_until_ready()` 或搬回主机（https://docs.jax.dev/en/latest/async_dispatch.html）。

设备缓冲区由运行时管理，用户手里没有单独的显存对象；`jax.Array` 是访问这块缓冲区的句柄。这是从 API 面推断的，官方文档没有把缓冲区暴露成单独类型。

如果只需要形状，JAX 还提供 `jax.eval_shape`，返回 `ShapeDtypeStruct`。这是一个只保留形状和类型的对象，不需要编译或执行（https://docs.jax.dev/en/latest/_autosummary/jax.eval_shape.html）。

PyTorch 的 eager 模式没有“将来会算出来”的数组对象。`torch.empty` 返回的 Tensor 从创建时就有 Storage，shape、stride、dtype 都定了，内容是没有写过的字节，延迟计算在这里没有发生（https://docs.pytorch.org/docs/2.7/generated/torch.empty.html）。

真正接近“形状已知、数值没有”的对象只在编译路径里出现，叫 FakeTensor。官方文档说 fake tensor 是“a real tensor in all respects, except that it doesn’t actually have any data”，实现上是 meta tensor 的 `__torch_dispatch__` 子类，由 `FakeTensorMode` 的 converter 从真实 tensor 转换出来（https://docs.pytorch.org/docs/2.7/torch.compiler_fake_tensor.html）。

FakeTensor 回答的是编译期问题：这个算子的输出 shape、stride、dtype、device 是什么。数值要等 `torch.compile` 生成的代码在真实 tensor 上运行才出现。自定义算子要参与这条路径，必须注册 fake kernel，官方 `torch.library` 文档把 `register_fake` 说成是算子进入 `torch.compile`、export、FX 的必要条件（https://docs.pytorch.org/docs/2.7/library.html）。

CUDA 没有对应的数组对象。`cudaMalloc` 或流有序的 `cudaMallocAsync` 返回一块设备地址，shape、dtype、生命周期都由调用者用 host 侧变量记住。内核启动是异步的，函数调用立刻返回，数值在流执行到那个内核时才写进这块地址（https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html）。CUDA 没有数组对象这一点是从 API 面推断的，文档没有一句这样的表述。

CUDA 里唯一像图节点的是 `cudaGraphNode_t`。`cudaGraphNodeParams` 放 kernel 的函数指针、grid、block 和 `kernelParams`，没有 shape 和 dtype；节点由 `cudaGraphAddNode` 加进 `cudaGraph_t`，`cudaGraphInstantiate` 之后才成为可启动的 executable graph（https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html）。

Metal 的层次更低。`MTLBuffer` 是一块显存，`MTLCommandBuffer` 的官方定义是“A container that stores a sequence of GPU commands that you encode into it”，两者都不带 shape（https://developer.apple.com/documentation/metal/mtlcommandbuffer）。

MLX 的 `mx.array` 和 `ArrayDesc` 是在这层之上补出来的对象。

同一个未算出的值，在 MLX 里是值对象的一个状态，在 JAX 里是 tracer 与 future 两个对象，在 PyTorch 里被关进编译器内部，在 CUDA 和 Metal 里没有对应物。这个归属差别会决定下一节的分工：谁知道形状，谁就有资格决定节点怎么算。

## 节点要知道怎么算，算子对象放在哪

上一节停在形状与数值的归属上，接下来的问题是，节点要知道怎么算，这份知识放在哪个对象里。MLX 的最小结构是一个 C++ 子类，下面的参数列表省略：

```cpp
struct Add : Primitive {
  void eval_cpu(...) override;
  void eval_gpu(...) override;
};
```

`Primitive` 是一个 C++ 抽象类（`mlx/primitives.h:49-51`）。

`eval_cpu` 与 `eval_gpu` 是纯虚函数（`mlx/primitives.h:70-75`），负责 CPU 和 GPU 的实际计算。

`jvp`、`vjp`、`vmap` 是虚函数（`mlx/primitives.h:80-102`），负责梯度与批量的变换。这三组接口把“怎么算”“梯度怎么传”“批量怎么扩”放在同一个对象上。

`Primitive` 还有 `name()`（`mlx/primitives.h:105`）和 `output_shapes()`（`mlx/primitives.h:112-114`）。name() 用于标识算子，output_shapes() 在抽象求值时给出输出形状。

`is_equivalent()` 默认返回 false（`mlx/primitives.h:108-110`），编译期的化简可以用它判断两个节点能否合并。这是从头文件接口推断的，头文件没有给出 pass 的实现。

`ArrayDesc` 直接持有 `shared_ptr<Primitive>`（`mlx/array.h:464`），构造时把 primitive 和 inputs 一起传进去（`mlx/array.h:497-501`）。节点持有一个算子实例，实例里装着这次调用的参数。

`Primitive` 的构造函数还要求传入 `Stream`（`mlx/primitives.h:51`），所以节点连目标 stream 都记住了。Stream 是设备队列的句柄，只有 index 和 device 两个字段（`mlx/stream.h:13-19`），提交时按 stream 分组。

JAX 的算子对象是一个 Python 对象。官方文档的用法是先创建 `multiply_add_p = core.Primitive("multiply_add")`，再用 `def_impl` 注册实现、用 `def_abstract_eval` 注册抽象求值、用 `mlir.register_lowering(multiply_add_p, ..., platform='cpu')` 注册 lowering（https://docs.jax.dev/en/latest/601/jax-primitives.html）。

primitive 本身是全局的、无状态的。每次调用的参数通过 `bind(*args, **params)` 传入，最后落在 jaxpr 的 `JaxprEqn.params` 里（https://docs.jax.dev/en/latest/jax.extend.core.html）。节点与算子对象在这里是分开的：primitive 是全局单例，方程才是这次调用的节点。

变换规则放在分开的表里。`jax.interpreters.ad` 管自动微分，`batching` 管批量，`mlir` 管 lowering，它们按 primitive 注册，primitive 对象本身不带这些规则。要接 C++ 或 CUDA 代码，走 FFI 的 `CustomCall`，注册入口仍然在 Python（https://docs.jax.dev/en/latest/ffi.html）。

PyTorch 的算子对象是 schema 加 dispatch key。算子的签名和元数据写在 `aten/src/ATen/native/native_functions.yaml`，代码生成器据此产生 `aten::` 函数和 `OpOverloadPacket`，运行时由 `c10::Dispatcher` 按 `DispatchKey` 找到对应 kernel（https://github.com/pytorch/pytorch/blob/v2.7.0/aten/src/ATen/native/native_functions.yaml）。

dispatch key 是 dispatcher 用来选择实现的一组标签，设备类型、autograd、fake、vmap 都在这一层分叉。`torch.ops.aten.matmul.default` 就是这条链上的 `OpOverload`，官方源码里的 key 定义在 `c10/core/DispatchKey.h`（https://github.com/pytorch/pytorch/blob/v2.7.0/c10/core/DispatchKey.h）。

自定义算子可以用 C++ 的 `TORCH_LIBRARY`，也可以用 Python 的 `torch.library.custom_op`。后者会往同一个 dispatcher 里注册 schema 和 kernel，autograd、fake、vmap 各自是单独的注册项（https://docs.pytorch.org/docs/2.7/library.html）。所以 PyTorch 的算子对象是一个全局名字，行为按 dispatch key 分叉。

CUDA 没有算子对象。一个 kernel 就是 `__global__` 函数，启动时把函数指针、grid、block 和参数数组交给运行时；CUDA graph 节点保存的也是这些字段，没有注册表、没有 shape 推断、没有自动微分、没有 vmap。cuBLAS、cuDNN 这些库在这一层之上提供 host 函数，没有进入图节点的表示。CUDA 没有算子对象这一点是从 kernel 启动接口推断的。

Metal 的对应物是 `MTL::ComputePipelineState`。从 `MTL::Library` 里取 function，编码时 `set_compute_pipeline_state` 设管线，`dispatch_threadgroups` 发线程组（`mlx/backend/metal/device.h:59-61`），MLX 通过 `Device::get_kernel` 取管线（`mlx/backend/metal/device.h:190-195`）。这里也没有节点对象，节点是 MLX 补的。

这三种把算子放在对象里的做法，带来三个可观察的后果。第一个是扩展方式。MLX 的新算子必须是一个 C++ 子类，因为纯虚接口不可能由 Python 类实现；官方给了编译扩展的通道，用 setuptools 和 cmake 把 `.cc`、`.cpp`、`.metal` 编进扩展，这一点生命周期文档已经写过。

JAX 的新 primitive 可以完全写在 Python 里，只要 lowering 能落到 StableHLO；落到不了就写 FFI。PyTorch 的新算子先写 schema，再按 device type 注册 kernel，Python 和 C++ 都能注册，但要让 `torch.compile` 认得，还得补 fake kernel。CUDA 的新算子就是写一个 kernel 再用 `cudaGraphAddNode` 或 `<<<>>>` 启动。

第二个是分派开销。MLX 每个节点走一次 C++ 虚调用，`eval_cpu`、`eval_gpu`、`jvp`、`vjp`、`vmap` 都是同一种分派。JAX 在 eager 模式下每个算子要走一次 Python 的 `bind` 和 trace 分派，`jit` 之后才是每个 executable 一次。

PyTorch 每个算子走一次 dispatcher 的 key 计算和表查找，再进 kernel。CUDA 每个 kernel 走一次驱动启动。这里只描述结构，没有测过任何一条路径的耗时，所以不给数字。

第三个是能否被别的语言前端复用。MLX 的头文件随包发布，`array.h`、`primitives.h`、`scheduler.h`、`compile.h` 和 Metal 后端头文件都在，任何能链接这个库的前端都可以构造同一张图，这是从发布产物推断，文档没有这样承诺。

JAX 的 primitive 是 Python 对象，非 Python 前端不能直接复用这些对象，但可以绕过 JAX 直接产出 StableHLO 或 jaxpr。XLA 官方架构文档把 JAX、PyTorch、TensorFlow 并列为 frontend，说明可复用的是 XLA 这一层（https://openxla.org/xla/architecture）。

PyTorch 的 ATen 和 dispatcher 是 C++ 接口，别的前端可以调 `torch.ops` 或 ATen。CUDA 的 kernel 在二进制层面可复用，但没有任何图语义。这一层差别会在下一节变成提交粒度的差别。

## 一次 eval 把多少工作交给 GPU

上一节停在算子对象上，接下来的问题是，这些节点什么时候、以多大的批次被交给设备。MLX 的最小例子是一次 eval 收三个节点：

```python
x = mx.ones((1024, 1024))
y = mx.matmul(x, x)
z = mx.exp(y)
mx.eval(z)
```

一次 `mx.eval` 会把 `z` 依赖的所有 unscheduled 节点收进来，按节点身份去重，排成拓扑序，再按 stream 编码。

每个 stream 对应一个 `CommandEncoder`（`mlx/backend/metal/device.h:24-26`），encoder 内部持有 Metal 的 queue 和 command buffer（`mlx/backend/metal/device.h:114-118`）。

command buffer 是 Metal 里承载一串已编码 GPU 命令的容器。`commit` 的声明在 `mlx/backend/metal/device.h:101`。

`needs_commit` 在 `mlx/backend/metal/device.h:100`，encoder 用两个计数器判断什么时候必须切 buffer（`mlx/backend/metal/device.h:117-118`）。

这两个计数器是 `buffer_ops_` 和 `buffer_sizes_`。上限来自 `Device::get_max_ops_mb_per_buffer()`（`mlx/backend/metal/device.h:169-171`）。

所以“攒多少”由两方决定：用户调用 `mx.eval` 的边界，以及 encoder 的 ops 与字节阈值。

`mx.eval` 在每个 encoder 结束时走 end_encoding、commit、synchronize 三步。

`Scheduler::enqueue` 按 stream 投递任务（`mlx/scheduler.h:31`），每个 stream 有独立的 `StreamThread`（`mlx/scheduler.h:72`）。

JAX 的一次提交对应一个 XLA executable。没有 `jit` 时，官方文档说代码在“sending one operation at a time to the accelerator”，每个 primitive 单独派发；`jit` 之后，一次调用对应一个编译好的 executable（https://docs.jax.dev/en/latest/jit-compilation.html）。

异步调度让 host 可以继续往下跑，提交出去的工作在设备队列里排队（https://docs.jax.dev/en/latest/async_dispatch.html）。XLA 的 GPU 后端还有一层更低的提交单位。

官方“Tuning GPU command buffers”文档说，XLA 会把一段符合条件的操作 capture 成 GPU graph，之后重复执行时重放这个 graph。capture 是把一段实际发射过的操作录成可重放的图，NVIDIA 上用的是 CUDA Graphs，AMD 上用 HIP graphs（https://openxla.org/xla/command_buffer_tutorial）。

默认最小 capture 区域是 5 个 leaf runtime 操作，可 capture 类别包含 `FUSION`、`CUBLASLT`、`CUDNN`。这个转换跑在 thunk 上，thunk 是 XLA 后端已经决定好如何发射的一个设备操作。JAX 用户直接控制的边界到 executable 为止，内部那一层 graph 不由用户直接控制。

PyTorch eager 的提交单位是一次 kernel launch，一个 aten 算子可能发多个 kernel。kernel 进当前 stream，stream 内按入队顺序执行，host 不等待；跨 stream 的依赖用 event 表达（https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html）。

要让多次 launch 变成一次提交，就用 CUDA Graph。`torch.cuda.CUDAGraph` 或 `torch.compile(backend="cudagraphs")` 都能做这件事，后者在 `torch.compiler` 的总览页里和 inductor、ipex 等并列（https://docs.pytorch.org/docs/2.7/torch.compiler.html）。capture 的边界由用户或后端决定，PyTorch 不会自动把整段 Python 都录进去。

CUDA 有两种提交粒度。不用 graph 时，单位是一次 kernel launch 或 memcpy，stream 是 in-order 队列；用 graph 时，定义、实例化、执行三个阶段分开，`cudaGraphLaunch` 一次提交整张 executable graph（https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html）。

CUDA graph 的节点参数可以在不重建拓扑的情况下更新，这一点把“录制”和“修改”分开了。Metal 的提交粒度是 command buffer：`commit()` 的官方描述是“Submits the command buffer to run on the GPU”（https://developer.apple.com/documentation/metal/mtlcommandbuffer/commit()）。

`enqueue()` 先在 queue 里占位，`MTLCommandQueue` 负责创建、提交和调度 command buffer（https://developer.apple.com/documentation/metal/mtlcommandqueue）。

四者回答的问题并不重合，差别在批次的生命周期。MLX 的 command buffer 活在一次 `mx.eval` 内，下次求值重新编码，头文件里看不到实例化或重放的缓存。

JAX 的 executable 活在一次编译签名内，shape 和 dtype 不变就反复使用，XLA 的 CUDA graph 再在这个 executable 内部跨调用重放。PyTorch 的 kernel launch 活在一次算子调用内，stream 只负责顺序。

CUDA Graph 的 executable 活在一次实例化内，可以跨任意多次 launch 重放。按问题归类，MLX 和 CUDA Graph 都在减少驱动提交次数，JAX 主要在决定编译范围与派发边界，PyTorch 的 stream 在解决依赖顺序。

把 MLX 的 command buffer 直接类比成 CUDA Graph，会漏掉“实例化一次、重放多次”这一步；把 JAX 的 executable 类比成 MLX 的 command buffer，会漏掉 XLA 内部还可能再录一层 graph。这个分层差别会决定下一节的编译产物。

## compile 在 eval 之前改了什么

上一节停在提交单位上，接下来的问题是，编译发生在提交之前还是提交之中。MLX 的最小例子是一个被编译的函数：

```python
@mx.compile
def step(x):
    return mx.exp(mx.matmul(x, x))
```

`mx.compile` 的 C++ 签名返回一个 `std::function`（`mlx/compile.h:13-19`）。std::function 是 C++ 的可调用对象包装，编译结果以这种形式交回调用方。

`CompileMode` 有 disabled、no_simplify、no_fuse、enabled 四档（`mlx/compile.h:10`）。no_simplify 和 no_fuse 两档说明化简和融合都是编译期可单独关掉的 pass。

`compile_impl.h` 把这条流水线拆成四步。

`compile_trace`（`mlx/compile_impl.h:46-50`）负责建图，`compile_dfs`（`mlx/compile_impl.h:55-59`）负责展开 tape 并建立节点到父节点的映射。`compile_dfs` 的注释写明“build a tape and a map of array ids to their parents”。tape 是记录下来的节点顺序。

`compile_simplify`（`mlx/compile_impl.h:61-66`）负责化简，`compile_replace`（`mlx/compile_impl.h:68-73`）负责替换。

另外有按线程的 `CompileCache`（`mlx/compile_impl.h:30-33`），它是编译结果的缓存。

`compile_trace` 和 `CompileCache` 说明第一次调用会建图，之后按函数 id 和输入签名查缓存；缓存键的完整定义头文件没有给出。

这些接口能确定的是，编译改写的是图，产物是一个被缓存的 callable。调用它仍然返回 unscheduled 的 array，还要 `mx.eval` 才会编码执行。编译和求值在这里是前后两件事。

头文件只露出 tape 的化简与替换。融合规则、是否为融合后的图生成新的 Metal 源，头文件里没有连接，这一点查不到。

MLX 确实有运行期编译 Metal 源的通道。`Device::get_library` 可以接收一个返回源码字符串的 builder（`mlx/backend/metal/device.h:173-186`），`backend/metal/jit/indexing.h` 里存着 gather、scatter 等 kernel 的源码模板。

这只能证明模板化 kernel 会在运行期编译，不能证明 `mx.compile` 的逐元素融合走的是同一条路。这一点查不到。

JAX 的编译入口是 `jax.jit`。官方 AOT 文档把一次调用拆成四步：trace 成 jaxpr、lower 成 StableHLO、编译成目标设备的 executable、用数组参数执行（https://docs.jax.dev/en/latest/aot.html）。

融合发生在 XLA 里。XLA 架构文档说 target-independent 阶段会做 operation fusion，GPU 后端还会做针对 GPU 编程模型的 fusion，并决定如何把计算切到 stream 上（https://openxla.org/xla/architecture）。产物是 executable，AOT API 里对应 `lowered.compile()` 返回的对象。

`jit` 按输入 shape 和 dtype 缓存，签名变了就重编。求值阶段仍然异步：`jit` 把编译和执行包在一次调用里，AOT API 把两者拆开，执行本身还是异步 dispatch。

PyTorch 的编译由 `torch.compile` 触发。Dynamo 用 CPython 的 frame evaluation API 改字节码，把一段 PyTorch 操作抽成 FX graph，再交给 backend。

FX graph 是 PyTorch 的 Python 层图表示。PyTorch 的图节点只在编译路径里出现，就是 FX graph 的 node；eager 路径没有这层节点。

guard 是运行时条件检查，条件不满足就重新 capture。遇到不能 capture 的代码就 graph break，回到 eager 执行（https://docs.pytorch.org/docs/2.7/torch.compiler_dynamo_overview.html）。

官方 Dynamo 文档给出 `__compiled_fn_0` 与 `__resume_at_*` 的分段结构，这两段名字对应编译段和恢复段。Triton 是 PyTorch 用来生成 GPU kernel 的语言。默认 backend 是 TorchInductor，GPU 上生成 Triton kernel，CPU 上生成 C++/OpenMP；AOTAutograd 还会把反向传播提前捕获（https://docs.pytorch.org/docs/2.7/torch.compiler.html）。

融合发生在 Inductor 生成 kernel 的这一层。官方 profiling 文档里的 kernel 名会出现 `triton_poi_fused_cat_155` 这样的片段，说明多个算子被合进一个 Triton kernel（https://docs.pytorch.org/docs/2.7/torch.compiler_inductor_profiling.html）。

产物是一段带 guard 的 Python 代码加若干编译好的 kernel，范围到 graph break 为止。调用即执行，没有单独的 eval 阶段，graph break 处回到 eager。

CUDA 和 Metal 的编译发生在另一个阶段。CUDA 的 kernel 由 nvcc 在构建期编成 cubin，运行期的 `cudaGraphInstantiate` 做的是校验、快照和启动结构准备，不做数学改写或融合（https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html）。

Metal 的 kernel 由 Metal 编译器编成 metallib，运行期取 `MTL::ComputePipelineState`。MLX 的模板化 kernel 由运行期编译产生（从 `get_library` 的 builder 和 jit 模板推断），作用域是单个 kernel 的代码生成，图级融合不在这一层。

于是编译和提交在这两个底层里是正交的：编译决定一个 kernel 怎么算，提交决定多少个 kernel 按什么顺序进设备。这个正交关系会在下一节变成可观察的性能边界。

## 这些位置什么时候会让代码变慢或报错

上一节停在编译与提交的分层上，接下来的问题是，这些分层在什么写法下会变成可观察的慢或错误。先看一个最小例子：

```python
for i in range(100):
    loss = step(x).item()
```

`item()` 会触发 `eval()`（`mlx/array.h:562-568`），循环里每次读值都要等 GPU 完成。如果这 100 次调用互不依赖，图本来可以一次收完，这个写法把每次迭代都变成一次提交和一次同步。

JAX 的边界在 shape 上。`jit` 按输入 shape 和 dtype 缓存 executable，shape 变了就重新 trace、重新编译（https://docs.jax.dev/en/latest/jit-compilation.html）。如果在循环里改变序列长度，编译次数会跟着变。

JAX 的 Python 控制流会受 tracer 限制。`if` 或 `for` 的条件如果依赖 tracer 的值，Python 在 trace 时拿不到布尔值，会报错；这类控制流要用 `jax.lax.cond` 或 `jax.lax.scan` 表达。这是从 tracer 的抽象值语义推断的，官方 jit 文档给出了控制流限制。

PyTorch 的边界在 graph break 上。Dynamo 遇到不能 capture 的 Python 代码就停下，回到 eager 执行，后面的代码重新开始 capture（https://docs.pytorch.org/docs/2.7/torch.compiler_dynamo_overview.html）。这类断点会把一张图拆成多段，编译收益下降。

自定义算子如果没有 fake kernel，`torch.compile` 在 trace 时算不出输出 shape，会在 trace 阶段报错。官方 `torch.library` 文档把 `register_fake` 列为编译路径的必要条件（https://docs.pytorch.org/docs/2.7/library.html）。

CUDA Graph 的边界在地址上。capture 之后 graph 记住的是设备地址，输入 tensor 的地址变了就要更新节点参数或重新 capture；graph 的拓扑变化也要重建（https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html）。

MLX 的 compile 缓存按函数 id 和输入签名查（`mlx/compile_impl.h:30-33`）。输入 shape 或 dtype 变了，签名就变了，缓存命中不了，这一点是从缓存键推断的，头文件没有给出完整的键定义。

这些边界有一个共同点：它们都出现在“宿主语言的对象”和“设备上的图”交界处。谁决定什么时候读值、什么时候重编、什么时候重录，谁就决定了这段代码的提交与同步次数。

## 这篇没有回答什么

前面五节回答了对象放在哪、算子放在哪、提交单位是什么、编译改写了什么。还有几类问题没有答案，原因分别是看不到实现体、官方文档没有描述、或者这一篇没有做实测。

MLX 一侧最关键的缺口在实现体。`mx.eval` 的收集、去重、拓扑排序、按 stream 分组都在 `.so` 里，头文件只给出 `gpu::eval(array&)` 和 `array::eval()`。

encoder 的 `needs_commit()` 比较哪些阈值、command buffer 是否在多次 eval 之间复用，也没有从头文件读到。`mx.compile` 的融合规则、融合后是否生成新的 Metal 源、编译缓存按什么键失效，同样只有 tape 接口，没有实现。这里能确认融合在编译期、eval 不做融合，不能确认融合的粒度和产物形态。MLX 的公开文档还提到 `mx.compile` 会生成并编译代码、合并公共工作；这一篇按本机头文件锚点约束，没有把这段文档措辞当作证据，因此融合产物仍然查不到。

JAX 一侧读的是 0.12 代际的文档，没有读 XLA 源码。XLA 的 command buffer 什么时候 capture、capture 后如何处理地址变化，官方文档只给到 flag 和默认值，选择算法在 `command_buffer_conversion_pass.cc` 一类的实现里，这一篇没有展开。JAX eager 路径的 Python 开销、`jit` 的缓存键包含哪些字段，也只有文档层面的描述。

PyTorch 一侧，dispatcher 的实现和 Inductor 的融合决策都没有读源码。FakeTensor 的实现是 Python，文档明确说它在小 tensor 上是 overhead bound，版本之间可能有行为差异。这里引用的 `torch.library` 和 Dynamo 文档能支持“注册了哪些东西”和“guard 怎么工作”，不能支持“某个自定义算子在所有 backend 上都能被编译”。

CUDA 和 Metal 一侧只读了官方文档，没有读驱动或运行时源码。CUDA graph 的更新限制、conditional node 的调度、Metal command buffer 的并发语义，文档有描述但没有实测。Metal 文档没有版本号，Apple 的接口在不同 macOS 和 GPU 代际上可能有差异。

### 所有权与生命周期

有一个维度比这四个问题更值得先问：所有权与生命周期。一张图要被跨调用复用，图里记的设备地址就必须在两次调用之间保持有效，并且不能被别的对象在两次调用之间改写。谁能释放、谁能原地改写，直接决定 command buffer 或 executable 能不能安全重放。

MLX 的 `ArrayDesc` 用 `shared_ptr<Data>` 管缓冲（`mlx/array.h:477`），多个数组可以共享同一块缓冲区。shared_ptr 是 C++ 的引用计数指针，最后一个持有者消失时才释放对象。`siblings` 跟踪多输出节点的兄弟数组，`is_tracer` 标记变换期间不应从图上摘下的节点。共享指针保证节点活着时缓冲区不会被提前释放，这是从头文件字段推断的所有权关系。

但共享指针不能自动解决原地写。多个数组共享同一块缓冲区时，另一个节点还在队列里读这块缓冲区，宿主侧如果对其中一个数组做原地写，队列里的读就可能看到新值。这个场景是从共享缓冲区的结构推断的，头文件没有给出同步规则。

如果缓冲区在队列还在读的时候被释放，就会出现 use after free，也就是释放后仍被读取。`shared_ptr` 只能保证 C++ 对象不析构，不能保证 GPU 队列已经执行完；真正的安全边界是 `synchronize` 或事件。这一点也是从接口推断的，没有实测。

PyTorch 的 Tensor 背后有 Storage（https://docs.pytorch.org/docs/2.7/generated/torch.empty.html），Storage 是持有内存的容器，view 与原地写会改变同一块存储；alias 规则这一篇没有逐条查证。一张 CUDA Graph 要重放，就必须保证 capture 时的地址仍然有效，并且写入顺序与 capture 时一致；否则重放读到的是新值或已释放内存。

JAX 一侧，这一篇只确认了 `jax.Array` 是 future，以及读值要 `block_until_ready()`。donation 是调用方声明输入缓冲区可以被复用，sharding 是把数组切到多设备；这两种机制会把缓冲区的复用和切分交给编译器，接口和生命周期规则没有查证，只能确认它们属于同一类所有权问题。

三种方案对“谁能释放、谁能改写”的回答不同，所以同一张图能否跨调用复用也不同。MLX 把缓冲区的生命周期挂在 `shared_ptr` 上，PyTorch 把内存挂在 Storage 上，JAX 把复用交给编译器的 donation 决策。这个轴比对象模型更接近“能不能安全重放”的根因。

### 其他缺口

自动微分的图表示没有展开。MLX 的 `jvp`、`vjp`、`vmap` 是 `Primitive` 的虚函数（`mlx/primitives.h:80-102`），JAX 的变换规则在 `jax.interpreters.ad` 里，PyTorch 的 autograd 在 dispatcher 的 autograd key 后面。三者都把微分规则挂在算子对象或算子注册上，图结构和缓存策略没有读。

多设备调度也没有展开。MLX 的 `Stream` 有 device 字段（`mlx/stream.h:13-19`），JAX 有 sharding，PyTorch 有 device mesh 和 stream。设备之间的拷贝、同步和集合通信会改变提交单位和编译产物，这一篇只讲了单设备路径。

最后是测量。这一篇没有给出任何一条路径的耗时，也没有给出融合前后的 kernel 数量。分派开销、capture 收益、编译缓存命中率都只有结构描述，没有实测数字。这些数字需要在本机用 profiler 取，不能从文档推出来。
