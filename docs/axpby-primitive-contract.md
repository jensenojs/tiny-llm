# Axpby：前端、CPU 与 Metal 执行之间的契约

Axpby 计算逐元素乘加 `out = alpha*x + beta*y`。本讲义解释这项计算如何从 Python 调用进入 MLX 计算图，再由 CPU 或 Metal 后端执行：前端确定输入类型与广播形状，后端分配输出缓冲、解释输入步长，并将每个输出元素交给相应的执行代码。

Metal 部分围绕两项设计展开。输入具有共同的连续遍历顺序时，三块缓冲可以共用一个下标；转置、切片和广播形成其他布局时，kernel 根据 shape 与 strides 换算输入地址。主机随后绑定缓冲和参数、指定线程总数与组尺寸，kernel 使用线程位置计算对应元素。末尾用访问地址推导和基准测试，分别检查布局如何改变读取位置、完整调用实际耗时多少。

`mx::array` 在惰性图中保存形状、类型、输入依赖与计算操作；`Primitive` 提供后端执行及变换规则。求值时，MLX 沿输入依赖执行操作，本例的后端函数负责分配输出数据。源码位于 `src/extensions/src/axpby.h`、`src/extensions/src/axpby.cpp` 和 `src/extensions/src/axpby.metal`；后文的复核命令用于检查这些实现的行为。

后文使用以下术语描述数组布局：

| 术语 | 含义 |
| --- | --- |
| 逻辑元素 | 把数组按行优先摊平之后的第 `i` 个。`z[i] = alpha*x[i] + beta*y[i]` 里的 `i` 就是它 |
| 缓冲下标 | 这个逻辑元素落在物理缓冲的第几格。读写发生在这一层 |
| strides | 每一维走一格，缓冲下标加多少。二维行优先 `(4,1)` 读作「换行 +4，换列 +1」 |
| 广播 | 将长度为 1 的维度扩展为所需长度；共享缓冲的视图可用 0 步长表达重复读取 |

连续 kernel 用同一个缓冲下标 `i` 索引 x、y、out。它要求三者按该下标访问时，指向同一个输出坐标对应的三个元素。当前实现通过共同的行连续或列连续布局满足这个前提；有间隔的切片与重复元素的广播需要按步长寻址。

## 复习：一维缓冲、strides 与广播的 0 步长

后面的连续性判据与 `elem_to_loc` 都建立在这一节上，已经熟的部分可以跳过。

数组的数据缓冲是一维存储，形状与步长共同确定各坐标落在哪个位置。下面用 `(4,4)` 数组比较行优先与列优先两种连续布局。

取 `x = mx.arange(16).reshape(4,4)`，逻辑上它是

```plain
     列0  列1  列2  列3
行0    0    1    2    3
行1    4    5    6    7
行2    8    9   10   11
行3   12   13   14   15
```

行优先（row-major，MLX 与 numpy 的默认）先在缓冲里排完第 0 行，再排第 1 行：

```plain
缓冲下标:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
缓冲内容:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
                    └─ 行0 ─┘ └─ 行1 ─┘ └─ 行2 ─┘
```

列优先（column-major，Fortran 与 BLAS 的默认）先排完第 0 列再排第 1 列：

```plain
缓冲下标:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
缓冲内容:  0  4  8 12  1  5  9 13  2  6 10 14  3  7 11 15
                    └─ 列0 ──┘ └─ 列1 ──┘ └─ 列2 ──┘
```

两种布局用同一套读法：每个维度配一个步长，沿这一维走一格，缓冲下标加多少。

```plain
x[i][j] = 缓冲[ i*strides[0] + j*strides[1] ]
```

行优先是 `(4,1)`：换行跳 4 格（跨过一整行），换列跳 1 格（相邻）。列优先是 `(1,4)`：换行 1 格，换列 4 格。取 `i=2, j=1`，行优先算得 `2*4+1=9`，列优先算得 `2*1+1*4=6`；缓冲本身也是按各自的规律填的，所以两种算法读到的都是同一个值 `9`。

转置的定义是 `xt[i][j] = x[j][i]`。对上述二维连续输入，交换两个轴的形状与步长即可用共享缓冲表达转置：`x` 的 `(4,1)` 变成 `xt` 的 `(1,4)`。MLX 的惰性图记录转置操作，求值时由后端建立对应的数据视图。

广播同样靠改步长。`y` 形状 `(1,4)`、步长 `(4,1)`，广播到 `(3,4)` 之后步长成 `(0,1)`：沿第 0 维走多少格，下标都加 0。

```plain
y[i][j] = 缓冲[ i*0 + j*1 ] = 缓冲[ j ]
```

```plain
y[0][2] = 缓冲[2] = 30
y[1][2] = 缓冲[2] = 30
y[2][2] = 缓冲[2] = 30
```

第 0 维的不同坐标均贡献 0 偏移，缓冲里的四个数被三个逻辑行共享。`elem_to_loc` 对该维计算 `(elem % shape[i]) * 0 = 0`；general kernel 依靠同一套步长寻址规则表达这种共享关系。

四种视图放一张表，后面反复要用：

| 视图 | 形状 | strides | 缓冲里发生了什么 | row_contiguous | col_contiguous |
| --- | --- | --- | --- | --- | --- |
| `x = arange(16).reshape(4,4)` | (4,4) | (4,1) | 16 个数一一对应 | 真 | 假 |
| `x.T` | (4,4) | (1,4) | 同一块缓冲，换个读法 | 假 | 真 |
| `x8[::2]`（`x8` 是 `(8,4)`） | (4,4) | (8,1) | 中间空 4 格，`x8` 的奇数行不在视图里 | 假 | 假 |
| `zeros((1,4))` 广播到 `(3,4)` | (3,4) | (0,1) | 4 个数被 12 个逻辑位置共享 | 假 | 假 |

## 三个文件的分工由编译单元决定

`axpby.metal` 编译成 metallib，作为一个资源嵌进 dylib 旁边。它只认识 Metal 语言，看不见 `mx::array`、`stream()`、`alpha_`。`axpby.cpp` 进 dylib，能看见这些，却看不见着色器源码里的参数表。`bindings.cpp` 用 nanobind 把 C++ 符号注册成 Python 名字，解释器侧只存在这一个名字。

三件产物由 `src/extensions/CMakeLists.txt` 分别声明：`add_library(tiny_llm_ext)` 编五个 `.cpp` 并链上 `mlx`；`mlx_build_metallib(...)` 把五个 `.metal` 连同 MLX 的头文件目录编成 `tiny_llm_ext.metallib`；`nanobind_add_module(_ext ...)` 生成 `_ext.cpython-310-darwin.so`，`rpath` 指向 `@loader_path`。`tiny_llm_ext/__init__.py` 在 import 时调一次 `load_library`，后者把 metallib 按 `"tiny_llm_ext"` 这个名字登记进 MLX 的 GPU device（`src/extensions/src/utils.cpp:12-15`）；`eval_gpu` 里的 `d.get_library("tiny_llm_ext")` 取的正是这一份。

名字在这里已经对过一次：注册用的字符串与取用时的字符串是两处独立的字面量，写错的表现是运行期报找不到库。

主机代码与 Metal 函数分别编译，调用时通过参数槽位对应数据。`[[buffer(0)]]` 声明 kernel 的第 0 个缓冲参数，`set_input_array(x, 0)` 将 x 绑定到该槽位。当前实现由两侧源码分别声明编号和类型，匹配关系需要通过源码核对与运行测试验证。

## 前端构造计算图并确定类型与形状

```cpp
mx::array axpby(const mx::array &x, const mx::array &y, const float alpha, const float beta,
                mx::StreamOrDevice s) {
    auto promoted_dtype = promote_types(x.dtype(), y.dtype());
    auto out_dtype = mx::issubdtype(promoted_dtype, mx::float32)
                         ? promoted_dtype
                         : promote_types(promoted_dtype, mx::float32);
    auto x_casted = mx::astype(x, out_dtype, s);
    auto y_casted = mx::astype(y, out_dtype, s);
    auto broadcasted_inputs = broadcast_arrays({x_casted, y_casted}, s);
    auto out_shape = broadcasted_inputs[0].shape();
    return mx::array(out_shape, out_dtype,
                     std::make_shared<Axpby>(to_stream(s), alpha, beta),
                     broadcasted_inputs);
}
```

前端先计算输出类型和形状，再用图节点表达所需的转换与计算。`astype` 在输入 dtype 等于目标 dtype 时返回原数组；类型不同时构造 `Cast` 操作，求值时执行数据转换。

`broadcast_arrays` 先求共同形状，再为需要扩展的输入构造广播操作；已经具有目标形状的输入可以原样返回。广播操作在求值时可以通过共享输入缓冲、将扩展维度的 stride 设为 0 来表达重复读取。图上存在广播节点与执行时共享缓冲是两个可以同时成立的事实。本机用 `mx.export_to_dot` 导出 `mx.broadcast_to(mx.zeros((1,4)), (3,4))` 的图，能直接看到 `Broadcast` 节点。

`mx::array(shape, dtype, primitive, inputs)` 在前面的类型转换与广播操作之后构造 Axpby 输出节点。`primitive` 指向一个刚造的 `Axpby`，保存两个标量 `alpha_`、`beta_` 和一个 stream。输出形状与 dtype 到此确定，输出数据缓冲由后端求值函数分配，实际 strides 也由后端分配策略确定。

`promote_types` 与 `issubdtype` 决定输出 dtype。下表实测显式将两个 `mx.ones` 输入和 Axpby 调用都指定到 CPU，以分别观察建图与 CPU 求值：

| 输入 dtype | 输出 dtype | 原因 |
| --- | --- | --- |
| float32 | float32 | `issubdtype` 为真，原样保留 |
| float16 | float32 | `issubdtype` 为假，与 float32 提升 |
| bfloat16 | float32 | 同上 |
| int32 | float32 | 同上，整数被抬到浮点 |
| complex64 | complex64 | 与 float32 提升得 complex64 |
| float64 | 建图得到 float64，CPU 求值报错 | 当前 `eval_cpu` 分派未覆盖 float64，抛出 `RuntimeError: Axpby is only supported for floating point types.` |

前端将 float16 与 bfloat16 提升为 float32，complex64 保持为 complex64。float64 说明前端可表达的输出类型与当前后端实际支持的类型需要分别检查。若创建输入时沿用默认 GPU，float64 输入本身就会触发 `ValueError: float64 is not supported on the GPU`；该错误发生在输入创建阶段。

## 广播做两件事：校验形状，把步长写成 0

`broadcast_arrays` 使用的形状合并函数在 MLX 头文件中声明为：

```cpp
// include/mlx/utils.h:108
MLX_API Shape broadcast_shapes(const Shape &s1, const Shape &s2);
```

`broadcast_shapes` 从末维向前对齐两个 shape；对应维度长度相同或其中一个为 1 时可以合并，缺失的前导维度按 1 处理。对应维度长度不同且两者都大于 1 时抛出异常。实测（mlx 0.32.4，本机）：

```plain
>>> mx.broadcast_arrays(mx.zeros((3,4)), mx.zeros((2,4)))
ValueError: [broadcast_shapes] Shapes (3,4) and (2,4) cannot be broadcast.
```

这里的两个末维都为 4，可以对齐；倒数第二维分别为 3 与 2，且都大于 1，因此形状合并失败。相同维数的输入也会发生广播冲突，维数不同的输入则可以通过前导维补 1 合并。

顺带一个接口差异，写 C++ 时会踩：C++ 侧签名是 `std::vector<array>`（`include/mlx/ops.h:432-435`），Python 侧是变参 `broadcast_arrays(*arrays)`。把 list 传给 Python 版本不会得到形状错误，得到的是 `RuntimeError: std::bad_cast`——参数类型在绑定层就没进去。

广播阶段负责形状校验与形状对齐，`eval_gpu` 开头的 bool 则依据求值后输入的布局选择 kernel。广播将长度大于 1 的扩展维度表示为 0 步长时，多个逻辑元素共享一个缓冲位置，这种输入需要 `general` 的步长寻址。最终选路以输入实际 flags 为准；两条 kernel 各自在其适用布局下实现同一项逐元素运算。后面的测量比较整体调用耗时，整数寻址与访存布局的独立成本仍需隔离实验。

## 连续性判据：三块缓冲共享一种扁平顺序

`eval_gpu` 开头的分流：

```cpp
bool contiguous_kernel = (x.flags().row_contiguous && y.flags().row_contiguous) ||
                         (x.flags().col_contiguous && y.flags().col_contiguous);
```

两个 flag 的定义在 `include/mlx/array.h`：

```cpp
bool row_contiguous : 1;   // strides[-1]==1 且 strides[i]==shape[i+1]*strides[i+1]
bool col_contiguous : 1;   // strides[0]==1  且 strides[i]==shape[i-1]*strides[i-1]
```

取 `x` 为 `(4,4)` 的行优先数组，`strides=(4,1)`；`x.T` 交换两个轴的形状与步长，共享原缓冲，`strides=(1,4)`。

- `x`：`row_contiguous` = `strides[1]==1` 成立，且 `strides[0]==shape[1]*strides[1]` 即 `4==4`，为真；`col_contiguous` 要求 `strides[0]==1`，为假。
- `x.T`：`row_contiguous` 要求 `strides[1]==1`，`strides[1]=4`，为假；`col_contiguous` 要求 `strides[0]==1`（成立）且 `strides[1]==shape[0]*strides[0]` 即 `4==4`（成立），为真。

两个条件分别是 `(false && …)` 与 `(… && false)`，都不成立，于是 `axpby(x.T, x, 1.0, 2.0)` 走 **general** 分支，`kname` 拼成 `axpby_general_float32`。

判据的形状由连续 kernel 的主体决定，那个 kernel 的整个函数体只有一句：

```metal
out[index] = static_cast<T>(alpha) * x[index] + static_cast<T>(beta) * y[index];
```

它拿一个扁平下标同时索引三块缓冲。这在语义上成立，当且仅当三块缓冲的**扁平顺序**指向同一个逻辑元素。行优先与行优先同序，列优先与列优先也同序，行列混用不同序，于是条件写成两个「与」再取「或」。

这个 bool 的语义因此落在「共享一种遍历顺序」上，与「各自连续」是两个命题。一个列优先的数组自己是连续的，与一个行优先的数组放进同一个调用就不同序，条件照样不成立。`axpby(x.T, x, …)` 正是这一情形：`x.T` 的 `col_contiguous` 为真、`row_contiguous` 为假，`x` 恰好相反，两个「与」都没有两边的真。

在本节各维长度均大于 1 的二维示例中，行连续要求相邻列相隔一个元素、相邻行首尾相接；列连续则交换两个方向。以行连续为例，内层条件是 `strides[-1]==1`，外层条件是 `strides[i]==shape[i+1]*strides[i+1]`。切片和广播分别形成以下布局：

| 破坏方式 | 产生原因 | 例子 | 断在哪条 |
| --- | --- | --- | --- |
| 有格子不属于任何逻辑元素 | 切片跳过 | `x8[::2]`，shape `(4,4)`，strides `(8,1)`，中间空 4 格 | `8 == 4` 为假 |
| 一个格子属于多个逻辑元素 | 广播把步长写 0 | `(1,4)` 广播到 `(3,4)`，strides `(0,1)`，三行共享同一行数据 | `0 == 4` 为假 |

两种都让 `x[index]` 取到错的格子。`general` kernel 的 `elem_to_loc` 用步长重算下标，才取得到正确位置：上面 `x8[::2]` 的第 4 个逻辑元素的缓冲下标是 `8`，而 `x8[1][0]` 落在下标 `4`。

输出也被绑进了这个前提，只是不在条件里：

```cpp
out.set_data(mx::allocator::malloc(x.data_size() * out.itemsize()),
             x.data_size(), x.strides(), x.flags());
```

`out` 采用 `x` 的 strides 与 flags，也就是说 `out` 的扁平顺序**跟着 x 走**。行优先分支里 `out` 是行优先，列优先分支里 `out` 是列优先——那个分支下 `out[index]` 写的是第 index 个列优先元素。这解释了为什么条件只需要比较 `x` 与 `y`：`out` 已经和 `x` 同序了。

本节非空二维连续示例的 `data_size()` 等于逻辑元素数，输出据此分配并继承 x 的 strides 与 flags。长度大于 1 的维度采用 0 步长时，多个逻辑位置共享缓冲元素，需要 general 寻址；长度为 1 的轴本身只有一个坐标，讨论连续性时还需考虑这种退化情形。实际分流以 MLX 提供的 flags 为准。

## 一条 general kernel 的偏移算术

general 分支多压四个参数（shape、x 的 strides、y 的 strides、ndim），kernel 里换掉那唯一的一行：

```metal
// include/mlx/backend/metal/kernels/utils.h:97-109
template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc(IdxT elem, constant const int* shape,
                            constant const int64_t* strides, int ndim) {
  IdxT loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  return loc;
}
```

就是逐维取余再除，把扁平下标拆成多维坐标，乘上各维步长累加。拿上面那个转置的例子手算 `index=6`、shape `(4,4)`、strides `(1,4)`：第一轮 `6 % 4 = 2`，`loc += 2*4 = 8`，`elem` 变 `1`；第二轮 `1 % 4 = 1`，`loc += 1*1`，得 `9`。逻辑上的第 6 个元素是第 1 行第 2 列，在转置前的缓冲里位置是 `2*4+1 = 9`，对上。

CPU 分支用的是同一个算法，少一个 `ndim` 参数（`include/mlx/backend/common/utils.h:16-25`）。两处独立实现同一个式子，是因为编译单元不同：一份在 Metal 方言里，一份在主机 C++ 里。

代价是每元素若干次除法。MLX 自己的逐元素算子不这么干，它先用 `collapse_contiguous_dims`（`include/mlx/backend/common/utils.h:48-51`）把相邻的连续维度折叠成一个，再走少维 kernel。`axpby` 作为示例没做这层优化，代价可测但不影响正确性。

## CPU 路径：模板函数与队列中的计算闭包

`axpby_impl` 是 CPU 后端的模板函数，按 dtype 分派给 float、float16、bfloat16 或 complex64 实例。函数分配输出、登记输入输出，再将逐元素循环封装成闭包交给 CPU encoder：

```cpp
auto &encoder = mx::cpu::get_command_encoder(stream);
encoder.set_input_array(x);
encoder.set_input_array(y);
encoder.set_output_array(out);
encoder.dispatch([x_ptr = x.data<T>(), y_ptr = y.data<T>(), out_ptr = out.data<T>(),
                  size = out.size(), shape = out.shape(),
                  x_strides = x.strides(), y_strides = y.strides(), alpha_, beta_]() { ... });
```

`encoder.dispatch` 接收闭包并排入 CPU 执行队列。闭包按值捕获三个数据指针及形状、步长、标量参数，`set_input_array` 与 `set_output_array` 则向 encoder 登记输入输出，使后端能管理异步执行涉及的资源与依赖。

CPU 与 GPU 都完成输入寻址、乘加和输出写回。当前 CPU 实现按行优先分配输出，并统一调用 `elem_to_loc` 计算输入偏移；GPU 实现另设直接线性访问的连续分支。CPU 的实际访存效率仍受输入布局影响。

## GPU 路径：三张表

第一张是名字。`kname` 由两个维度拼出来：分支（`contiguous` / `general`）与 dtype 名（`type_to_name(out)`，得 `float32` 这类字符串）。四组 dtype 乘两个分支八条名字，全部由 `axpby.metal` 末尾的宏展开提前实例化：

```metal
#define instantiate_axpby(type_name, type)                             \
  instantiate_kernel("axpby_general_" #type_name, axpby_general, type) \
  instantiate_kernel("axpby_contiguous_" #type_name, axpby_contiguous, type)

instantiate_axpby(float32, float);
instantiate_axpby(float16, half);
instantiate_axpby(bfloat16, bfloat16_t);
instantiate_axpby(complex64, complex64_t);
```

C++ 侧用字符串拼名字（`d.get_kernel(kname.str(), library)`），Metal 侧用预处理拼名字，两边靠字符串字面量对齐。写错一个字母的表现是运行期抛异常说找不到 kernel，而不是编译错误——这是上一节说的「手写契约」的第一处代价。

第二张是 buffer 槽位。

| 槽位 | `axpby.cpp` 侧 | `axpby.metal` 侧 | 何时 |
| --- | --- | --- | --- |
| 0 | `set_input_array(x, 0)` | `device const T* x` | 总是 |
| 1 | `set_input_array(y, 1)` | `device const T* y` | 总是 |
| 2 | `set_output_array(out, 2)` | `device T* out` | 总是 |
| 3 | `set_bytes(alpha_, 3)` | `constant const float& alpha` | 总是 |
| 4 | `set_bytes(beta_, 4)` | `constant const float& beta` | 总是 |
| 5 | `set_vector_bytes(x.shape(), 5)` | `constant const int* shape` | 仅 general |
| 6 | `set_vector_bytes(x.strides(), 6)` | `constant const int64_t* x_strides` | 仅 general |
| 7 | `set_vector_bytes(y.strides(), 7)` | `constant const int64_t* y_strides` | 仅 general |
| 8 | `set_bytes(ndim, 8)` | `constant const int& ndim` | 仅 general |

`constant const int*` 看着像设备指针，实际上 `setBytes` 是把主机内存里的这几个整数**拷进命令缓冲**（`include/mlx/backend/metal/device.h:63-86`），kernel 读的是命令缓冲内嵌的那份。所以传 `std::vector` 的临时副本是安全的，不需要它活到执行结束。也正因为它们是一个字节一个字节抄进去的，槽位号与类型必须两侧一致：类型不匹配不会在 C++ 编译期被发现。

第三张是 dtypes 与分支的组合。`axpby.metal` 实例化八条，`eval_cpu` 的分派表四条（无整数），`contiguous_kernel` 的判据决定用哪一半。这三张表没有任何一处由编译器替你校核。

## 线程模型：三维的坐标，一维的编号

Metal 的模型分三层，三个内建变量直接暴露：

- **线程 grid**：这次派发一共多少线程，编号 `thread_position_in_grid`，就是 kernel 里的 `uint index`；
- **线程组 threadgroup**：硬件调度与共享内存的单位，编号 `threadgroup_position_in_grid`；
- **组内线程序号**：`thread_position_in_threadgroup`。

派发尺寸按 `(width,height,depth)` 描述三个方向的范围，kernel 可用 `uint3` 接收位置，也可像本例一样用标量接收一维派发的位置。三维线程数为三个范围长度的乘积。为了将三维坐标写入一维缓冲，下面的探针自行选择 x 方向相邻坐标对应相邻槽位的映射：

```plain
lin = x + W * (y + H * z)          W、H 取 grid 的宽与高
```

探针把线程坐标与归属写回缓冲。grid 填 `(4,3,1)`、组填 `(2,2,1)`，程序按自选的 `lin = gid.x + 4*gid.y` 排列输出：

| lin | gid.x | gid.y | tgp.x | thread_in_tg.x |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 0 | 1 |
| 2 | 2 | 0 | 1 | 0 |
| 3 | 3 | 0 | 1 | 1 |
| 4 | 0 | 1 | 0 | 0 |
| 5 | 1 | 1 | 0 | 1 |
| 6 | 2 | 1 | 1 | 0 |
| 7 | 3 | 1 | 1 | 1 |
| 8 | 0 | 2 | 0 | 0 |
| 9 | 1 | 2 | 0 | 1 |
| 10 | 2 | 2 | 1 | 0 |
| 11 | 3 | 2 | 1 | 1 |

按这个缓冲映射，x 每增加 1，`lin` 增加 1；y 每增加 1，`lin` 增加 4。沿 x 方向每跨过两个位置，`tgp.x` 增加 1。另一次派发取 `grid=(12,1,1)`、组 `(4,1,1)` 时，x 范围变成 0 至 11，`tgp.x` 依次是 0,0,0,0,1,1,1,1,2,2,2,2。两次派发都包含 12 个逻辑线程，但坐标与分组不同。表格顺序来自缓冲布局和 CPU 打印循环，GPU 的实际执行先后需要另行观测。

`axpby` 是一维问题，主机侧 `MTL::Size(nelem, 1, 1)` 即 `W = nelem`、`H = D = 1`，代入公式得 `lin = x + nelem * (0 + 0) = x`。所以 `uint index` 声明成标量时取到的是 `.x` 分量，而 `.x` 在这个 grid 里恰好就是线性编号，不必再乘加。矩阵乘那类二维 kernel 声明 `uint3`，自己写 `gid.x + grid.x * gid.y`，因为两个分量都要用。

本机 M4 的 limits 探针打印设备 `maxThreadsPerThreadgroup=(1024,1024,1024)`、所建 pipeline 的 `maxTotalThreadsPerThreadgroup=1024` 与 `threadExecutionWidth=32`。设备属性约束单组各方向尺寸，pipeline 属性约束单组线程总数。SIMD 组宽度为 32；组大小取其整数倍可以避免完整组因数量未整除而产生的尾部不足，具体性能还取决于寄存器、共享内存和工作负载。

`axpby` 的派发是三行：

```cpp
size_t tgp_size = std::min(nelem, kernel->maxTotalThreadsPerThreadgroup());
MTL::Size group_dims = MTL::Size(tgp_size, 1, 1);   // 每组多少线程
MTL::Size grid_dims  = MTL::Size(nelem, 1, 1);      // 一共多少线程
compute_encoder.dispatch_threads(grid_dims, group_dims);
```

`min` 使请求的组大小符合当前 pipeline 的总线程上限。本机未开启 API validation 的探针中，请求组大小 2048 时 `cb.error` 为空，前 16 个观测槽位均保持初值。开启 `MTL_DEBUG_LAYER=1` 后，同一请求在编码时触发断言 `2048 must be <= 1024. (device threadgroup size limit)`，进程以 SIGABRT 结束。合法尺寸应由查询到的上限保证；这两种运行模式说明非法请求的暴露方式受验证配置影响。

下界那一半不是必须（实测 `grid=12, group=1024` 被驱动钳成一组 12 线程、`tgSize=12`，`grid=1000, group=1024` 钳成 1000），但它让请求的组大小等于实际工作量，读代码时不必依赖驱动的钳位。钳位只发生在 group 大于 grid 的方向上。

### 派发语义：给线程数还是给组数

两个函数最容易和 CUDA 混淆。量它只需要一个内核函数和一次回读，下面的探针直接编成可执行文件（`clang++ -fobjc-arc -std=c++17 -framework Metal -framework Foundation probe.mm -o probe`），不建工程、不出 metallib：

```objc
#import <Metal/Metal.h>
#include <cstdio>

static const char* kSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;
[[kernel]] void probe(device int* out [[buffer(0)]],
                      uint gid [[thread_position_in_grid]],
                      uint tgs [[threads_per_threadgroup]]) {
  if (gid < 16) { out[gid] = int(tgs); }
}
)METAL";

static void run(id<MTLDevice> dev, id<MTLCommandQueue> q,
                id<MTLComputePipelineState> ps, unsigned grid, unsigned group) {
  id<MTLBuffer> buf = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
  int* o = (int*)[buf contents];
    for (int i = 0; i < 16; ++i) o[i] = -1;            // 哨兵初值，kernel 写入组大小
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:ps];
  [enc setBuffer:buf offset:0 atIndex:0];
  [enc dispatchThreads:MTLSizeMake(grid,1,1) threadsPerThreadgroup:MTLSizeMake(group,1,1)];
  [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
  int written = 0;
  for (int i = 0; i < 16; ++i) if (o[i] >= 0) ++written;
  printf("grid=%-5u group=%-5u err=%-3s 被写槽位=%2d  o[0]=%d\n",
         grid, group, cb.error ? "yes" : "no", written, o[0]);
}

int main() { @autoreleasepool {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> q = [dev newCommandQueue];
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                         options:nil error:&err];
  id<MTLComputePipelineState> ps =
      [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"probe"] error:&err];
  run(dev, q, ps, 4096, 1024);
  run(dev, q, ps, 4096, 2048);
  run(dev, q, ps, 1000, 1024);
  run(dev, q, ps, 10, 4);
} return 0; }
```

对象层次在这里一一对应：`MTLCreateSystemDefaultDevice` 选 GPU，`newCommandQueue` 开一条提交通道，`newLibraryWithSource` 在运行时编译那段 MSL 文本，`newComputePipelineStateWithFunction` 把函数编成可执行的 pipeline，`newBufferWithLength:options:MTLResourceStorageModeShared` 拿一块 CPU 与 GPU 都能访问的内存，kernel 写完主机直接读 `[buf contents]`。

初值 `-1` 是哨兵，`o[i] >= 0` 表示该槽位被写过；`written` 统计的是 16 个槽位中的实际写入数。`gid < 16` 将写入限制在已分配的 64 字节内。这个探针同时观察缓冲和 `cb.error`，因为本机普通运行时，非法组尺寸曾出现错误属性为空而观测槽位均未写入的情况。观测覆盖前 16 个线程位置，整次派发的执行数量和并发程度需要其他证据。

四组请求与一组 `dispatchThreadgroups` 对照（本机 M4）：

```plain
[A] dispatchThreads,       grid=12   线程, group=4      -> 12 线程, 3 组, 每组 4 个, tgSize=4
[B] dispatchThreads,       grid=10   线程, group=4      -> 10 线程, 3 组, 末组 2 个线程
[C] dispatchThreadgroups,  grid=3    组,   group=4      -> 12 线程, 3 组, 每组 4 个, tgSize=4
[D] dispatchThreads,       grid=1000 线程, group=1024   -> 一组, 驱动把组钳成 1000 线程
[E] dispatchThreads,       grid=4096 线程, group=2048   -> 普通运行 err=no，前 16 槽均未写
[E'] dispatchThreadgroups, grid=2    组,   group=2048   -> 普通运行 err=no，前 16 槽均未写
```

B 那一行是整个线程模型的关节点：`dispatchThreads` 的第一个参数是**线程数**，最后不足一组的那些线程照样跑，末组变小（4,4,2）。`dispatchThreadgroups` 的第一个参数是**组数**，一启动就是整数个满组，凑不出整除时必须多算。D 那一行说明钳位的方向：请求的组比线程总数还大时，驱动把组压到总数上，不会多造线程。

把这几行翻成 CUDA 的词汇，表格如下。

| Metal | CUDA | 语义 |
| --- | --- | --- |
| thread | thread | 最小执行单元 |
| threadgroup | block | 同组线程可共享内存、barrier 同步 |
| threads_per_grid（总数） | gridDim × blockDim | `dispatchThreads` 给总数 |
| threadgroups_per_grid（组数） | gridDim | `dispatchThreadgroups` 给组数 |
| `thread_position_in_grid` | `blockIdx.x*blockDim.x + threadIdx.x` | 全局编号 |
| `thread_execution_width` = 32 | warp size = 32 | 同一条指令推进的线程数 |

CUDA 的 `<<<grid, block>>>` 与 Metal 的 `dispatchThreadgroups` 都给出组数和组大小。当一维元素数 N 无法被组大小整除、需要向上取整派发时，一线程处理一个元素的 kernel 应在访问前执行 `if (i >= N) return;`。当前 axpby 使用 `dispatchThreads` 精确给出 `nelem` 个线程位置，末组可以不足，因而 `index` 落在 `[0,nelem)`。改用按组派发时，整除的工作量可以恰好覆盖，未整除的工作量需要尾部保护。

把数字对一遍：`nelem = 12` 时 `tgp_size = 12`，一个组、12 个线程；`nelem = 5000` 时 `tgp_size = 1024`，四个满组加一个 904 线程的末组，仍然是 5000 个线程、每个元素一次。

## 转置输入的访问地址、缓存复用与耗时

取脚本中的 `N=4096`、float32 输入。general kernel 按行优先编号输出，线程 `index=r*N+c` 处理输出坐标 `(r,c)`。x 保持行连续，y 使用转置视图，两个输入的元素偏移分别为：

```text
x_offset   = r*N+c
y.T_offset = c*N+r
```

处理输出第 0 行前 32 列的线程 `index=0..31`，对 x 读取偏移 `0,1,..,31`，对 y 读取偏移 `0,4096,..,31*4096`。float32 每元素 4 字节，前者地址相邻，后者相邻请求相差 16384 字节。这个地址关系由输入 strides 和 kernel 寻址式确定。general kernel 也能处理行内连续的切片，实际地址分布由具体布局决定。

固定输出列 `c=5`，检查输出第 0 至第 7 行：

| 输出坐标 | 线程 index | y 原缓冲的元素偏移 | 相对缓冲起点的字节偏移 |
| --- | --- | --- | --- |
| (0,5) | 5 | 20480 | 81920 |
| (1,5) | 4101 | 20481 | 81924 |
| (2,5) | 8197 | 20482 | 81928 |
| (3,5) | 12293 | 20483 | 81932 |
| (4,5) | 16389 | 20484 | 81936 |
| (5,5) | 20485 | 20485 | 81940 |
| (6,5) | 24581 | 20486 | 81944 |
| (7,5) | 28677 | 20487 | 81948 |

这八个线程必然读取相邻的八个元素。完整派发中，`(r,c) → c*N+r` 是原缓冲所有元素的一次排列，每个 y 元素恰好被一个输出线程读取。跨线程复用发生在包含多个元素的缓存块上：某次读取将相邻元素一起带入缓存，其他线程读取这些不同元素时可以使用同一缓存块。

原先的八倍估算使用如下假设模型：缓冲起点按 32 字节对齐，考察某一层缓存中按 32 字节划分的访问单位，每次未命中都完整获取一个单位；统计区域中的数据初始均未驻留，获取仅由这些读取引起。连续的 32 个 float32 占 128 字节，覆盖 4 个单位；第 0 行对转置 y 的 32 次读取覆盖 32 个单位。两者覆盖数量之比是 `32/4=8`，对应这一批请求涉及的分段数量。

同一模型中，上表八次读取落在同一个 32 字节单位内，分别消费其偏移 0、4、8、12、16、20、24、28 处的元素。将列 c 扩展到全部 4096 列，输出前八行总计读取 32768 个 y 元素，覆盖 4096 个不同的 32 字节单位，有效数据为 `4096×32=128 KiB`。每个单位获取一次时流量为 128 KiB；八次读取各自重新获取时流量为 `8×128 KiB=1 MiB`。

推广到整个矩阵，y 的有效数据为 `4096²×4=64 MiB`，包含 2097152 个这样的分段。每段获取一次对应 64 MiB，每个元素都重新获取所在分段对应 512 MiB。后一种情形要求同块的其他元素被读取时，该块又需要从所考察接口的下一级获取。调度次序、缓存容量及同期访问都可能影响块的驻留。

每个输出同时需要读 x、读 y、写 out。将三项放在同一个接口处计数，并进一步假设 x 读取和 out 写入各传输 64 MiB，完整计算的账目为：

| 同一接口处的假设传输量 | 读 x | 读 y | 写 out | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 连续组，每块获取一次 | 64 MiB | 64 MiB | 64 MiB | 192 MiB |
| 转置组，y 每个元素都重新获取整块 | 64 MiB | 512 MiB | 64 MiB | 640 MiB |

y 在连续组中占总量的三分之一；仅将这一项放大八倍，完整流量比为 `(1+8+1)/(1+1+1)=10/3≈3.33`。一般地，若 y 在该接口处的完整读取量放大 A 倍，x 和 out 保持上述假设，则总流量比为 `(2+A)/3`。这些数值描述模型规定的获取情形，实际读写事务还取决于所考察层级的规则。

读取哪些元素已经由地址公式完全确定。块是否驻留、由哪层缓存服务及实际传输多少字节，需要硬件资料或性能计数器支持。32 字节单位在这里是模型前提；当前材料尚未给出 M4 对应访存层级的单位大小与事务规则，原先“M4 的 DRAM 最小访问单位为 32 字节”的说法应撤回。线程 index 的大小也不规定跨组执行顺序。

下面的纯地址计算可在仓库根目录复核；它验证表格及分段计数：

```sh
pdm run python - <<'PY'
N = 4096
for r in range(8):
    offset = 5*N+r
    print(r, r*N+5, offset, 4*offset, (4*offset)//32)
assert len({(4*(5*N+r))//32 for r in range(8)}) == 1
assert len({(4*(c*N))//32 for c in range(32)}) == 32
assert len({(4*c)//32 for c in range(32)}) == 4
segments = {(4*(c*N+r))//32 for r in range(8) for c in range(N)}
assert len(segments) == 4096
assert len(segments)*32 == 128*1024
assert 8*N*32 == 1024*1024
PY
```

### 从传输量到 GPU 执行时间

在选定的内存接口处传输 Q 字节，需要接口持续提供服务。若该工作负载能够持续获得的带宽上限为 B 字节每秒，则完成传输至少需要 `Q/B` 秒。只有接口持续繁忙、传输主导执行时，GPU 执行时间才会接近这个数值。

取一组纯演算条件：同一个接口每毫秒能够服务 64 MiB，连续组传输 192 MiB，转置组传输 640 MiB。两个传输需求分别占用 3 ms 和 10 ms。若两组执行均受该接口的带宽限制，GPU 执行时间便接近 3 ms 与 10 ms，比例为 3.33。这个时间预测同时使用了完整流量、相同可用带宽和带宽主导执行三个前提。

GPU 还要执行偏移计算、加载与存储指令、乘加，并处理数据依赖。不同线程的这些工作可以与数据传输重叠。若另一个执行资源在两组中都至少需要 12 ms，那么传输所需的 3 ms 和 10 ms 都短于这项资源需求；充分重叠时，两组执行时间可以都接近 12 ms。若传输只能部分重叠，额外的等待会进入最终时长。完成时间由这些资源需求、依赖关系和实际重叠共同决定。

由此可以构造一个用于判断的简化下界：

```text
GPU 执行时间 ≥ max(
    外部内存所需服务时间,
    各层缓存所需服务时间,
    整数指令所需服务时间,
    浮点指令所需服务时间,
    必须串行等待的数据依赖时间
)
```

每项都采用对应资源自己的工作量和服务能力。这个下界在充分重叠时可能接近实际时间；启动、收尾、依赖和资源竞争会使实际时间增加。转置组同时改变了访问布局与寻址指令，因此其时间变化需要结合这两类工作量检查。

同一批数据可以经过外部内存与多层缓存，每个接口各有流量和带宽。某块已驻留末级缓存时，后续请求可以增加缓存服务量，同时复用先前从外部内存取回的数据。因此，缓存服务量放大与外部读取量接近原值可以同时出现。各层流量分别回答各自接口承担多少工作；它们的比值应分别对应各层的服务能力。

可以用一个 8×8 的 float32 矩阵演算这种情形。在 32 字节块模型中，原缓冲的每行恰好占一块，依次记为 A 至 H。连续组的第一行读取偏移 0 至 7，整批覆盖 A；转置组的第一行读取 0、8、16 至 56，整批覆盖 A 至 H，第二行读取 1、9、17 至 57，又覆盖 A 至 H。完整八行消费同一批块中的全部八个元素。

进一步指定一种服务情形：按输出行逐批执行，较近缓存无法跨批保留这八块，末级缓存可以保留；每批对涉及的块获取一次。连续组从外部获取八块，末级缓存也提供八块；转置组从外部仍只获取八块，末级缓存却需要在八批中提供 `8×8=64` 块。末级缓存保存数据，节省了外部重复取数；它向请求方提供数据的服务仍逐批发生。加上各传输八块的 x 和 out，两组在这个内部接口处的总量分别为 24 块与 80 块，比例仍为 3.33。这个例子规定了执行和驻留情形，用来区分缓存内部服务与外部取数；M4 的实际缓存组织和逐请求命中位置需另外取证。

还需区分资源能力与实际达到的吞吐。观测到总流量 Q 和执行时间 T 后，`Q/T` 给出执行期间的平均吞吐。若转置组让缓存更繁忙，平均吞吐也会升高，流量的增长就能部分表现为单位时间服务更多数据。能否继续提高吞吐，以及哪个资源限制完成时间，需要独立的能力依据或受控实验；用同一次观测的 `Q/T` 代回 `Q/B` 只会还原原来的 T。

### 从 GPU 执行时间到整次调用时间

`4_axpby_bandwidth.py` 用 CPU 秒表包住 `ext.axpby(...)` 与 `mx.eval(...)`。一次计时迭代包含建图、输出内存准备、编码与提交、GPU 开始前的调度或依赖等待、GPU 执行，以及结果完成后 CPU 返回。运行时可以让其中一些阶段重叠，端到端时间由实际时间线上这段调用的起点与终点确定。

GPU 执行期间，CPU 也可能正在 `mx.eval` 内等待结果。因此核算墙钟时间时，应沿时间线统计互不重叠的阶段；CPU 等待区间可能已经包含 GPU 执行，直接把两段完整时长相加会重复计数。

即使在更简化的模型中，GPU 时间比也会被其他固定成本改变。假设连续组 GPU 执行 3 ms、转置组执行 10 ms，两组各有 2 ms 与 GPU 执行互不重叠的额外成本，那么端到端时间为 5 ms 与 12 ms，比例为 2.4。这里全部数字均为演算示例，用于说明计时范围的作用。

要解释一次实测的端到端时间，应在同一次运行中把 CPU 调用区间、GPU 执行区间和等待关联起来。普通 bench 与计数器录制有不同的观测开销，下面分别保留两次运行的数据与各自内部的比值。

### 普通运行的墙钟测量

实测（M4，float32，`N = 4096`，每组跑 30 次取平均，计时区间含 `mx.eval`）：

```python
import time, mlx.core as mx, tiny_llm_ext as ext
N = 4096
x = mx.random.normal((N, N)).astype(mx.float32)
y = mx.random.normal((N, N)).astype(mx.float32)

def bench(xx, yy, label, iters=30):
    for _ in range(3):                       # 预热，第一次要建 pipeline
        mx.eval(ext.axpby(xx, yy, 1.0, 1.0))
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(ext.axpby(xx, yy, 1.0, 1.0))
    dt = (time.perf_counter() - t0) / iters
    print(f"{label:30s} {dt*1e6:9.1f} us  "
          f"{xx.size*4*3/dt/1e9:6.1f} GB/s  {dt/xx.size*1e12:6.1f} ps/elem")

bench(x, y,      "contiguous (x, y)")
bench(x, y.T,    "general (x, y.T)   strides=(1,N)")
bench(x[:N//2], y[::2], "general (x, y[::2]) strides=(2N,1)")
```

```plain
contiguous (x, y)                   2465.6 us   81.7 GB/s   146.9 ps/elem
general (x, y.T)    strides=(1,N)   3074.7 us   65.5 GB/s   183.2 ps/elem
general (x, y[::2]) strides=(2N,1)  1431.4 us   70.3 GB/s   170.6 ps/elem
```

脚本落在 `probes/week2/day1/4_axpby_bandwidth.py`，跑法见 `probes/README.md`。同一台机器第二次跑得 152.6 / 187.8 / 174.9 ps/elem；这两次运行的顺序一致，不能由两次观测保证今后的漂移幅度。

这三行是三种输入布局下的墙钟时间。`y[::2]` 一组只处理前两组一半的元素，归一化为 ps/elem 仍不能消除派发、分配与缓存行为的规模依赖。前两组工作量相同，但同时改变了所选 kernel 和访存布局。三行的差值无法单独测出整数寻址或分散访存的成本，也不能据此判断 L2 命中情况。

这组实验支持的结论是：在这台机器、这些输入规模与这次测量中，三种布局的每元素耗时依次为 146.9、170.6、183.2 ps。若要归因，需要固定输出元素数及其他条件，每次只改变待测的一项；本脚本没有做这种隔离。

2026-11-16 重跑同一脚本，连续、转置和跳行三组依次为 2602.0、3046.3、1473.3 µs，对应 155.1、181.6、175.6 ps/elem。转置组与连续组的耗时比为 `3046.3/2602.0≈1.171`；上表历史测量的比值为 `3074.7/2465.6≈1.247`。

### 计数器录制的来源与计算

同日另用 Instruments 的 Metal System Trace、Performance Limiters 计数集录制脚本。以下数据来自该 trace 的 Run 2，目标进程为 `python (89798)`。脚本依次执行连续、转置和跳行三组，每组预热 3 次、计时 30 次。导出的目标进程 encoder 恰好为 99 个，汇总脚本按创建顺序分为三组，每组排除前三个，再通过 encoder ID 找到 GPU 执行片段。同一 encoder 的片段先取并集，长度之和给出这次调用被记录的 GPU 活跃时间。

计数器在时间区间内报告带宽，带宽乘以区间长度便得到这段时间服务的数据量。汇总脚本取计数器样本与目标 GPU 执行区间的交集，再求和，摊到每组的 30 次调用：

例如，一个样本为 100 GB/s、持续 1 ms，服务量为 100 MB；下一个样本为 200 GB/s、持续 2 ms，服务量为 400 MB。两个区间合计 500 MB。这里的积分就是对每段的“速率乘时长”求和，表格中的每次服务量再除以计入的调用数。

```text
平均流量 MB/次
    = Σ(样本带宽 GB/s × 交集时长 ns) / 10⁶ / 30

平均 GPU 活跃时间
    = 30 次调用各自执行片段的并集长度之和 / 30
```

带宽积分使用样本区间内的常值近似。仓库中的 `docs/evidence/axpby-counters/overlapping-samples.csv` 保存样本值和交集时长，`summary.json` 保存汇总与执行区间；`4_axpby_trace_summary.py` 实现上述关联和积分。对前两组各 30 次计时执行，结果为：

| 观测量 | 连续输入 | 转置输入 |
| --- | ---: | ---: |
| GPU 活跃时间，每次 | 3.7059 ms | 4.0887 ms |
| GPU 外部读取带宽积分，每次 | 155.70 MB | 152.98 MB |
| GPU 外部写入带宽积分，每次 | 67.63 MB | 68.06 MB |
| GPU 末级缓存读写带宽积分，每次 | 250.16 MB | 1094.75 MB |
| Buffer L1 Miss Rate，按时间加权 | 61.53% | 66.77% |

这里的末级缓存来自名为 Last Level Cache Bandwidth 的实际计数器，其定义为 GPU 末级缓存每秒读或写的数据量；GPU Read Bandwidth 则观察 GPU 从外部内存读取的速率。它们分别统计对应层级的服务，缓存指标包含读写、外部读取指标只包含读。上述采集还提供 L1 miss，但这些指标本身未给出 M4 的完整缓存层级、容量或逐请求路径。

表中 MB 为十进制；前面模型中的 MiB 为二进制。采样指标覆盖整个 GPU，因此表格的流量表示目标调用活跃期间的全 GPU 服务量，其中包括同期其他进程的活动。前两组所用指标覆盖各自执行区间的 100%。GPU Read Bandwidth 描述 GPU 从外部内存读取的带宽，进一步对应实际 DRAM 事务需要核对其硬件定义。

使用汇总文件中保留的精度，三个比值分别来自：

```text
末级缓存读写服务量：1094.7538 / 250.1550 ≈ 4.3763
GPU 外部读取量：    152.9760 / 155.6963 ≈ 0.9825
GPU 平均活跃时间：    4.0887 /   3.7059 ≈ 1.1033
```

缓存服务量除以同组 GPU 活跃时间，得到执行期间观测到的平均吞吐：连续组约 `250.1550 MB / 3.7059 ms = 67.50 GB/s`，转置组约 `1094.7538 MB / 4.0887 ms = 267.75 GB/s`。两组平均吞吐相差约 3.97 倍，因此服务量增加 4.38 倍与时间增加约 10% 在算术上相容。这两个吞吐数值描述该次录制的平均服务速率；具体资源瓶颈的判定仍需独立证据。

本次观测将额外服务量主要表现在末级缓存指标上，外部读取量接近连续组。缓存复用与这组现象相符；数据的全 GPU 归属范围和两组同时变化的寻址工作，使各项成本的独立时间贡献仍需受控实验确定。

general 路径增加的工作可以从 `elem_to_loc` 直接定位：它用整数除法和取余拆出坐标，再通过 strides 的整数乘加求地址，之后才执行元素的浮点运算。这些寻址指令与乘加一样占用 GPU 执行资源。Apple 的 Integer and Conditional Limiter、Integer and Complex Limiter 分别观察对应整数相关执行管线相对于峰值能力的压力；本次导出的定义明确说明，100% 表示该管线达到满容量。

Run 2 中，两项指标的时间加权值分别从 2.30% 升至 23.49%、从 2.34% 升至 39.28%。这支持检查 general 的整数寻址需求。百分比的分母是对应管线的峰值能力，因此 39.28% 表示一项资源压力指标；要判断这些工作实际延长多少时间，还要检查该资源是否限制完成时间以及其他工作与之如何重叠。这两项也具有上述全 GPU 采样范围。将它们归到单个 Axpby，需要另外捕获具体 dispatch 并在 GPU profiler 中分析。

### 假设流量比与实测时间比的对应范围

| 比值 | 来源与比较对象 | 数值 |
| --- | --- | ---: |
| 假设的完整流量比 | 同一接口，x 与 out 各一次，y 从一次获取变为逐元素重取 | 640/192≈3.33 |
| 观测的末级缓存服务量比 | Run 2，目标执行区间内的全 GPU 采样积分 | 4.38 |
| 观测的外部读取量比 | 同一 Run 2，另一个接口的采样积分 | 0.98 |
| GPU 平均活跃时间比 | 同一 Run 2，各组 30 次，按执行片段并集计算 | 1.10 |
| 端到端平均墙钟时间比 | 普通运行，各组 30 次，CPU 秒表包含求值等待 | 3046.3/2602.0≈1.17 |

3.33 来自指定获取情形的模型；本次外部读取的采样比值为 0.98，末级缓存的采样比值为 4.38。要将其中某个流量比转为时间预测，还需确定该接口在两组中的可用带宽及其对完成时间的约束。模型、计数器与秒表在这里分别承担假设、观测资源工作量和观测完成时间的职责。

采集配置、XML 导出、执行区间关联、采样积分公式及本次证据保存在 `docs/metal-counter-investigation.md` 和 `docs/evidence/axpby-counters/`。计数器录制与普通墙钟测试是两次运行，时间比值应在各自运行内比较。

### 从源码寻址到单 kernel 的实测

前面的 Instruments 采样覆盖目标执行期间的全 GPU 活动。要把整数压力与某个具体实现对应起来，可以捕获一次已预热的 Axpby，再由 GPU profiler 重放。入口 `probes/week2/day1/4_axpby_capture.py` 保留同样的 `4096×4096` float32 输入与系数，连续和转置两组分别运行：

```sh
CAPTURE_DIR="$(mktemp -d "${TMPDIR%/}/axpby-kernels.XXXXXX")"
for layout in contiguous transpose; do
  MTL_CAPTURE_ENABLED=1 PYTHONPATH=src/extensions pdm run python \
    probes/week2/day1/4_axpby_capture.py --layout "$layout" \
    --trace "$CAPTURE_DIR/$layout.gputrace"
done
```

脚本在捕获前创建输入并预热三次，在 `start_capture` 后同步一次以建立新的 command buffer 边界，再捕获一次求值；正确性比较在停止捕获后执行。每份 trace 应在命令树中实际看到一个计算 dispatch。本机省略捕获内同步时只记录到同步命令，因而“计算结果正确”与“计算已被捕获”需要分别验证。

使用以下步骤重放连续组，并在重新打开后加载嵌入的 profile：

```sh
gpudebug --json --oneshot -t "$CAPTURE_DIR/contiguous.gputrace" --timeout 1800 \
  -c 'profile run --gpu-state default --exec serial --embed' -c 'wait'
gpudebug --json --oneshot -t "$CAPTURE_DIR/contiguous.gputrace" --timeout 120 \
  -c 'profile load' \
  -c 'go /commands/cb0/ce0' \
  -c 'info disp0' \
  -c 'info /performance/timeline' \
  -c 'info /performance/shaders/comp0' \
  -c 'info /performance/timeline/counters/integer_and_complex/integer_and_complex_limiter'
```

`commands/cb0/ce0/disp0` 将 command buffer、encoder 和 dispatch 连起来。`performance/timeline` 提供重放的 GPU 时间，`performance/shaders/comp0` 提供编译后 shader 概览，最后一个节点提供整数资源指标及其定义。转置组对自己的 trace 重复相同步骤；实际子节点以工具列出的名称为准。

2026-11-17，Apple M4、macOS 27.2.1、MLX 0.32.4、gpudebug v1.0 的两份单 kernel 捕获分别完成一次 profiling session：

| 单 kernel 重放字段 | 连续路径 | general 转置路径 |
| --- | ---: | ---: |
| 实际名称 | axpby_contiguous_float32 | axpby_general_float32 |
| GPU time | 2.2706 ms | 3.5630 ms |
| shader Instructions | 7 | 118 |
| shader INT32 | 0 | 24 |
| shader Branch | 0 | 4 |
| shader Temp registers | 3 | 16 |
| Integer and Complex Limiter | 0.00% | 63.85% |
| Last Level Cache Limiter | 31.17% | 65.56% |

两条 dispatch 均为 `16777216×1×1` 个线程，单组 `1024×1×1`。general 的内联参数实际解码为 shape `(4096,4096)`、x strides `(4096,1)`、y strides `(1,4096)`、ndim `2`。这使 kernel 名称与输入布局都得到直接核对。

shader 概览中的指令与寄存器字段显示了 general 更复杂的编译结果；逐维寻址的整数和分支需求，也对应到本次较高的整数 Limiter。Last Level Cache Limiter 则按缓存峰值能力衡量读写服务压力。其 65.56% 与前面 4.38 倍服务量分别描述资源压力和数据量，分母各异。

这次重放时间比约为 1.57，两个 trace 各保存一次 profile，采样为 3 / 10 个核心。该数值属于重放条件下的观测，前一天 Instruments 的 1.10 与普通 bench 的 1.17 分别属于另外两组测量。两类资源压力同时增加，这两份捕获留下了固定布局或固定实现的对照需求。采集、加载、逐项读取与复核细节保存在 `docs/metal-counter-investigation.md`，这批原始证据在 `docs/evidence/axpby-kernels/`。

后续 `probes/week2/day1/4_axpby_control.mm` 直接派发同一份已编译的 Metal kernel，完成 A“连续＋contiguous”、B“连续＋general”、C“转置＋general”三组实验。两个普通进程中，每组各测六十次：B/A 的 GPU 中位数比为 1.0179 和 0.9979，C/B 为 1.2307 和 1.2565。B、C 的单 kernel 报告都为 118 条指令，当前规模的普通计时却显示 C 持续更慢，布局因此成为更有时间信号的调查方向。

三组对照固定了输入规模、输出布局与派发尺寸，A/B 覆盖完整实现差异，B/C 覆盖 y strides 的变化。独立的整数寻址耗时仍需从执行重叠中进一步识别。详细计时边界、全部表格、捕获参数和复跑命令集中在 `docs/metal-counter-investigation.md` 的三组对照一节，数据位于 `docs/evidence/axpby-controls/`；课程实现保持原有自动分派。

连续分支利用三块缓冲共享遍历顺序的前提，直接用 `index` 访问，省去逐维偏移换算。general 分支依据 strides 支持更多布局；其相邻线程是否访问相邻地址，需要由具体 strides 推导。分段数量比与整次调用耗时比对应不同观测量，单凭两者的差距无法识别具体缓存行为。

## 这个算子在课程里的位置

`src/extensions/test.py` 通过一次调用检查 Python 绑定与 CPU 求值：

```python
c = tiny_llm_ext.axpby(a, b, 4.0, 2.0, stream=mx.cpu)
print(f"c correct: {mx.all(c == 6.0).item()}")
```

`probes/week2/day1/4_axpby_bandwidth.py` 用它比较三种布局。Axpby 在课程中承担原生扩展示例的作用：C++ 符号属于 `tiny_llm_ext` 命名空间，Python 绑定由扩展模块提供，MSL 函数由自己的 metallib 提供。这个实现将前端建图、后端分配、参数绑定、线程派发与变换规则组合成可运行的样板。后续自定义算子的计算和布局需求各异，可以分别对照这些接口的职责：

1. `Primitive` 子类持有标量参数与 stream；
2. `eval_cpu` 按 dtype 分派到模板算体；
3. `eval_gpu` 拼 kernel 名、绑槽位、算网格；
4. 需要的地方补 `jvp` / `vjp`（`axpby` 的 `vjp` 只是把伴随量乘 `alpha` 或 `beta`，即 $\\partial z/\\partial x = \\alpha$）；
5. `is_equivalent` 比较标量参数，供编译期算子去重。

Week 2 Day 1 的主线是 Python 模型中的 KV cache。Axpby 示例用于理解后续原生算子的执行接口；缓存任务则通过每层保存历史 K/V，减少下一步 decode 的重复计算。

## 自己复核这几条

源码注释里的数字来自本机实测，命令如下，都可以重跑。

冒烟测试，验证 CPU 路径与绑定：

```bash
cd src/extensions && pdm run python test.py
```

期望输出三行，最后一行是 `c correct: True`。

形状校验的报错文本与发生位置：

```bash
pdm run python -c "
import mlx.core as mx
try:
    mx.broadcast_arrays(mx.zeros((3,4)), mx.zeros((2,4)))
except Exception as e:
    print(type(e).__name__, ':', e)
"
```

期望 `ValueError : [broadcast_shapes] Shapes (3,4) and (2,4) cannot be broadcast.`

dtype 提升表：

```bash
cd src/extensions && pdm run python -c "
import mlx.core as mx, tiny_llm_ext
for dt in (mx.int32, mx.float16, mx.bfloat16, mx.float32, mx.complex64, mx.float64):
    try:
        r = tiny_llm_ext.axpby(mx.ones((2,2), dtype=dt, stream=mx.cpu), mx.ones((2,2), dtype=dt, stream=mx.cpu), 1.0, 2.0, stream=mx.cpu)
        mx.eval(r); print(dt, '->', r.dtype)
    except Exception as e:
        print(dt, '->', e)
"
```

线程组上限与 SIMD 宽度读自 Metal 设备本身，`axpby.cpp` 里的 `kernel->maxTotalThreadsPerThreadgroup()` 就是这条 API 的封装：

```bash
pdm run python -c "import mlx.core as mx; print(mx.device_info())"
```

它给出设备名与架构（本机 `Apple M4`、`applegpu_g16g`），两个上限不在这份信息里，要从设备与 pipeline 读：

```objc
id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
printf("%s\n", [[dev name] UTF8String]);
MTLSize limits = dev.maxThreadsPerThreadgroup;
printf("device.maxThreadsPerThreadgroup=(%lu,%lu,%lu)\n",
       (unsigned long)limits.width, (unsigned long)limits.height, (unsigned long)limits.depth);
printf("pipeline.maxTotalThreadsPerThreadgroup=%lu\n", (unsigned long)ps.maxTotalThreadsPerThreadgroup);
printf("pipeline.threadExecutionWidth=%lu\n", (unsigned long)ps.threadExecutionWidth);
```

```plain
device.name=Apple M4
device.maxThreadsPerThreadgroup=(1024,1024,1024)
pipeline.maxTotalThreadsPerThreadgroup=1024
pipeline.threadExecutionWidth=32
```

`axpby.cpp` 里 `kernel->maxTotalThreadsPerThreadgroup()` 读取当前 kernel pipeline 的单组总线程上限，对应这里的 `pipeline.maxTotalThreadsPerThreadgroup`。

当前 Python array 接口可查看 shape 与 dtype；精确复核本例使用的 strides、flags 和 kernel 选路，可以在 C++ `eval_gpu` 读取 `x.strides()`、`x.flags()` 和拼接后的 `kname`。这些值来自后端实际处理的数组，适合与按布局推导的结果对照。
