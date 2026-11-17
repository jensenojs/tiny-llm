// Copyright © 2023-2025 Apple Inc.

// ===========================================================================
// axpby.metal —— GPU 端执行的算体
// ===========================================================================
//
// 三个文件的分工：
//
//   axpby.cpp    主机侧（CPU 上跑的 C++）：决定用哪块设备、算出线程总数、
//                分配输出、把参数写进命令缓冲、发出一条 dispatch 命令。
//   axpby.metal  设备侧（GPU 上跑的函数）：每条线程一个实例，读三个指针、
//                算一个下标、写一个结果。它不知道 grid 有多大，看不到 Python，
//                也拿不到 argc/argv 之类的东西——参数只有签名里列的那些。
//   axpby.h      两边的公共声明。
//
// 编译与取用：本文件由 clang 编进 metallib，作为 "tiny_llm_ext" 这个 library
// 注册到 GPU device（src/extensions/src/utils.cpp:12-15）。运行时主机侧
// d.get_library("tiny_llm_ext") 拿到 library，再用 d.get_kernel("axpby_
// contiguous_float32", lib) 按字符串取符号。字符串由 axpby.cpp 里的 kname
// 拼出，符号由本文件末尾的 instantiate_axpby 造出，两边的拼法必须逐字一致。
//
// 两个 kernel 是同一个语义 z[i] = α*x[i] + β*y[i]，区别只在一次下标换算：
// general 每条线程多做两次 elem_to_loc，contiguous 三个指针共用 index。
// 阅读顺序：先 general 弄清参数表，再看 contiguous 少了什么。
//
// ===========================================================================

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

// ---------------------------------------------------------------------------
// axpby_general —— 每条线程做两次下标换算
// ---------------------------------------------------------------------------
//
// 先把签名读一遍：它同时是「主机要传什么」和「GPU 拿到什么」的契约。
//
//   template <typename T>
//       一份源码要产出四种 dtype。T 决定指针的元素宽度（float 4 字节、
//       half 2 字节）和算术类型，每种 T 各实例化一份独立符号，见文件末尾。
//
//   device const T* x [[buffer(0)]]
//       device 是地址空间：这块内存在 GPU 全局显存里，读写要经过内存控制器。
//       const 告诉编译器这条线程不写它。T* 是元素指针而不是字节指针，
//       x[3] 已经按 sizeof(T) 步进，所以 elem_to_loc 返回的是元素偏移。
//       方括号里的 0 是参数槽位号，与主机侧 set_input_array(x, 0)、
//       set_output_array(out, 2) 的数字配对。序号唯一的作用就是配对，配错了
//       编译期查不出来，运行时读到的是别的参数。
//
//   constant const float& alpha [[buffer(3)]]
//       constant 是第二块地址空间：参数内存。主机每次 dispatch 时用 set_bytes
//       把 4 个字节拷进命令缓冲，kernel 内只读，硬件允许整个线程组共享这一份
//       而不用每个线程各占一个寄存器。引用绑定到那块内存的偏移上。
//       α、β 声明成 float 而不是 T：主机侧的 alpha_/beta_ 就是 float，kernel
//       里再 static_cast<T> 落到计算类型（正文最后有说明）。
//
//   constant const int* shape            [[buffer(5)]]
//   constant const int64_t* x_strides    [[buffer(6)]]
//   constant const int64_t* y_strides    [[buffer(7)]]
//   constant const int& ndim             [[buffer(8)]]
//       缓冲布局不连续时才需要这四项，contiguous 分支没有它们。shape 用 int
//       而 strides 用 int64_t：单维长度不会超过 2^31，strides 是维度长度的
//       乘积，可以超过。ndim 单独传是因为 Metal 的指针参数不带长度信息，
//       kernel 无法自己推断要循环几维。
//
//   uint index [[thread_position_in_grid]]
//       这条线程在 grid 里的线性编号，0 到 nelem-1。它的值由硬件在启动每个
//       线程实例时写进参数表，主机没有为它传过任何东西——这就是主机侧
//       set_*_array 只排到槽位 8 的原因。
//
// 这个编号的原始类型是 uint3，这里声明成 uint 取的是 x 分量。三维的来源是
// 图形负载：一张图的线程空间天然是宽×高×深度。线性化公式是
// lin = x + W*(y + H*z)；axpby 是一维问题，W=nelem、H=Z=1，所以 .x 就是
// 线性编号，与主机侧 MTL::Size(nelem, 1, 1) 的填法对应。矩阵乘那类二维
// kernel 会声明 uint3，在函数体里分别用 .x 和 .y。
//
template <typename T>
[[kernel]] void axpby_general(
    device const T* x [[buffer(0)]],
    device const T* y [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const float& alpha [[buffer(3)]],
    constant const float& beta [[buffer(4)]],
    constant const int* shape [[buffer(5)]],
    constant const int64_t* x_strides [[buffer(6)]],
    constant const int64_t* y_strides [[buffer(7)]],
    constant const int& ndim [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
  // 每个输出元素做一次：先把输出序号拆成各维坐标，再乘各自的 strides 得到
  // 缓冲偏移。x 和 y 的布局可以互不相同，所以两次换算各用自己的一份 strides。
  auto x_offset = elem_to_loc(index, shape, x_strides, ndim);
  auto y_offset = elem_to_loc(index, shape, y_strides, ndim);
  // out 由主机按行优先连续分配（axpby.cpp 的非连续分支走 out.set_data(nbytes)），
  // 所以输出侧不需要换算，index 本身就是偏移。
  // static_cast<T>(alpha)：T 是 half 时 α 先截成 half 再参与乘法，这与「在
  // float 里乘完再转 half」在数值上不同。选前者是为了让整条表达式落在 T 上，
  // 每个元素省掉两次类型转换。
  out[index] =
      static_cast<T>(alpha) * x[x_offset] + static_cast<T>(beta) * y[y_offset];
}

// ---------------------------------------------------------------------------
// axpby_contiguous —— 三块缓冲共用同一个下标
// ---------------------------------------------------------------------------
//
// 主机侧的判据 (x 行连续 && y 行连续) || (x 列连续 && y 列连续) 为真时走这条。
// 此时三块缓冲共享同一种遍历顺序，可以直接按缓冲下标 index 同时读写。
// 行连续分支按行遍历，列连续分支按列遍历；out 继承 x 的布局。
//
// 直接下标访问省去逐维整数寻址，相邻 index 在每块缓冲中访问相邻元素。
// 对 general 分支中的 4096×4096 转置 y，线程 index=r*4096+c 读取原缓冲
// 偏移 c*4096+r：同一输出行内相邻列的地址相差 16 KB；固定列 c、输出行
// r=0..7 的八条线程则必然读取相邻的八个 float，共 32 字节。
// 若按假设的 32 字节对齐单位计数，输出前八行读取 4096 个这样的单位。
// 每单位获取一次为 128 KiB；八次元素读取各自重新获取为 1 MiB。
// 完整矩阵的 y 对应 64 MiB 或 512 MiB；若 x/out 各传输 64 MiB，
// 总量分别为 192 MiB 与 640 MiB，比例为 10/3。
// 32 字节是计算假设，M4 的对应硬件单位与实际缓存获取次数尚待取证。
// 地址关系、分段获取次数和整次耗时是三个层次；完整推导与复核命令见
// docs/axpby-primitive-contract.md。
// 2026-11-16 Instruments Performance Limiters 采样中，连续/转置组各30次：
// GPU 活跃区间的全GPU外部读量积分为155.70/152.98 MB每次，末级缓存读写
// 积分为250.16/1094.75 MB每次。前两组覆盖率100%，包含同期其他GPU负载。
// 采集与积分方法见 docs/metal-counter-investigation.md。
//
// 本机历史测量三种输入布局的每元素耗时：
//   连续 146.9 ps、y[::2]（相邻线程读相邻列）170.6 ps、y.T（相邻线程
//   跨行读）183.2 ps。三组的形状、元素数和访存布局未全部保持一致，
//   因此耗时差不能分别归给整数寻址和分散访存；这些数只比较三种实际输入。
//
// 这条 kernel 没有 if (index < nelem) 这样的边界判断，原因在主机侧的派发
// 方式：dispatch_threads 声明的是线程总数，驱动保证不为多出来的位置创建
// 线程，末组允许不满（实测 grid=10、group=4 得到 4,4,2）。换成 CUDA 的
// <<<blocks, threads>>> 或 Metal 的 dispatch_threadgroups，线程总数只能是
// 组数乘组内线程数，向上取整后多出的线程必须由 kernel 自己挡住。
//
template <typename T>
[[kernel]] void axpby_contiguous(
    device const T* x [[buffer(0)]],
    device const T* y [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const float& alpha [[buffer(3)]],
    constant const float& beta [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  out[index] =
      static_cast<T>(alpha) * x[index] + static_cast<T>(beta) * y[index];
}

// ---------------------------------------------------------------------------
// 实例化与符号名
// ---------------------------------------------------------------------------
//
// instantiate_kernel(name, func, type) 定义在
// include/mlx/backend/metal/kernels/defines.h:23-25，展开后是
//   template [[host_name(name)]] [[kernel]] decltype(func<type>) func<type>;
// 一次完成两件事：把模板实例化成真正的 GPU 入口，并给这个入口一个字符串名。
//
// 四次 instantiate_axpby 产出 8 个符号：axpby_general_float32、
// axpby_contiguous_float32、axpby_general_float16……主机侧 kname 拼的是
// "axpby_" + ("contiguous_" 或 "general_") + type_to_name(out)，与这里的
// 字符串必须逐字相同，否则 get_kernel 取不到符号。
//
// #type_name 是宏的字符串化参数；float32 这类名字与 mlx 的 type_to_name 输出
// 一致，half、bfloat16_t、complex64_t 是 metal_stdlib 与本文件 include 的
// utils.h 提供的类型名。complex64 走同一条表达式：α 先转成 complex64_t，
// 再按复数分量相乘相加。
//
// clang-format off
#define instantiate_axpby(type_name, type)                             \
  instantiate_kernel("axpby_general_" #type_name, axpby_general, type) \
  instantiate_kernel(                                                  \
          "axpby_contiguous_" #type_name, axpby_contiguous, type)

instantiate_axpby(float32, float);
instantiate_axpby(float16, half);
instantiate_axpby(bfloat16, bfloat16_t);
instantiate_axpby(complex64, complex64_t);
// clang-format on
