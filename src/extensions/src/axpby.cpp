// Copyright © 2023-2025 Apple Inc.

#include "axpby.h"

#include <iostream>
#include <sstream>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace tiny_llm_ext {

///////////////////////////////////////////////////////////////////////////////
// Operation Implementation
///////////////////////////////////////////////////////////////////////////////

/**
 *  Scale and sum two vectors element-wise
 *  z = alpha * x + beta * y
 *
 *  Follow numpy style broadcasting between x and y
 *  Inputs are upcasted to floats if needed
 *
 *  ---------------------------------------------------------------------------
 *  第一阶段：登记。本函数执行完，图里多了一个节点，显存一块没分配。
 *  ---------------------------------------------------------------------------
 *
 *  下面四步只做元数据：dtype、形状、依赖。它们合起来回答「这个节点长什么样」，
 *  不回答「数值是多少」。
 *
 *  精度这一段的实测结论（mlx 0.32.4，本机 M4）：
 *
 *    输入 dtype        输出 dtype    原因
 *    float32           float32       issubdtype(float32, float32) 为真，原样保留
 *    float16           float32       issubdtype(float16, float32) 为假，与 float32 提升
 *    bfloat16          float32       同上
 *    int32 / int8 等   float32       同上，整数被抬到浮点
 *    complex64         complex64     与 float32 提升得 complex64
 *    float64           float64       与 float32 提升得 float64，随后在设备检查处被拒
 *
 *  也就是说原始注释里「non-floating point inputs」说得太窄：float16 与 bfloat16
 *  同样会被抬到 float32。判断依据是 issubdtype(..., float32)，它只对 float32
 *  本身为真，其余一律走 promote_types。
 **/
mx::array axpby(const mx::array &x,              // Input mx::array x
                const mx::array &y,              // Input mx::array y
                const float alpha,               // Scaling factor for x
                const float beta,                // Scaling factor for y
                mx::StreamOrDevice s /* = {} */  // Stream on which to schedule the operation
) {
    // 两个输入 dtype 不同就无法逐元素相加，先按 MLX 的提升规则合成一个。
    // Promote dtypes between x and y as needed
    auto promoted_dtype = promote_types(x.dtype(), y.dtype());

    // 再把结果推到至少 float32。这一步的后果落在 eval_cpu 的分派表上：那张表
    // 只列浮点与复数，整数分支不存在，因为它们到不了那里。alpha 与 beta 是
    // float，整数输入乘上它们必须有浮点缓冲可写。
    // Upcast to float32 for non-floating point inputs x and y
    auto out_dtype =
        mx::issubdtype(promoted_dtype, mx::float32) ? promoted_dtype : promote_types(promoted_dtype, mx::float32);

    // astype 也是算子，它建节点。dtype 相同则原样返回，不同就往图里插一个 Cast。
    // 所以这里写的是「图的样子」，类型转换本身发生在 eval 时。
    // Cast x and y up to the determined dtype (on the same stream s)
    auto x_casted = mx::astype(x, out_dtype, s);
    auto y_casted = mx::astype(y, out_dtype, s);

    // 广播做两件事，且只在这里做第一件：
    //
    //   1. 校验。按 numpy 规则合并形状，合并不了就抛，报错来自深一层的
    //      broadcast_shapes，形如
    //        ValueError: [broadcast_shapes] Shapes (3,4) and (2,4) cannot be broadcast.
    //      这是本算子唯一的输入校验，它发生在调用当场。
    //
    //   2. 定型。被扩展的维度拿一个 0 步长，返回的是视图：不建节点、不搬字节。
    //      0 步长的意思是「沿这一维走多少格，下标都不动」，于是多个逻辑位置
    //      共享同一个缓冲格子。缓冲里没有重复数据。
    //
    // 注意这里不判断连续性。连续性在 eval_gpu 里判断，性质完全不同：那是选哪
    // 条 kernel 的优化决策，两条分支对同一组输入给出相同的数值结果。
    // Broadcast the shapes of x and y (on the same stream s)
    auto broadcasted_inputs = broadcast_arrays({x_casted, y_casted}, s);
    // 广播之后两者同形，取哪个都一样。
    auto out_shape = broadcasted_inputs[0].shape();

    // 唯一新建的节点。构造时 strides 取行优先连续（make_contiguous_strides），
    // 也就是说 general 分支写出来的结果默认是行优先的；contiguous 分支会在
    // eval_gpu 里用 set_data 把它改成与 x 同序。
    //
    // 传进去的 inputs 是广播后的视图，buf 所有权仍在原数组上，这里只加引用计数。
    // Construct the array as the output of the Axpby primitive
    // with the broadcasted and upcasted arrays as inputs
    return mx::array(
        /* const mx::Shape& shape = */ out_shape,
        /* mx::Dtype dtype = */ out_dtype,
        /* std::shared_ptr<mx::Primitive> primitive = */
        std::make_shared<Axpby>(to_stream(s), alpha, beta),
        /* const std::vector<mx::array>& inputs = */ broadcasted_inputs);
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Common Backend Implementation
///////////////////////////////////////////////////////////////////////////////

/**
 * 第二阶段之一：CPU 后端。
 *
 * axpby_impl 不是 kernel。kernel 这个词留给 Metal 那两条：有名字、进 metallib、
 * 由 GPU 编译器生成。这里是一个按 dtype 实例化的模板函数，形态是「分配 + 登记 +
 * 塞一个闭包进 CPU 命令队列」。
 *
 * CPU 路径不做连续/非连续分流。每一维、每个元素都调一次 elem_to_loc 换算偏移，
 * 换来一份能处理任意视图的代码。GPU 侧之所以敢分两条，是因为那里每多一次
 * 整数除法都要乘上几十万个线程，CPU 侧没有这个压力。
 *
 * 输出在 CPU 路径上永远是行优先连续：out.set_data 只给了指针，strides 沿用
 * 节点构造时的默认值。这与 eval_gpu 的 contiguous 分支不同，那里 out 会采用
 * x 的 strides。
 */
template <typename T>
void axpby_impl(const mx::array &x, const mx::array &y, mx::array &out, float alpha_, float beta_, mx::Stream stream) {
    // 求值函数自己分配，因为别的层只知道要几个元素，不知道这次要用什么布局。
    // nbytes() 等于 size() 乘 itemsize，这个数在 CPU 路径上总是对得上，因为
    // 输出被当作行优先连续。
    out.set_data(mx::allocator::malloc(out.nbytes()));

    // 三次登记不是仪式。闭包按值捕获了三个裸指针，如果这里不登记，两个输入
    // 数组可能在闭包真正执行之前就被回收，指针随即悬垂。登记让 encoder 持有
    // 引用，也让它知道这次执行依赖哪些数组。
    // Get the CPU command encoder and register input and output arrays
    auto &encoder = mx::cpu::get_command_encoder(stream);
    encoder.set_input_array(x);
    encoder.set_input_array(y);
    encoder.set_output_array(out);

    // 闭包按值捕获：指针、元素数、形状、两个 strides、两个系数全部拷进去。
    // 捕获引用会让它依赖调用栈，而它的实际执行时间在函数返回之后。
    // Launch the CPU kernel
    encoder.dispatch([x_ptr = x.data<T>(), y_ptr = y.data<T>(), out_ptr = out.data<T>(), size = out.size(),
                      shape = out.shape(), x_strides = x.strides(), y_strides = y.strides(), alpha_, beta_]() {
        // 系数在构造时以 float 存在 Primitive 里，这里落到当前计算 dtype。
        // Cast alpha and beta to the relevant types
        T alpha = static_cast<T>(alpha_);
        T beta = static_cast<T>(beta_);

        // Do the element-wise operation for each output
        for (size_t out_idx = 0; out_idx < size; out_idx++) {
            // out_idx 是逻辑序号。x_offset、y_offset 是它各自缓冲里的下标。
            // elem_to_loc 就是「逻辑序号 -> 缓冲下标」那个换算函数，主体是一个
            // 逐维拆位的循环：
            //
            //   for (int i = ndim - 1; i >= 0; --i) {
            //     auto q_and_r = ldiv(elem, shape[i]);
            //     loc += q_and_r.rem * strides[i];   // 余数 = 这一维的坐标
            //     elem = q_and_r.quot;               // 商   = 还没拆掉的高位
            //   }
            //
            // 两件事分工固定：余数是当前最低维的坐标，乘这一维的步长加进 loc；
            // 商是剥掉这一维之后剩下的数，交给下一轮。循环从最后一维往前，因为
            // 逻辑序号按行优先编，最后一维变化最快。
            //
            // 十进制拆位是同一个算法的熟悉形式。shape=(10,10,10,10)、elem=1234：
            //
            //   i=3: 1234 % 10 = 4   quot = 123    loc += 4*1     = 4
            //   i=2: 123  % 10 = 3   quot = 12     loc += 3*10    = 34
            //   i=1: 12   % 10 = 2   quot = 1      loc += 2*100   = 234
            //   i=0: 1    % 10 = 1   quot = 0      loc += 1*1000  = 1234
            //
            // 商不是中间结果，是被拆的数本身在缩小：1234 -> 123 -> 12 -> 1 -> 0。
            // elem 归零表示所有维度都取完了，Metal 版据此提前退出循环。
            // 混合进制只是把每层的 10 换成 shape[i]，(3,4) 就是低位四进制、
            // 高位三进制。
            //
            // 步长为 0 时这一维的贡献恒为 0，等于永远取第 0 个。广播因此在
            // general 路径上没有专门代码：它只按步长算，算出来自然是同一格。
            // 例：shape=(3,4)、strides=(0,1)、elem=7 拆出 (行 1, 列 3)，
            // loc = 3*1 + 1*0 = 3，与同一列另两行的 index 落在一起。
            //
            // 连续时两个 offset 恰好都等于 out_idx，这一步是纯粹的浪费；不连续
            // （切片、转置、广播）时它是正确性的唯一来源。CPU 侧不做判断，一律
            // 走换算。
            // Map linear indices to offsets in x and y
            // .venv/lib/python3.10/site-packages/mlx/include/mlx/backend/common/utils.h
            auto x_offset = mx::elem_to_loc(out_idx, shape, x_strides);
            auto y_offset = mx::elem_to_loc(out_idx, shape, y_strides);

            // 输出直接按下标写，因为它在 CPU 路径上一定是连续的：缓冲第 out_idx
            // 格就是逻辑第 out_idx 个元素。
            // We allocate the output to be contiguous and regularly strided
            // (defaults to row major) and hence it doesn't need additional mapping
            out_ptr[out_idx] = alpha * x_ptr[x_offset] + beta * y_ptr[y_offset];
        }
    });
}

/**
 * 按 dtype 分派到上面那个模板的某个实例化。
 *
 * 四张分支对应四条 instantiations，运行期只做四次 dtype 比较，没有虚调用。
 * 最后的 throw 在本机实测里几乎碰不到：float64 输入在更早的设备检查处就被
 * 拒绝（ValueError: float64 is not supported on the GPU）。它留给的是「以后
 * 出现新的浮点或复数 dtype」这一情形 —— 那时前端照样能把 dtype 推到这里，
 * 而这里没有对应的分支。
 *
 * 整数不会走到这个 throw，因为 axpby() 已经把它们抬成 float32 了。
 */
void Axpby::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &x = inputs[0];
    auto &y = inputs[1];
    auto &out = outputs[0];

    // Dispatch to the correct dtype
    if (out.dtype() == mx::float32) {
        return axpby_impl<float>(x, y, out, alpha_, beta_, stream());
    } else if (out.dtype() == mx::float16) {
        return axpby_impl<mx::float16_t>(x, y, out, alpha_, beta_, stream());
    } else if (out.dtype() == mx::bfloat16) {
        return axpby_impl<mx::bfloat16_t>(x, y, out, alpha_, beta_, stream());
    } else if (out.dtype() == mx::complex64) {
        return axpby_impl<mx::complex64_t>(x, y, out, alpha_, beta_, stream());
    } else {
        throw std::runtime_error("Axpby is only supported for floating point types.");
    }
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Metal Backend Implementation
///////////////////////////////////////////////////////////////////////////////

/**
 * 第二阶段之二：GPU 后端。
 *
 * 这段代码做四件事，顺序不能换：判两条 kernel 里用哪条、按那个判据分配输出、
 * 拼 kernel 名并取到 pipeline、把参数按槽位塞进命令缓冲，最后划出线程网格。
 *
 * 三处「手写契约」在这里全部出现，它们都没有编译期检查：
 *
 *   1. kernel 名字符串，C++ 侧拼、Metal 侧由宏拼，靠字面量对齐；
 *   2. buffer 槽位号，两侧各自照抄；
 *   3. 库名 "tiny_llm_ext"，注册在 tiny_llm_ext/__init__.py 里，取用于此。
 */
#ifdef _METAL_

/** Evaluate primitive on GPU */
void Axpby::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    // Prepare inputs
    auto &x = inputs[0];
    auto &y = inputs[1];
    auto &out = outputs[0];

    // Each primitive carries the stream it should execute on
    // and each stream carries its device identifiers
    auto &s = stream();
    // We get the needed metal device using the stream
    auto &d = mx::metal::device(s.device);

    // ---------------------------------------------------------------------
    // 分流判据
    // ---------------------------------------------------------------------
    //
    // 连续 kernel 的整个函数体只有一句：
    //
    //     out[index] = alpha * x[index] + beta * y[index];
    //
    // 它拿一个扁平下标同时索引三块缓冲。这在语义上成立，当且仅当三块缓冲的
    // 扁平顺序指向同一个逻辑元素。行优先与行优先同序，列优先与列优先也同序，
    // 行列混用不同序，于是条件写成两个「与」再取「或」。
    //
    // 所以这个 bool 表达的不是「x 和 y 都连续」，而是「x 和 y 共享一种遍历
    // 顺序」。两个 flag 各自的定义在 include/mlx/array.h：
    //
    //     row_contiguous: strides[-1]==1 且 strides[i]==shape[i+1]*strides[i+1]
    //     col_contiguous: strides[0]==1  且 strides[i]==shape[i-1]*strides[i-1]
    //
    // 两条合起来断言：缓冲格子与逻辑元素一一对应，且顺序是行优先（或列优先）。
    // 切片造出空格子（v = x8[::2]，strides (8,1)，中间空了 4 格），广播让多个
    // 逻辑位置共享一格（步长 0），两种都让这个断言失效。
    //
    // 这个判断拒绝任何输入吗？不。两条分支对同一组输入给出相同的数值结果，
    // general 只是每元素多几次整数除法。走 general 不是降级，是正常路径。
    //
    // Prepare to specialize based on contiguity
    bool contiguous_kernel = (x.flags().row_contiguous && y.flags().row_contiguous) ||
                             (x.flags().col_contiguous && y.flags().col_contiguous);

    // 分配有两种，区别在 out 的 strides：
    //
    //   contiguous 分支：把 out 绑成与 x 同序。data_size() 在这里安全，因为
    //                    row/col_contiguous 为真时缓冲里没有空格子，元素数
    //                    等于 size()。两个 flag 一起搬过去，下游才知道 out
    //                    是列优先的。
    //   general    分支：只给指针，strides 沿用节点构造时的行优先连续，这正
    //                    是 kernel 里 out[index] 直接写的依据。
    //
    // Allocate output memory with strides based on specialization
    if (contiguous_kernel) {
        out.set_data(mx::allocator::malloc(x.data_size() * out.itemsize()), x.data_size(), x.strides(), x.flags());
    } else {
        out.set_data(mx::allocator::malloc(out.nbytes()));
    }

    // kernel 名由两个维度拼成：分支（contiguous / general）与 dtype 名。八条
    // 组合全部由 axpby.metal 末尾的宏提前实例化：
    //
    //     instantiate_axpby(float32, float) -> axpby_general_float32
    //                                          axpby_contiguous_float32
    //     ... float16 / bfloat16 / complex64 同理，共八条
    //
    // 两侧靠字符串字面量对齐，拼错一个字母的表现是运行期报找不到 kernel，
    // 而不是编译错误。
    // Resolve name of kernel (corresponds to axpby.metal)
    std::ostringstream kname;
    kname << "axpby_";
    kname << (contiguous_kernel ? "contiguous_" : "general_");
    kname << type_to_name(out);

    // Make a kernel from this metal library (use lib name overload)
    //
    // 这个 "tiny_llm_ext" 是注册名，不是文件名：tiny_llm_ext/__init__.py 在
    // import 时调 load_library，后者把同目录下的 tiny_llm_ext.metallib 按这个名字
    // 登记进 GPU device（src/extensions/src/utils.cpp:12-15）。两处字符串独立
    // 书写，对不上同样是运行期才发现。
    auto library = d.get_library("tiny_llm_ext");
    auto kernel = d.get_kernel(kname.str(), library);

    // Prepare to encode kernel
    auto &compute_encoder = mx::metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);

    // 槽位表，两侧必须一致：
    //
    //   槽位  axpby.cpp                      axpby.metal                        何时
    //   0     set_input_array(x, 0)          device const T* x                  总是
    //   1     set_input_array(y, 1)          device const T* y                  总是
    //   2     set_output_array(out, 2)       device T* out                      总是
    //   3     set_bytes(alpha_, 3)           constant const float& alpha        总是
    //   4     set_bytes(beta_, 4)            constant const float& beta         总是
    //   5     set_vector_bytes(x.shape(), 5) constant const int* shape          仅 general
    //   6     set_vector_bytes(x.strides(), 6) constant const int64_t* x_strides 仅 general
    //   7     set_vector_bytes(y.strides(), 7) constant const int64_t* y_strides 仅 general
    //   8     set_bytes(ndim, 8)             constant const int& ndim           仅 general
    //
    // Kernel parameters are registered with buffer indices corresponding to
    // those in the kernel declaration at axpby.metal
    int ndim = out.ndim();
    size_t nelem = out.size();

    // Encode input arrays to kernel
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_input_array(y, 1);

    // Encode output arrays to kernel
    compute_encoder.set_output_array(out, 2);

    // Encode alpha and beta
    compute_encoder.set_bytes(alpha_, 3);
    compute_encoder.set_bytes(beta_, 4);

    // shape 与 strides 只在 general 分支需要：连续 kernel 里它们等于恒等映射。
    //
    // 这几个参数在 Metal 侧写成 constant const int*，看着像设备指针，实际上
    // setBytes 是把主机内存里的这几项拷进命令缓冲，kernel 读的是缓冲内嵌的
    // 那一份（include/mlx/backend/metal/device.h:63-86）。所以传临时 vector
    // 是安全的，不需要它活到执行结束。
    // Encode shape, strides and ndim if needed
    if (!contiguous_kernel) {
        compute_encoder.set_vector_bytes(x.shape(), 5);
        compute_encoder.set_vector_bytes(x.strides(), 6);
        compute_encoder.set_vector_bytes(y.strides(), 7);
        compute_encoder.set_bytes(ndim, 8);
    }

    // ---------------------------------------------------------------------
    // 线程网格
    // ---------------------------------------------------------------------
    //
    // 三层可加性模型：
    //
    //   线程 grid     这次派发一共多少线程，编号 thread_position_in_grid，
    //                 就是 kernel 里的 uint index。
    //   线程组        硬件调度与共享内存的单位，编号 threadgroup_position_in_grid。
    //   组内线程序号  thread_position_in_threadgroup。
    //
    // 本机实测（Apple M4）：maxThreadsPerThreadgroup = 1024，
    // threadExecutionWidth（SIMD 宽度）= 32。线程组取 32 的整数倍时利用率最高。
    //
    // min 保的上界属于「静默失败」那一类，实测值得记下来：请求
    // threadsPerThreadgroup = 2048 或 4096（grid = 4096）时，命令缓冲不报错，
    // 但 kernel 一次都没执行，输出缓冲保持初值。对照 CUDA 的 1025 线程/block
    // 是启动即被拒（invalid configuration argument）。同一个越界，一边当场报错，
    // 一边静默给零。少了这个 min，大数组会算出未初始化的结果而没有任何提示。
    //
    // nelem 小于上限时 min 让请求的组大小等于实际工作量，也不必依赖驱动的钳位：
    // 实测 grid=1000、group=1024 时驱动把整组钳成 1000；grid=12、group=1024
    // 时钳成 12。这条钳位只发生在 group 大于 grid 的方向上。
    //
    // dispatch_threads 的第一个参数是「线程总数」，与 CUDA 的 <<<grid, block>>>
    // 不同，后者给的是「组的个数」。差别在尾巴上：dispatch_threads 允许最后一组
    // 变小（实测 grid=10、group=4 得到 4,4,2 三组，共 10 个线程），所以本 kernel
    // 不需要 if (index < nelem) 这样的边界判断；换成 dispatch_threadgroups 加
    // grid=3 组，就会启动 12 个线程，多出的两个写到缓冲外面。
    //
    // We launch 1 thread for each input and make sure that the number of
    // threads in any given threadgroup is not higher than the max allowed
    size_t tgp_size = std::min(nelem, kernel->maxTotalThreadsPerThreadgroup());

    // Fix the 3D size of each threadgroup (in terms of threads)
    MTL::Size group_dims = MTL::Size(tgp_size, 1, 1);

    // Fix the 3D size of the launch grid (in terms of threads)
    MTL::Size grid_dims = MTL::Size(nelem, 1, 1);

    // Launch the grid with the given number of threads divided among
    // the given threadgroups
    compute_encoder.dispatch_threads(grid_dims, group_dims);
}

#else  // Metal is not available

/**
 * 没有 Metal 时仍然要给 eval_gpu 一个定义，否则链接会失败：基类是虚函数，
 * 派生类不覆盖也能编译，但 nodes 在 GPU stream 上求值时会走到基类的默认实现，
 * 报错信息远不如这一句直接。
 *
 * Fail evaluation on GPU
 */
void Axpby::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &out) {
    throw std::runtime_error("Axpby has no GPU implementation.");
}

#endif

///////////////////////////////////////////////////////////////////////////////
// Primitive Transforms
///////////////////////////////////////////////////////////////////////////////

/**
 * 第三阶段：图变换钩子。
 *
 * 这四个函数不碰数据。它们在图被变换时被调用，返回的仍然是图节点，所以对
 * axpby 求导、求向量化之后，运行期照样只走 eval_cpu / eval_gpu。这也解释了
 * 为什么它们用 mx::multiply 这类现成算子就能写：它们描述的是「计算怎么连」。
 */

/** Print primitive name and parameters */
void Axpby::print(std::ostream &os) { os << name() << "(alpha=" << alpha_ << ", beta=" << beta_ << ")"; }

/**
 * 前向模式微分。
 *
 * z = alpha*x + beta*y 对 x 与 y 都是线性的，所以沿任意方向 (tx, ty) 的变化率
 * 就是 alpha*tx + beta*ty，不需要新的 kernel。
 *
 * 两个分支对应 argnums 给了一个输入还是两个：
 *   只对一个输入求导：结果是那个输入的切向量乘以它自己的系数。
 *   对两个都求导：两项相加，这里直接复用 axpby 本身 —— 图里于是出现同一算子
 *   的第二个节点，系数相同，可以被 is_equivalent 合并。
 *
 * 注意 scale_arr 用 tangents[0].dtype() 造常量：标量与切向量的 dtype 必须一致，
 * 否则 mx::multiply 会再做一次类型提升，多出一个节点。
 */
std::vector<mx::array> Axpby::jvp(const std::vector<mx::array> &primals, const std::vector<mx::array> &tangents,
                                  const std::vector<int> &argnums) {
    // Forward mode diff that pushes along the tangents
    // The jvp transform on the primitive can built with ops
    // that are scheduled on the same stream as the primitive

    // If argnums = {0}, we only push along x in which case the
    // jvp is just the tangent scaled by alpha
    // Similarly, if argnums = {1}, the jvp is just the tangent
    // scaled by beta
    if (argnums.size() == 1) {
        auto scale = argnums[0] == 0 ? alpha_ : beta_;
        auto scale_arr = mx::array(scale, tangents[0].dtype());
        return {mx::multiply(scale_arr, tangents[0], stream())};
    }
    // If, argnums = {0, 1}, we take contributions from both
    // which gives us jvp = tangent_x * alpha + tangent_y * beta
    else {
        return {axpby(tangents[0], tangents[1], alpha_, beta_, stream())};
    }
}

/**
 * 反向模式微分。
 *
 * dz/dx = alpha、dz/dy = beta，两者都是与输入无关的常数，所以每个输入的梯度
 * 只是上游伴随量乘一个标量。这也是为什么这段没有任何形状运算：广播在反向时
 * 的求和问题由框架在别处处理，这个算子本身不产生额外的梯度形状。
 *
 * 返回的顺序必须与 argnums 一致，框架按位置把梯度配回输入。
 */
std::vector<mx::array> Axpby::vjp(const std::vector<mx::array> &primals, const std::vector<mx::array> &cotangents,
                                  const std::vector<int> &argnums, const std::vector<mx::array> &) {
    // Reverse mode diff
    std::vector<mx::array> vjps;
    for (auto arg : argnums) {
        auto scale = arg == 0 ? alpha_ : beta_;
        auto scale_arr = mx::array(scale, cotangents[0].dtype());
        vjps.push_back(mx::multiply(scale_arr, cotangents[0], stream()));
    }
    return vjps;
}

/**
 * 向量化。本类没有实现，直接抛。
 *
 * 未实现是显式选择：mx::vmap 用到这个算子时会在它身上报错，而不是给出悄悄
 * 错误的结果。对照 week 2 之后的算子，有的补了 vmap，有的同样抛，取舍在于
 * 是否真的会在批量维度上调用到它。
 *
 * Vectorize primitive along given axis
 */
std::pair<std::vector<mx::array>, std::vector<int>> Axpby::vmap(const std::vector<mx::array> &inputs,
                                                                const std::vector<int> &axes) {
    throw std::runtime_error("Axpby has no vmap implementation.");
}

/**
 * 编译期算子去重。
 *
 * 框架用它判断两个节点能不能合并。只比较系数，因为算子种类已由类决定，而
 * stream、inputs 由外层处理。
 *
 * 这里的 static_cast 没有类型检查：调用方保证同类型才进来。如果这个类日后
 * 被复制成另一个 Primitive 而忘了改 cast，比较会读到错位的字节，得到无意义的
 * true/false，表现为图被错误合并。
 *
 * Equivalence check
 */
bool Axpby::is_equivalent(const Primitive &other) const {
    const Axpby &r_other = static_cast<const Axpby &>(other);
    return alpha_ == r_other.alpha_ && beta_ == r_other.beta_;
}

}  // namespace tiny_llm_ext
