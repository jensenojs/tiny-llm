// Copyright © 2023-2025 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

/**
 * ===========================================================================
 * axpby：一个 Primitive 与它三个后端之间的契约
 * ===========================================================================
 *
 * 这个算子是 MLX 官方 examples/extensions/axpby 的移植，课程拿它当「最小完整
 * Primitive」的样板。它数值上没有用处（模型里没有任何地方调用它），但它把一个
 * 自定义算子必须交代的每一件事都摆到了最短的篇幅里：前端登记、CPU 执行、
 * GPU 执行、三个图变换钩子。week 2 起的 quantized_matmul / rms_norm / rope /
 * swiglu / paged_attention 都是这套骨架换掉中间那一段。
 *
 * 三个文件各有分工，分工由编译单元决定：
 *
 *   axpby.h      契约声明。谁在什么时候被调用、参数去哪、必须返回什么。
 *   axpby.cpp    前端登记 + 两个后端 + 三个图变换钩子。能看见 mx::array，
 *                看不见 Metal 着色器的参数表。
 *   axpby.metal  两条真正的 GPU kernel。只认识 Metal 语言，看不见 mx::array。
 *
 * bindings.cpp 把 axpby 这个 C++ 符号注册成 Python 名字，解释器侧只存在那一个
 * 名字。三段互不相识，数据却要从第三段流到第一段，能承担这件事的只剩一份
 * 手写的、两侧各自照抄的参数表，也就是 axpby.metal 里的 [[buffer(N)]] 与
 * axpby.cpp 里的 set_*(..., N)。没有任何编译期检查确认两侧对得上。
 *
 * 阅读顺序（端到端一次走完）：
 *   1. 本文件的 axpby() 声明与 Axpby 类；
 *   2. axpby.cpp 的 axpby() 实现 —— 调用发生时不计算，只登记一个图节点；
 *   3. axpby.cpp 的 axpby_impl 与 eval_cpu —— 无 Metal 时它才是真正执行的那条路；
 *   4. axpby.cpp 的 eval_gpu —— 连续性分流、kernel 名、buffer 槽位、线程网格；
 *   5. axpby.metal —— 拿到 index 之后那两行赋值；
 *   6. axpby.cpp 尾部的 jvp / vjp / is_equivalent —— 图层面的三个钩子。
 *
 * 三个贯穿全文的概念，后面不再重复定义：
 *
 *   逻辑元素   把输出按行优先摊平之后的第 i 个。z[i] 里的 i 就是它。
 *   缓冲下标   这个逻辑元素落在物理缓冲的哪一格。读写发生在这一层。
 *   strides    每个维度走一格，缓冲下标加多少。二维行优先 (4,1) 的意思是
 *              「换行 +4，换列 +1」，两项各自乘上坐标再相加。
 *   广播       把长度为 1 的维度扩成 N，做法是把那一维的步长写成 0：沿它走
 *              多少格，下标都不动。缓冲里没有任何重复数据。
 *
 * 一条贯穿全文的判据：连续 kernel 用同一个 i 索引 x、y、out 三块缓冲。这个
 * 做法成立的前提是三块缓冲的「第 i 格」指向同一个逻辑元素，也就是缓冲格子与
 * 逻辑元素一一对应。切片造出空格子（有缝），广播让多个逻辑位置共享一格
 * （不单射），两种都破坏这个对应，于是都只能走 general。
 */

namespace tiny_llm_ext {

///////////////////////////////////////////////////////////////////////////////
// Operation
///////////////////////////////////////////////////////////////////////////////

/**
 *  Scale and sum two vectors element-wise
 *  z = alpha * x + beta * y
 *
 *  Follow numpy style broadcasting between x and y
 *
 *  这个函数只登记，不算。四件事：对齐 dtype、广播形状、造一个 Axpby 节点、
 *  返回它。返回的 mx::array 形状与 dtype 都已确定，data 为空 —— 它记录的是
 *  「由谁算」，不是算出来的结果。真正的分配与下发在 eval_cpu / eval_gpu 里。
 *
 *  alpha、beta 不进 inputs。inputs 装的是上游数组（依赖关系），而系数是这个
 *  算子自身的属性：同一个 axpby 换个系数就是另一个算子，这件事由
 *  is_equivalent 负责比较。把它们塞进 inputs 会让图多出两个常量节点。
 *
 *  s 决定节点落在哪个 stream 上，也就决定了执行时走 eval_cpu 还是 eval_gpu。
 *  这一层不做选择，它把 s 原样交给 Primitive 的构造函数。
 *
 *  精度：输出至少是 float32，但会悄悄改变输入精度。float16 与 bfloat16 输入
 *  会被抬到 float32 再算，输出也是 float32；只有 float32 与 complex64 保持
 *  原 dtype。axpby.cpp 的 axpby() 那段注释记了实测表。
 *
 *  这个算子会抛异常，且发生在调用当场而非 eval 时：x 与 y 的形状按 numpy
 *  广播规则合并不了时立刻抛 ValueError。
 **/
mx::array axpby(const mx::array &x,        // Input array x
                const mx::array &y,        // Input array y
                const float alpha,         // Scaling factor for x
                const float beta,          // Scaling factor for y
                mx::StreamOrDevice s = {}  // Stream on which to schedule the operation
);

///////////////////////////////////////////////////////////////////////////////
// Primitive
///////////////////////////////////////////////////////////////////////////////

/**
 * 为什么需要这一层：mx::array 的节点里存着一个 shared_ptr<Primitive>，它回答
 * 「将来按什么算」。这个回答必须能以多态方式被调用、必须能在 C++ 侧持有标量
 * 参数、必须活得比任何 Python 对象都久，所以 C++ 里有一个基类，下面六个虚
 * 函数就是这套契约的全部。
 *
 * 六个 override 的调用时机，按一次求值的顺序：
 *
 *   eval_cpu / eval_gpu   求值时调用一次，stream 决定哪一个。义务是自己给
 *                         outputs[0] 分配内存，因为别的层不知道该分配多少、
 *                         什么布局、什么 dtype。
 *   jvp                   对某个输入求前向模式导数时调用。返回新的图节点，
 *                         不做数值计算。
 *   vjp                   反向传播经过这个节点时调用。同上，只建图。
 *   vmap                  把这个算子向量化到某个轴上时调用。本类直接抛。
 *   name / print          name 是算子在报错与调试输出里的名字，与 Metal
 *                         kernel 名无关，后者在 eval_gpu 里现拼。
 *   is_equivalent         编译期算子去重的依据：两个节点能否合并成一个。
 *
 * 六个函数里只有前两个碰数据，后四个都只描述「怎么算」。
 */
class Axpby : public mx::Primitive {
public:
    /**
     * 构造只拷贝两次标量，没有计算。alpha_ 与 beta_ 在这里被固化进算子，
     * 之后不能改：图节点不可变，改系数等于换一个算子。
     */
    explicit Axpby(mx::Stream stream, float alpha, float beta) : mx::Primitive(stream), alpha_(alpha), beta_(beta) {};

    /**
     * A primitive must know how to evaluate itself on the CPU/GPU
     * for the given inputs and populate the output array.
     *
     * To avoid unnecessary allocations, the evaluation function
     * is responsible for allocating space for the array.
     *
     * inputs  第一阶段登记的上游数组，已完成 dtype 对齐与形状广播。它们是
     *         视图：广播只在步长上做了文章，缓冲没有复制。
     * outputs 元素个数与形状已由前端确定、data 为空的那个数组。本函数负责
     *         让它有数据。
     */
    void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;

    /**
     * The Jacobian-vector product.
     *
     * 前向模式微分：给定上游的切向量 tangents，回答「输出沿这个方向变化多快」。
     * z = alpha*x + beta*y 是线性的，jvp 就是 alpha*tx + beta*ty，不需要另写
     * kernel —— 用已有的 op 在同一个 stream 上再建一小段图即可。反向传播因此
     * 也是惰性图的一部分，它同样要等 eval 才算。
     *
     * argnums 说明这次对哪几个输入求导，决定要不要把两项加起来。
     */
    std::vector<mx::array> jvp(const std::vector<mx::array> &primals, const std::vector<mx::array> &tangents,
                               const std::vector<int> &argnums) override;

    /**
     * The vector-Jacobian product.
     *
     * 反向模式微分：给定上游传来的伴随量 cotangents，回答每个输入该收多少梯度。
     * dz/dx = alpha、dz/dy = beta 都是常数，所以每个输入的梯度就是伴随量乘上
     * 对应的系数，逐元素的矩阵乘法在这里退化成一次标量乘。
     */
    std::vector<mx::array> vjp(const std::vector<mx::array> &primals, const std::vector<mx::array> &cotangents,
                               const std::vector<int> &argnums, const std::vector<mx::array> &outputs) override;

    /**
     * The primitive must know how to vectorize itself across
     * the given axes. The output is a pair containing the array
     * representing the vectorized computation and the axis which
     * corresponds to the output vectorized dimension.
     *
     * 本类没有实现，直接抛。未实现是合法状态：框架在需要 vmap 时会在第一个
     * 不支持它的节点上报错，而不是给出悄悄错误的结果。
     */
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &inputs,
                                                             const std::vector<int> &axes) override;

    /** Print the primitive. */
    void print(std::ostream &os);

    /** Name of the primitive (not virtual in some MLX versions). */
    const char *name() const override { return "Axpby"; }

    /**
     * Equivalence check
     *
     * 两个 Axpby 节点能不能合并成一个，取决于系数是否相同。漏比一个系数会让
     * 不同系数的算子被当成同一个，结果是静默的错数：没有异常、没有日志，只有
     * 数值不对。这也是 alpha_、beta_ 必须存成成员的原因 —— 存成成员，比较
     * 才有地方可做。
     */
    bool is_equivalent(const mx::Primitive &other) const override;

private:
    // 系数存成成员，不进 inputs：它们是算子自身的属性，is_equivalent 要比对它们。
    float alpha_;
    float beta_;
};

}  // namespace tiny_llm_ext
