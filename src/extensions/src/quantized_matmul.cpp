#include <sstream>
#include <stdexcept>
#include <string>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"
#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace tiny_llm_ext {

namespace {

[[noreturn]] void checkpoint_todo(const char *function, const char *checkpoint) {
    throw std::runtime_error(std::string(function) + " is a starter stub; implement it in " + checkpoint);
}

}  // namespace

// Week 2, Day 2. Later days extend the dispatch policy behind this API.
//
// 这个函数只登记，不算。四件事：检查参数、算出输出形状、建一个
// QuantizedMatmul 节点、返回它。返回的 mx.array 形状与 dtype 都对，
// data 为空，它记录的是「由谁算」，不是算出来的结果。分配显存与下发
// kernel 在 eval_gpu 里，Task 3 才写。
//
// 12 条检查按依赖分组，顺序不能换：后面的检查用前面的结论。
//
//   dtype，4 条
//     1 scales 必须是 bf16 或 f16       量化参数只有这两种精度
//     2 biases 与 scales 同 dtype       同一组的 scale 与 bias 配对使用
//     3 b 必须是 uint32                 每 8 个 4bit 码塞进一个 uint32
//     4 a 与 scales 同 dtype            混合精度乘加不在这个 kernel 的约定里
//
//   维度，2 条
//     5 a 是二维                       后面要取 a.shape()[1]，这一条必须先成立
//     6 b 是二维                       后面按行数与列数核对
//
//   取值，3 条
//     7 group_size == 128              kernel 里写死 128，别的值没有实现
//     8 bits == 4                      同上
//     9 transpose_b == true            b 按 (K, N) 存放，不转置的分支没有写
//
//   形状关系，3 条
//    10 scales 与 biases 同形          每个 (k, g) 都要有一对
//    11 b 的行数 == scales 的行数      两者都以 K 为行数
//    12 列数三组关系                  N 是激活的列数：
//         a 的列数能被 group_size 整除
//         scales 的列数 == a 的列数 / group_size
//         b 的列数 == a 的列数 / 8
//
// 建节点时输出形状从 a 复制，再把第 1 维换成 K = b.shape()[0]。
// inputs 的顺序被 eval_gpu 按下标取用，固定为 {scales, biases, a, b}。
mx::array quantized_matmul(const mx::array &scales, const mx::array &biases, int group_size, int bits,
                           const mx::array &a, const mx::array &b, bool transpose_b, bool use_simdgroup,
                           bool use_split_k, mx::StreamOrDevice s) {
    if (scales.dtype() != mx::bfloat16 && scales.dtype() != mx::float16) {
        throw std::runtime_error("quantized_matmul: scales must be bf16 or f16");
    }
    if (biases.dtype() != scales.dtype()) {
        throw std::runtime_error("quantized_matmul: biases must have the same dtype as scales");
    }
    if (b.dtype() != mx::uint32) {
        throw std::runtime_error("quantized_matmul: b must be uint32");
    }
    if (a.dtype() != scales.dtype()) {
        throw std::runtime_error("quantized_matmul: a must have the same dtype as scales");
    }
    if (a.ndim() != 2) {
        throw std::runtime_error("quantized_matmul: a must be a 2D array");
    }
    if (b.ndim() != 2) {
        throw std::runtime_error("quantized_matmul: b must be a 2D array");
    }
    if (group_size != 128) {
        throw std::runtime_error("quantized_matmul: group_size must be 128");
    }
    if (bits != 4) {
        throw std::runtime_error("quantized_matmul: bits must be 4");
    }
    if (!transpose_b) {
        throw std::runtime_error("quantized_matmul: b must be transposed");
    }
    if (scales.shape() != biases.shape()) {
        throw std::runtime_error("quantized_matmul: scales and biases must have the same shape");
    }
    if (b.shape()[0] != scales.shape()[0]) {
        throw std::runtime_error("quantized_matmul: b must have one row per output column");
    }
    const int packs_per_item = 32 / bits;
    if (a.shape()[1] % group_size != 0) {
        throw std::runtime_error("quantized_matmul: a columns must be divisible by group_size");
    }
    if (scales.shape()[1] != a.shape()[1] / group_size) {
        throw std::runtime_error("quantized_matmul: scales must have one column per input group");
    }
    if (b.shape()[1] != a.shape()[1] / packs_per_item) {
        throw std::runtime_error("quantized_matmul: b must have one word per groups of 8 codes");
    }

    // 输出形状是声明，不是分配：行数与 a 相同（每个激活行产出一行输出），
    // 列数是 K = b.shape()[0]。Shape 是变长数组，所以先拷贝 a 的形状再改第 1 维。
    // 这块形状要到 eval_gpu 里才变成显存，那时按 out.nbytes() 分配。
    auto out_shape = a.shape();
    out_shape[1] = b.shape()[0];

    // 建节点：形状与 dtype 描述结果，primitive 记录产生方式，inputs 记录依赖。
    // inputs 的顺序是约定，不是推导：eval_gpu 按下标取用，0 scales、1 biases、
    // 2 a、3 b。改这里必须同步改 eval_gpu，否则读到别的数组，形状对、数值错。
    // to_stream 把 variant 规约成 Stream：monostate 取当前线程的默认流，
    // Device 取该设备在本线程上的默认流，Stream 原样返回。
    return mx::array(out_shape, a.dtype(),
                     std::make_shared<QuantizedMatmul>(mx::to_stream(s), use_simdgroup, use_split_k),
                     {scales, biases, a, b});
}

// 只在有人把计算显式放到 CPU 流上时被调用。这个算子没有 CPU 实现，
// 第一句就抛，不留半成品。
void QuantizedMatmul::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("QuantizedMatmul runs only on the GPU");
}

// Task 3 填这里：inputs[0..3] 依次是 scales、biases、a、b，outputs[0] 是
// 形状已定、data 为空的输出数组，分配与编码都由这个函数自己做。
void QuantizedMatmul::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
#ifdef _METAL_
    auto &scales = inputs[0];
    auto &biases = inputs[1];
    auto &a = inputs[2];
    auto &b = inputs[3];
    auto &out = outputs[0];

    // 输出数组只有形状声明, data 为空; 显存由本函数分配, 分配之后 out 才有地方写.
    out.set_data(mx::allocator::malloc(out.nbytes()));

    // kernel 拿到的是 buffer 基址, 没有形状信息. 循环界与越界判断需要的
    // 三个长度必须作为标量显式传入: M = 输出行数(a 的行数),
    // N = 点积长度(a 的列数, 内层循环界), K = 输出列数(b 的行数, 转置存放).
    const int M = a.shape()[0];
    const int N = a.shape()[1];
    const int K = b.shape()[0];

    auto &s = stream();
    auto &d = mx::metal::device(s.device);

    std::ostringstream kname;
    kname << "quantized_matmul_vanilla_w4a16_g128";

    auto library = d.get_library("tiny_llm_ext");
    auto kernel = d.get_kernel(kname.str(), library);

    auto &compute_encoder = mx::metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);

    // 槽位契约: 数组走 set_input/output_array(绑定 buffer 基址),
    // 标量走 set_bytes. 槽号与 kernel 的 [[buffer(n)]] 一一对应.
    compute_encoder.set_input_array(scales, 0);
    compute_encoder.set_input_array(biases, 1);
    compute_encoder.set_input_array(a, 2);
    compute_encoder.set_input_array(b, 3);
    compute_encoder.set_output_array(out, 4);
    compute_encoder.set_bytes(M, 5);
    compute_encoder.set_bytes(N, 6);
    compute_encoder.set_bytes(K, 7);

    // dispatch_threadgroups: 第一参数是组数, 第二参数是每组的线程形状。
    // x 维沿 m、y 维沿 k, 组数按组尺寸向上取整; 驱动不挡越界,
    // 由 kernel 自己判断 m < M && k < K。
    //
    // 组尺寸的两个约束: 不超过 kernel 的硬件上限 tgp_size (Apple GPU ≤ 1024,
    // 寄存器吃多了会更低), 且尽量填满——组驻留 SM 后按整组占资源,
    // 半满的组浪费驻留位。
    //
    // x_size 是 m 方向的宽度。decode 时 M 常为 1, x 给大了整组都是进来
    // 就 return 的空线程, 所以 M 小时只给 16, 其余 (y_size = tgp_size/x)
    // 全给 k 方向干真活; M 大时给 32, m 方向少切几刀。
    // M=1 时 x_size=1 确实更省, 但 vanilla 的主顾是小 M 的 prefill 区间;
    // 真正的 decode (M ≤ 8) 在 Task 4 会切给专用 matvec kernel, 组形状
    // 完全不同。placeholder 阶段 matvec 也路由到这里, M=1 的空转税
    // 由 Task 4 消灭。
    const size_t tgp_size = kernel->maxTotalThreadsPerThreadgroup();
    const int x_size = M <= 16 ? 16 : 32;
    const int y_size = tgp_size / x_size;
    compute_encoder.dispatch_threadgroups(
        MTL::Size((M + x_size - 1) / x_size, (K + y_size - 1) / y_size, 1),
        MTL::Size(x_size, y_size, 1));
#else
    throw std::runtime_error("QuantizedMatmul has no GPU implementation.");
#endif
}
// Week 3, Day 4. The earlier Week 2 checkpoints keep the readable row lookup.
mx::array quantized_embedding(const mx::array &, const mx::array &, const mx::array &, const mx::array &, int, int,
                              mx::StreamOrDevice) {
    checkpoint_todo("quantized_embedding", "Week 3, Day 4");
}

void QuantizedEmbedding::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("QuantizedEmbedding::eval_cpu", "Week 3, Day 4");
}

void QuantizedEmbedding::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("QuantizedEmbedding::eval_gpu", "Week 3, Day 4");
}

}  // namespace tiny_llm_ext
