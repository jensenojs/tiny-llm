#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Starter interface map. Implement the named kernels at these checkpoints;
// their argument lists are defined by the matching C++ encoder you complete.
//
// Week 2, Day 2:
//   quantized_matmul_vanilla_w4a16_g128
// Week 2, Day 3:
//   quantized_matvec_x4_fast_w4a16_g128
// Week 2, Day 4:
//   quantized_matmul_simdgroup_w4a16_g128
// Week 3, Day 4:
//   quantized_embedding_w4a16_g128
//
// The x2/x8 tuning variants in the reference extension are deliberately not
// starter interfaces. Add an experimental variant only while running the
// optional scheduling comparison, then keep the selected course path.

// Week 2, Day 2 / Task 3: vanilla 版。一条线程负责一个输出元素 out[m,k],
// 串行扫完整条长度为 N 的点积。解包结果只活在寄存器里, 不物化权重矩阵。
// 槽位契约与 quantized_matmul.cpp 的 eval_gpu 一一对应:
//   数组 0 scales / 1 biases / 2 a / 3 b, 输出 4, 标量 5 M / 6 N / 7 K.
// dispatch 的 x 维沿 m、y 维沿 k, 组数向上取整, 越界线程直接返回。
kernel void quantized_matmul_vanilla_w4a16_g128(
    device const bfloat16_t* scales [[buffer(0)]],
    device const bfloat16_t* biases [[buffer(1)]],
    device const bfloat16_t* a      [[buffer(2)]],
    device const uint32_t*  b       [[buffer(3)]],
    device bfloat16_t*      out     [[buffer(4)]],
    constant int& M [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& K [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint m = tid.x;
    uint k = tid.y;
    if (m >= uint(M) || k >= uint(K)) return;

    const int words_per_row  = N / 8;    // b 每行的 uint32 数, 8 个 4bit 码一个字
    const int groups_per_row = N / 128;  // scales/biases 每行的组数, 128 个元素一组

    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        // 对照 quantize.py 的 step1: Python 版对整个数组一次算完,
        // weight[..., None] >> shifts 靠广播把每个 uint32 同时右移
        // [0,4,...,28] 八种量, 得到 (K, M, 8) 再摊平。
        // 这里一条线程只负责一个元素 n, 不需要广播: 先定位 n 住在哪个字
        // (第 n/8 个 uint32, 行首偏移 k*words_per_row), 再只算它自己的
        // 那一次移位。n%8 就是这个元素在字里的槽位, 低位优先, 所以
        // 移位量是 4*(n%8)——对应 Python 里 shifts[n%8] 那一项。
        uint32_t packed = b[k * words_per_row + n / 8];
        uint q = (packed >> (4 * (n % 8))) & 0xF;  // 0xF 就是 Python 的 mask

        // 对照 step2: Python 版把 q reshape 成 (K, G, S) 再让 scales 广播;
        // 这里用下标直接算出 n 属于第几组 (n/128), 行首偏移 k*groups_per_row。
        // 公式相同: w = q*scale + bias。
        float w = float(q) * float(scales[k * groups_per_row + n / 128])
                + float(biases[k * groups_per_row + n / 128]);
        acc += float(a[m * N + n]) * w;
    }
    out[m * K + k] = bfloat16_t(acc);
}
