// probes/week2/day1/3_mtl_submit_batching.mm
//
// 问题：把 K 个 kernel 装进一次提交，与逐个提交，每个 kernel 的固定开销差多少。
//       对应 docs/mlx-array-lifecycle.md:215-236 那组数字（1 个 194.3 us、
//       100 个攒一次 538.9 us、100 个各自 eval 14664.4 us）。
//
// 三种提交形态，都用同一个 kernel、grid=1 线程（GPU 侧工作量趋近于零），
// 所以量到的时间几乎全部是提交链的固定开销：
//
//   [逐个+等待]  K 次 (取缓冲 → 编码 → commit → wait)。每次都要等 GPU 跑完，
//                主机与 GPU 完全串行，这是"每个算子单独 eval"的形态。
//   [逐个+不等待] K 次 (取缓冲 → 编码 → commit)，最后只等最后一个。
//                队列按提交顺序执行，等最后一个等于等全部；主机在 GPU 跑第一条
//                时已经在记录后面的命令，这就是 MLX 平时的形态。
//   [一次提交]   1 个缓冲里连开 K 个 encoder，各写一条 dispatch，
//                最后 commit 一次、wait 一次。
//
// 编译（仓库根目录）：
//   clang++ -fobjc-arc -std=c++17 -framework Metal -framework Foundation \
//     probes/week2/day1/3_mtl_submit_batching.mm -o /tmp/3_mtl_submit_batching
// 运行：
//   /tmp/3_mtl_submit_batching
//
// 怎么读输出：每行是"某个形态、某个 K"的总耗时中位数与每 kernel 均摊。预期形状是
// K=100 时 逐个+等待 ≫ 逐个+不等待 > 一次提交；K=1 时三行相差不大，因为只有一次
// 提交，摊薄还没发生。若 K=1 的三行就拉开几十倍，说明预热不足或后台有负载。
//
// ---------------------------------------------------------------------------
// 语法速查（Objective-C++，.mm 文件）：
//
//   [obj msg]                    无参消息
//   [obj msg:arg]                一个参数；冒号属于方法名，写作 "msg:"
//   [obj a:x b:y]                多参数；选择器是 "a:b:"
//   [[a b] c]                    嵌套：内层结果当外层的接收者
//   obj.name                     等价于 [obj name]，只对无参取值有效
//   向 nil 发消息返回 0/nil，不崩溃
//
// 本文件用到的五个 Metal 对象，后一个挂在前一个上：
//
//   MTLDevice                 GPU 本身
//   MTLCommandQueue           提交通道；同一队列按提交顺序执行
//   MTLCommandBuffer          这一次提交装的东西
//   MTLComputeCommandEncoder  往这次提交里写命令的接口
//   MTLComputePipelineState   某个 kernel 编译后的可执行形态
//
// MSL 源码里的 [[kernel]] 是"登记为可从主机调用的入口"的属性；主机侧
// newFunctionWithName:@"touch" 按**函数名**查它。两个名字必须逐字一致，
// 忘了标 [[kernel]] 或名字打错，症状相同：查不到函数，pipeline 建不出来
// （向 nil 继续发消息也不会崩，只是拿到 nil）。
// ---------------------------------------------------------------------------

#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

static const char* kSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;
[[kernel]] void touch(device int* out [[buffer(0)]]) {
  out[0] = 1;
}
)METAL";

enum Form { SEPARATE_WAIT, SEPARATE_NO_WAIT, ONE_BUFFER };
// C++ 的 enum 只是个整数常量集合，上面这行给三种提交形态各起一个名字，
// 后面 run_form 用它在三条分支里选一条。

static void encode_one(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps,
                       id<MTLBuffer> buf) {
  // 编一条 dispatch 要做的四件事固定不变：装内核、绑缓冲、写命令、收尾。
  [enc setComputePipelineState:ps];
  [enc setBuffer:buf offset:0 atIndex:0];
  [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  [enc endEncoding];
}

static void run_form(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLComputePipelineState> ps,
                     id<MTLBuffer> buf, Form form, int k, int reps, const char* label) {
  // 同一个形态重复 reps 次，每次记一个总耗时；最后取中位数。
  std::vector<double> totals;
  for (int r = 0; r < reps; ++r) {
    // steady_clock 是单调时钟，不受系统对时影响；duration<double, std::micro>
    // 把时间差换算成浮点微秒。
    auto t0 = std::chrono::steady_clock::now();

    if (form == ONE_BUFFER) {
      // 一个命令缓冲装 K 条命令。encoder 可以开 K 个，缓冲只有一个，
      // 所以 commit 与 wait 各只发生一次。
      id<MTLCommandBuffer> cb = [q commandBuffer];
      for (int i = 0; i < k; ++i) {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        encode_one(enc, ps, buf);
      }
      [cb commit];
      [cb waitUntilCompleted];
    } else {
      // K 个命令缓冲，各自 commit。区别只在要不要每次 wait。
      // last 只用来在最后等一次：队列按提交顺序执行，等最后一个等于等全部，
      // 这也正是"不等待"那一档能把主机与 GPU 重叠起来的原因。
      id<MTLCommandBuffer> last = nil;
      for (int i = 0; i < k; ++i) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        encode_one(enc, ps, buf);
        [cb commit];
        if (form == SEPARATE_WAIT) [cb waitUntilCompleted];
        last = cb;
      }
      if (form == SEPARATE_NO_WAIT) [last waitUntilCompleted];
    }

    auto t1 = std::chrono::steady_clock::now();
    totals.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  // 单次计时的噪声能差出一倍，取中位数（reps 次里排序后中间那个）。
  std::sort(totals.begin(), totals.end());
  double us = totals[reps / 2];
  printf("%-14s K=%-4d 中位数 %10.1f us   每 kernel %8.2f us\n", label, k, us, us / k);
}

int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                          options:nil
                                            error:&err];
    if (!lib) {
      printf("library: %s\n", [[err description] UTF8String]);
      return 1;
    }
    id<MTLComputePipelineState> ps =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"touch"] error:&err];
    // `StorageModeShared` 的意思是这块 buffer 对 CPU 和 GPU 用**同一个物理地址**
    // Apple Silicon 上 CPU 和 GPU 本来就连着同一片内存，不需要 PCIe 拷贝，
    //「给 GPU 一块显存」实际上只是「划一块内存，把地址告诉 GPU」。
    // 这就是 `set_data(mx::allocator::malloc(...))` 之后 `set_output_array` 能直接绑地址的原因。
    id<MTLBuffer> buf = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];

    // 预热：第一次提交要建 pipeline 的各种状态，不预热会把一次性成本算进第一组。
    for (int i = 0; i < 3; ++i) run_form(dev, q, ps, buf, SEPARATE_WAIT, 1, 1, "warmup");

    // K 取 1 与 100：1 看的是"一次提交本身的成本"，100 看的是摊薄之后每个 kernel 的成本。
    // {1, 100} 是 C++ 的初始化列表，range-for 依次取这两个值。
    for (int k : {1, 100}) {
      run_form(dev, q, ps, buf, SEPARATE_WAIT, k, 5, "逐个+等待");
      run_form(dev, q, ps, buf, SEPARATE_NO_WAIT, k, 5, "逐个+不等待");
      run_form(dev, q, ps, buf, ONE_BUFFER, k, 5, "一次提交");
      printf("\n");
    }
  }
  return 0;
}
