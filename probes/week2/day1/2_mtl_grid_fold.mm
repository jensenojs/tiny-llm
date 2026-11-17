// probes/week2/day1/2_mtl_grid_fold.mm
//
// 任务：给一个行优先矩阵的每个元素加 100，每条线程处理一个元素。
// 第一组输入是 3 行、4 列，缓冲依次存放 0..11；主机派发 grid=(4,3,1)，
// kernel 把 gid.y 当行、gid.x 当列，以 lin=gid.y*4+gid.x 访问输入和结果。
// 例如线程 gid=(3,2) 读取 input[11]=11，写入 result[11]=111。
//
// 输出先显示每条线程的位置、所属组、输入值与结果，再按行打印完整矩阵。
// 据此观察：矩阵行列怎样对应线程坐标；每组覆盖哪块矩阵区域；边缘组怎样缩小。
// 最后保持矩阵和 grid 不变，只换组形状，观察线程归属改变而逐元素计算结果相同。
// 当前任务中各元素独立计算；组内共享数据与同步协作需要相应的计算任务来展示。
//
// 线程组织：grid 包含 threadgroup，每个 threadgroup 包含线程。
// CUDA 对应 grid、block、thread。gid、tgp、tie 是这些对象的位置描述。
// uint3 是含 x、y、z 三个 uint 分量的向量类型；分量数描述坐标维度。
// 本探针用 x、y 表达二维位置，z 为 0；二维坐标也可以用 uint2 接收。
// 前一份一维探针用 uint 接收 x 分量。本文件选择 uint3，保留与三维尺寸的对应。
//
// ===========================================================================
// dispatch 的两个参数
// ===========================================================================
//
// 主机这样调用：
//
//   [enc dispatchThreads:MTLSizeMake(gx, gy, 1)
//       threadsPerThreadgroup:MTLSizeMake(bx, by, 1)];
//
// MTLSize 没有二维版本，永远是三个数：(width, height, depth)。每个数表示这一维
// 上有多少个位置，而且第一个数绑 gid.x、第二个绑 gid.y、第三个绑 gid.z：
//
//   MTLSize 分量   线程坐标   本探针   这个数决定的取值范围
//   width          gid.x      gx      0 .. gx-1
//   height         gid.y      gy      0 .. gy-1
//   depth          gid.z      1       0 .. 0
//
// 本任务选择 x 对应列、y 对应行。矩阵 shape 按 (行数,列数) 写，
// Metal 尺寸按 (宽,高,深) 写，因此 3 行 4 列的矩阵对应 grid=(4,3,1)。
// width 是宽度，height 是高度：沿用图像坐标的叫法，x 表示横向位置，y 表示纵向位置。
// 线程坐标与矩阵元素的对应关系由 kernel 的寻址代码落实。
//
// 第一个参数描述整个 grid，第二个描述一个线程组。这两个三元组是探针的输入，
// 它们规定坐标范围与分组尺寸，每条线程在执行时取得自己的位置。
//
// ===========================================================================
// 线程坐标从哪来
// ===========================================================================
//
// 每条线程拿到三个整数 gid = (gid.x, gid.y, gid.z)，取值范围由 grid 三元组决定：
//
//   gid.x 从 0 到 gx-1        宽度方向
//   gid.y 从 0 到 gy-1        高度方向
//   gid.z 从 0 到 gz-1        深度方向
//
// 这条范围规则直接说明"深度填 1 时 gid.z 恒为 0"：1 对应的取值区间是 0..(1-1)，
// 只有 0 这一个元素，于是每条线程的 gid.z 都是 0。它没有消失，只是没有变化。
//
// 三个维度不必都用，常见三种填法：
//
//   负载              grid 三元组       哪些分量只取 0      实际用到的分量
//   axpby 逐元素      (nelem, 1, 1)     gid.y、gid.z        只有 gid.x
//   本探针            (4, 3, 1)         gid.z               gid.x、gid.y
//   三维卷积/体渲染   (W, H, D)         无                  三个都用
//
// axpby 的 kernel 能把参数声明成标量 uint index，靠的就是它填了 (nelem,1,1)：
// 只有 .x 会变，取 .x 就等于取到全部信息。
//
// ===========================================================================
// 从行列坐标找到缓冲中的元素
// ===========================================================================
//
// 矩阵在缓冲里逐行存放：第 0 行的 4 个元素，接着第 1 行，再接第 2 行。
// 这种存储顺序称为行优先。第 y 行、第 x 列之前有 y*W+x 个元素，
// 所以缓冲下标 lin=y*W+x；这里 W 是列数。
//
//   矩阵下标 [行][列] = [y][x]
//   线程坐标 (x,y)
//   缓冲下标 y*W+x
//
// 例如 gid=(3,2)：先跳过 2 行共 8 个元素，再向右走 3 格，得到 lin=11；
// 于是 result[11]=input[11]+100=111。
//
// 本任务让 grid 宽度等于矩阵列数，因此从 tpg.x 取得 W。
// 代码保留三维写法 lin=x+W*y+W*H*z；当前 z=0，使用的就是 y*W+x。
// shape 的 (行,列) 与线程的 (x,y) 是两套参数顺序，内存采用上述行优先排列。
//
// ===========================================================================
// 图：12 条线程的两种描述
// ===========================================================================
//
//           列 x（本探针中 x 每加 1，一维编号加 1）→
//           x=0      x=1      x=2      x=3
//         ┌────────┬────────┬────────┬────────┐
//  行 y   │ (0,0)  │ (1,0)  │ (2,0)  │ (3,0)  │
//  步长 4 │ lin 0  │ lin 1  │ lin 2  │ lin 3  │
//   ↓     ├────────┼────────┼────────┼────────┤
//         │ (0,1)  │ (1,1)  │ (2,1)  │ (3,1)  │
//         │ lin 4  │ lin 5  │ lin 6  │ lin 7  │
//         ├────────┼────────┼────────┼────────┤
//         │ (0,2)  │ (1,2)  │ (2,2)  │ (3,2)  │
//         │ lin 8  │ lin 9  │ lin 10 │ lin 11 │
//         └────────┴────────┴────────┴────────┘
//
// 把三行接成一串，就是 lin 的全集：
//
//   lin:   0  1  2  3 │ 4  5  6  7 │ 8  9  10 11
//          └── y=0 ──┘ └── y=1 ──┘ └── y=2 ───┘
//
// 验算一格：(2,1) → 2 + 4*1 = 6，与图上 lin 6 一致。
//
// 组按坐标在 grid 上切，与上述 lin 的计算方式无关。本例组宽高是 2×2：
// 横向两组、纵向两组（第二排只有一行）。左上组的 lin 是 0、1、4、5，
// 在一维缓冲里不是连续的一段。
//
//            x=0   x=1        x=2   x=3
//   y=0     ┌──────────┐    ┌──────────┐
//           │ tgp(0,0) │    │ tgp(1,0) │
//   y=1     │ 4 条线程 │    │ 4 条线程 │
//           └──────────┘    └──────────┘
//   y=2     ┌──────────┐    ┌──────────┐
//           │ tgp(0,1) │    │ tgp(1,1) │   ← 只剩一行：3 除不尽 2
//           │ 2 条线程 │    │ 2 条线程 │
//           └──────────┘    └──────────┘
//
// ===========================================================================
// 同一条线程的三种位置描述
// ===========================================================================
//
//   gid   这条线程在整个 grid 中的位置
//   tgp   这条线程所属组在 grid 中的位置
//   tie   这条线程在所属组内部的位置
//
// 例如 grid=(4,3)、请求组尺寸=(2,2)，右下角线程：
//   gid=(3,2)，所属组 tgp=(1,1)，组内 tie=(1,0)，实际组尺寸 tgs=(2,1)。
//   x: 3 = 1*2 + 1；y: 2 = 1*2 + 0。
// 两个乘法中的 2 来自主机请求的组尺寸 (bx,by)，用来确定各组起点的间距。
// tgs 是当前组的实际尺寸，边缘组会缩小；tpg 是整个 grid 的尺寸。
// 因此各方向上的位置关系为：
//   gid.x = tgp.x * bx + tie.x
//   gid.y = tgp.y * by + tie.y
//
// ===========================================================================
// 探针要观察的关系
// ===========================================================================
//
//   (1) 探针按自己选的公式写出的缓冲槽位覆盖 0..11，且没有重复；
//   (2) 第一组请求按 2×2 划分线程，底部边缘组只占一行；
//   (3) 每行同时显示 gid、tgp、tie、tgs 的 x、y，能直接读出线程归属与末组尺寸。
//
// 第二次运行换成 grid=(12,1)、组=(4,1)：仍有 12 条线程，坐标和分组都变了；
// 两种派发各自算出的缓冲槽位仍覆盖 0..11。
// 第三次换成 grid=(3,4)、组=(2,2)，观察右侧边缘组实际宽度缩小为 1。
// 第四次回到 grid=(4,3)，组换为 (4,1)，每组覆盖一整行。
// 与第一次相比，矩阵、全局线程坐标与计算结果相同，所属组及组内位置改变。
//
// ===========================================================================
// 怎么读输出
// ===========================================================================
//
// 每行是图里一个格子，lin 是本探针为这条记录选择的存放编号。
// 后面四对数依次是：全局坐标 gid、所属组坐标 tgp、组内坐标 tie、实际组尺寸 tgs。
// input、result 是该线程读取的输入元素与写出的计算结果。
// 后面的矩阵每行分别打印输入、结果；行号对应 gid.y，列号对应 gid.x。
// -1 是缓冲的初值，读到 -1 表示对应字段保留了初值。
//
// ---------------------------------------------------------------------------
// 读代码前备三条语法。这个文件是 Objective-C++（扩展名 .mm）：
//
//   [obj msg]                    给 obj 发一条消息，等同于调用它的 msg 方法
//   [obj msg:arg]                带一个参数；冒号属于方法名的一部分
//   [obj a:arg1 b:arg2]          多参数；每个冒号前的词是参数的关键名
//   id<MTLDevice>                遵守 MTLDevice 协议的任意对象
//   @autoreleasepool { ... }     ARC 的释放池：作用域退出时回收临时对象
//   R"METAL( ... )METAL"         C++11 原始字符串，内容原样保留，不需要转义引号
//
// 然后是 Metal 的四层对象，后一层挂在前一层上，每层回答一个问题：
//
//   MTL::Device                         用哪块 GPU
//     └─ MTL::CommandQueue              从哪条通道提交；同一队列按提交顺序执行
//         └─ MTL::CommandBuffer         这一次提交装什么
//             └─ MTL::ComputeCommandEncoder   用什么往这个 buffer 里写命令
//
// 命令是记录出来的，不是调用即执行：encoder 把 dispatchThreads 这类命令写进
// command buffer，写完要 endEncoding 才能 commit；commit 之后 GPU 才开始跑。
//
// MSL 源码里的 [[kernel]] 是"登记为可从主机调用的入口"的属性，主机侧
// newFunctionWithName:@"probe" 按**函数名**查它，两者名字必须逐字一致。
// 坐标参数（gid / tgp / tie / tgs / tpg）由硬件在启动线程时注入，
// 数据参数通过 setBuffer 绑定：槽位 0 是坐标记录，1 是矩阵输入，2 是矩阵结果。
//
// 编译（仓库根目录）：
//   clang++ -fobjc-arc -std=c++17 -framework Metal -framework Foundation \
//     probes/week2/day1/2_mtl_grid_fold.mm -o /tmp/2_mtl_grid_fold
// 运行：
//   /tmp/2_mtl_grid_fold

#import <Metal/Metal.h>
#include <cstdio>

static const char* kSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;

// 每条线程读取一个矩阵元素、加 100、写到结果矩阵的同一位置，
// 并把坐标记录下来，让主机把线程组织与实际数据访问并排打印。
//
// 五个位置与尺寸参数由 Metal 提供。每个都是 uint3，
// 即三个 uint 捆在一起，分量依次是 .x / .y / .z：
//   gid  这条线程在 grid 里的坐标             (thread_position_in_grid)
//   tie  它在所属线程组内的坐标               (thread_position_in_threadgroup)
//   tgp  它所属线程组在 grid 里的坐标         (threadgroup_position_in_grid)
//   tgs  它所属线程组的大小                   (threads_per_threadgroup)
//   tpg  整个 grid 的大小                     (threads_per_grid)
// 三个数据缓冲由主机绑定：
//   out    [[buffer(0)]] 每条线程的坐标记录
//   input  [[buffer(1)]] 行优先矩阵输入
//   result [[buffer(2)]] 同形状的结果矩阵
//
// 只在 .x 有用时可以声明成标量 uint，直接拿到那个分量（axpby 就是这么做的，
// 因为它的 grid 填了 (nelem,1,1)，.y 与 .z 没有变化）。
[[kernel]] void probe(device int* out [[buffer(0)]],
                      device const int* input [[buffer(1)]],
                      device int* result [[buffer(2)]],
                      uint3 gid [[thread_position_in_grid]],
                      uint3 tie [[thread_position_in_threadgroup]],
                      uint3 tgp [[threadgroup_position_in_grid]],
                      uint3 tgs [[threads_per_threadgroup]],
                      uint3 tpg [[threads_per_grid]]) {
  // 本任务让矩阵列数等于 tpg.x：跳过 gid.y 行，再前进 gid.x 个元素。
  // 第三项用于跨越完整的二维层；本探针 gid.z=0。
  uint lin = gid.x + tpg.x * gid.y + tpg.x * tpg.y * gid.z;

  // 本任务约定矩阵形状为 (tpg.y,tpg.x)，每条线程对应 [gid.y][gid.x]。
  // lin 同时是这个元素在行优先矩阵缓冲中的下标。
  result[lin] = input[lin] + 100;

  // 每条线程记录四对 x、y，共 8 个 int；第 lin 条记录从 8*lin 开始。
  // lin=0 占 [0..7]，lin=1 占 [8..15]，lin=11 占 [88..95]。
  // 主机按相同的字段顺序读回，每条线程占输出表格的一行。
  out[8 * lin + 0] = int(gid.x);
  out[8 * lin + 1] = int(gid.y);
  out[8 * lin + 2] = int(tgp.x);
  out[8 * lin + 3] = int(tgp.y);
  out[8 * lin + 4] = int(tie.x);
  out[8 * lin + 5] = int(tie.y);
  out[8 * lin + 6] = int(tgs.x);
  out[8 * lin + 7] = int(tgs.y);
}
)METAL";

static void run(unsigned gx, unsigned gy, unsigned bx, unsigned by) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                          options:nil
                                            error:&err];
    if (!lib) {
      printf("library: %s\n", [[err description] UTF8String]);
      return;
    }
    id<MTLComputePipelineState> ps =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"probe"] error:&err];

    // 每条线程写 8 个 int；按线程总数 n 分配 8 * sizeof(int) * n 字节。
    unsigned n = gx * gy;
    id<MTLBuffer> buf = [dev newBufferWithLength:8 * sizeof(int) * n options:MTLResourceStorageModeShared];
    // 初值 -1：可据此识别尚未写入的字段，坐标 0 则会被正常打印。
    int* o = (int*)[buf contents];
    for (unsigned i = 0; i < 8 * n; ++i) o[i] = -1;

    // 矩阵形状为 gy 行、gx 列；缓冲按行存放 n 个 int。
    // 输入依次填 0..n-1，结果初始化为 -1，执行后预期为 100..100+n-1。
    id<MTLBuffer> input = [dev newBufferWithLength:sizeof(int) * n options:MTLResourceStorageModeShared];
    id<MTLBuffer> result = [dev newBufferWithLength:sizeof(int) * n options:MTLResourceStorageModeShared];
    int* in = (int*)[input contents];
    int* values = (int*)[result contents];
    for (unsigned i = 0; i < n; ++i) {
      in[i] = int(i);
      values[i] = -1;
    }

    // 一次完整的派发，每行一件事：
    //   [q commandBuffer]            向队列要一个空的命令缓冲（这次提交的容器）
    //   [cb computeCommandEncoder]   在这个缓冲里开一个计算编码器
    //   setComputePipelineState      绑定要跑的 kernel
    //   setBuffer:offset:atIndex:    绑定坐标记录、矩阵输入、矩阵结果到槽位 0、1、2
    //   dispatchThreads              记一条启动线程的命令，形状由两个 MTLSize 给
    //   endEncoding                  结束当前 encoder 的记录；提交前可另开 encoder
    //   commit                       交给队列；到这一步 GPU 可能还没开始
    //   waitUntilCompleted           阻塞到执行完，之后才能安全读缓冲
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc setBuffer:input offset:0 atIndex:1];
    [enc setBuffer:result offset:0 atIndex:2];
    // 两个 MTLSize 分别给 grid 与组的形状；这里两个维度都可能是非 1，
    // 所以用 MTLSizeMake(gx, gy, 1) 而不是 axpby 里那个一维写法。
    [enc dispatchThreads:MTLSizeMake(gx, gy, 1) threadsPerThreadgroup:MTLSizeMake(bx, by, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    printf("grid=(%u,%u) group=(%u,%u) err=%s\n", gx, gy, bx, by, cb.error ? "yes" : "no");
    printf("  lin : gid(x,y)  tgp(x,y)  tie(x,y)  tgs(x,y) | input -> result\n");
    for (unsigned i = 0; i < n; ++i)
      printf("  %3u : (%2d,%2d)  (%2d,%2d)  (%2d,%2d)  (%2d,%2d) | %3d -> %3d\n",
             i, o[8 * i], o[8 * i + 1], o[8 * i + 2], o[8 * i + 3],
             o[8 * i + 4], o[8 * i + 5], o[8 * i + 6], o[8 * i + 7], in[i], values[i]);

    printf("  matrix shape=(%u,%u): input -> result (each row)\n", gy, gx);
    for (unsigned row = 0; row < gy; ++row) {
      printf("  row %u: [", row);
      for (unsigned col = 0; col < gx; ++col)
        printf("%s%d", col == 0 ? "" : ", ", in[row * gx + col]);
      printf("] -> [");
      for (unsigned col = 0; col < gx; ++col)
        printf("%s%d", col == 0 ? "" : ", ", values[row * gx + col]);
      printf("]\n");
    }
  }
}

int main() {
  // 二维：grid 宽 4 高 3，组 2×2。组在宽和高两个方向都要切，因此能看到
  // 除不尽时第二排组只覆盖一行。
  // 每格为输入元素；分隔线为组边界：
  //   0  1 |  2  3
  //   4  5 |  6  7
  //   -----+------
  //   8  9 | 10 11
  run(4, 3, 2, 2);
  // 同一批 12 条线程改成一维：grid 宽 12 高 1，此时 gid.y 只取 0，
  // 每条线程的 gid.x 就是 0..11。run 同时把矩阵形状改为 (1,12)：
  //   0 1 2 3 | 4 5 6 7 | 8 9 10 11
  // 每组覆盖连续四个元素；缓冲内容和逐元素结果保持相同。
  run(12, 1, 4, 1);
  // 横向末组不满：最右一列线程的实际组宽为 1，组起点仍按请求宽度 2 排列。
  run(3, 4, 2, 2);
  // 保持第一组的矩阵与 grid，只改变组形状：每组一行，结果仍是逐元素加 100。
  //    0  1  2  3
  //   -----------
  //    4  5  6  7
  //   -----------
  //    8  9 10 11
  // 与第一组请求对照：2×2 组覆盖小块，4×1 组覆盖整行。
  // 本任务各元素独立；需要组内共享数据或同步时，组覆盖区域应匹配协作任务。
  run(4, 3, 4, 1);
  return 0;
}
