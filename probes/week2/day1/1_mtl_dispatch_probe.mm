// probes/week2/day1/1_mtl_dispatch_probe.mm
//
// 处理 10 个元素、每组安排 4 条线程时，主机可以请求 10 条线程，
// 也可以请求 3 个完整的组。前者形成 4+4+2 条线程，后者形成 4+4+4 条线程。
// 本探针让线程把自己所在组的实际大小写回缓冲，用输出观察这两种派发的差异。
// 随后再观察总线程数小于请求组尺寸、请求组尺寸超限时的本机行为。
//
// 每次运行先把 16 个槽位初始化为 -1，编号 gid<16 的线程执行 out[gid]=tgs。
// 输出同时展示命令缓冲错误状态、被写槽位数，以及这 16 个槽位的完整内容。
//
// 编译（仓库根目录）：
//   clang++ -fobjc-arc -std=c++17 -framework Metal -framework Foundation \
//     probes/week2/day1/1_mtl_dispatch_probe.mm -o /tmp/1_mtl_dispatch_probe
// 运行：
//   /tmp/1_mtl_dispatch_probe
//
// 开启 Metal API 校验，再运行同一个程序：
//   MTL_DEBUG_LAYER=1 /tmp/1_mtl_dispatch_probe
// 校验层检查主机调用的参数。本机实测，前三次合法请求执行完成，
// 第四次请求每组 2048 条线程时，校验层报告：
//   2048 must be <= 1024. (device threadgroup size limit)
// 随后断言终止进程（SIGABRT），后面的请求留待普通运行观察。
// 普通运行与校验运行共同展示：输出缓冲记录写入结果，API 校验定位非法调用；
// cb.error 记录命令缓冲的执行错误状态。
//
// ---------------------------------------------------------------------------
// 读代码前先备三条语法。这个文件是 Objective-C++（.mm），消息语法是它独有的：
//
//   [obj msg]                    给 obj 发一条消息，等同于调用它的 msg 方法
//   [obj msg:arg]                带一个参数；冒号属于方法名的一部分，写作 msg:
//   [obj a:arg1 b:arg2]          多参数；每个冒号前的词是参数的关键名
//   id<MTLDevice>                遵守 MTLDevice 协议的任意对象，带协议的类型标注
//   @"..."                       NSString 字面量，前面的 @ 是语法必需
//   @autoreleasepool { ... }     ARC 的释放池：作用域退出时回收临时对象
//
// 然后是 Metal 的四层对象，后一层挂在前一层上，每层回答一个问题：
//
//   MTL::Device               用哪块 GPU
//     └─ MTL::CommandQueue    从哪条通道提交；同一队列按提交顺序执行
//         └─ MTL::CommandBuffer   这一次提交装什么
//             └─ MTL::ComputeCommandEncoder   用什么往这个 buffer 里写命令
//
// 关键性质：命令是"记录"出来的，不是调用即执行。encoder 把每条命令写进
// command buffer，写完要 endEncoding 才能 commit；commit 之后 GPU 才开始跑，
// 而主机不等它。这一条解释了下面为什么每个 run_* 都要单独 commit + wait：
// 探针要逐个观察每次派发，所以一次提交只装一条命令。
//
// ---------------------------------------------------------------------------
// 怎么读输出：
//
//   被写槽位  前 16 个槽位中实际被写入的数量。本 kernel 只让 gid<16 的线程写入，
//             因此它直接观测的是这些线程的写入情况。为 0 表示这些槽位均保留初值。
//   首槽tgs   out[0] 的值，即 0 号线程所在组的实际大小。
//   out[0..15] 每个位置对应同编号线程写入的 tgs。grid=10、group=4 时，
//             out[0..7]=4，out[8..9]=2，out[10..15] 保留初值 -1。
//             这行直接展示末组只有两条线程。
//   err       命令缓冲的错误状态；与缓冲内容一起核对。Apple M4 上此前实测，
//             普通运行中组尺寸为 2048 的两次请求得到 err=no，16 个槽位均保留 -1。
//             开启 API 校验后的断言发生在编码派发命令时，见上方运行方式。
//
// 前三个合法请求、完整组对照的结果，以及本机此前观测到的超限行为：
//
//   grid=12   group=4      12 条线程，3 组，每组 4 个，tgs=4
//   grid=10   group=4      10 条线程，末组只有 2 条，out[8]=out[9]=2
//   grid=1000 group=1024   驱动把整组钳成 1000，tgs=1000
//   grid=4096 group=2048   超过 1024 上限：无 err，0 个槽位被写
//   dispatchThreadgroups 3 组 × 4    12 条线程（对照组）
//   dispatchThreadgroups 2 组 × 2048 同样超限：无 err，0 个槽位被写

#import <Metal/Metal.h>
#include <cstdio>

static const char* kSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;

// 每条被观测线程负责一个槽位：gid 决定写到哪里，tgs 决定写入什么。
// 主机读回后，便能看到各个被观测线程所在组的实际大小。
//
// 三个参数：
//   out   [[buffer(0)]] 与主机侧 setBuffer(buf, 0) 的 0 配对
//   gid   这条线程在整个 grid 中的 x 坐标；本探针使用一维派发，它就是全局编号。
//   tgs   这条线程所在线程组的实际 x 方向尺寸；本探针中它就是组内线程数。
//         Metal 在执行入口函数时提供这两个带属性的参数。
//
// grid=10、group=4 时，8、9 号线程都属于最后一个两线程组，各自写下 2。
// 后续二维探针用 uint3 接收 x、y、z 三个分量；本探针使用一维派发，
// 用 uint 接收 x 分量，gid 就是线程编号，tgs 就是该组的线程数。
[[kernel]] void probe(device int* out [[buffer(0)]],
                      uint gid [[thread_position_in_grid]],
                      uint tgs [[threads_per_threadgroup]]) {
  // 缓冲包含 out[0] 到 out[15]。gid<16 的线程各写一格，其余线程结束执行。
  // 16 来自这个探针选择的观测容量；派发的线程总数由主机的 dispatch 参数指定。
  if (gid < 16) { out[gid] = int(tgs); }
}
)METAL";

static id<MTLBuffer> make_buffer(id<MTLDevice> dev) {
  // newBufferWithLength:options: 创建数据缓冲，长度按字节算，这里 64 字节 = 16 个 int。
  // Apple Silicon 的 CPU 与 GPU 共用物理内存；MTLResourceStorageModeShared
  // 指定双方都可访问这块资源。CPU 通过下面的 contents 指针初始化和读取数据，
  // GPU 通过绑定到 [[buffer(0)]] 的 out 指针写入同一块缓冲。
  // 本探针先由 CPU 填入 -1，再提交 GPU 写入，waitUntilCompleted 返回后由 CPU
  // 读取结果。共享访问需要这样安排读写顺序，GPU 的结果可直接从缓冲读取。
  id<MTLBuffer> buf = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
  // [buf contents] 返回这块内存的 CPU 侧首地址，类型是 void*，转成 int* 按 4 字节读。
  int* o = (int*)[buf contents];
  // 初值 -1 是哨兵：kernel 写进去的是 threads_per_threadgroup（≥1），-1 表示没人写过。
  for (int i = 0; i < 16; ++i) o[i] = -1;
  return buf;
}

static void report(const char* tag, id<MTLCommandBuffer> cb, id<MTLBuffer> buf) {
  int* o = (int*)[buf contents];
  // 统计前 16 个槽位的写入情况。每个槽位由同编号线程负责，
  // 因此统计结果直接反映这些被观测线程是否写入；cb.error 另行打印。
  int written = 0;
  for (int i = 0; i < 16; ++i)
    if (o[i] >= 0) ++written;
  printf("%-10s err=%-3s 被写槽位=%2d 首槽tgs=%d\n",
         tag, cb.error ? "yes" : "no", written, o[0]);
  printf("  out[0..15] = [");
  for (int i = 0; i < 16; ++i)
    printf("%s%d", i == 0 ? "" : ", ", o[i]);
  printf("]\n");
  // %-10s 左对齐占 10 格，%-3s 左对齐占 3 格，%2d 右对齐占 2 格；
  // 这些都是 printf 的宽度修饰，只影响排版，不影响数值。
}

static void run_threads(id<MTLDevice> dev, id<MTLCommandQueue> q,
                        id<MTLComputePipelineState> ps, const char* tag,
                        unsigned grid, unsigned group) {
  id<MTLBuffer> buf = make_buffer(dev);

  // 下面六行就是四层对象的一次完整走法，逐行对号：
  //
  // [q commandBuffer]        第 3 层：向队列要一个空的命令缓冲，代表"这一次提交"。
  //                          此时它里面什么都没有，GPU 也不会因此启动。
  // [cb computeCommandEncoder]  第 4 层：在这个缓冲里开一个计算编码器。
  //                          一个缓冲可以装多个 encoder，命令按写入顺序排。
  // setComputePipelineState  绑定接下来要用哪个编译好的 kernel（源码 + 函数名
  //                          + 设备 三者编译出的 pipeline state）。
  // setBuffer:offset:atIndex:   把 buf 绑到槽位 0，对应 MSL 签名里的 [[buffer(0)]]。
  //                          offset 单位是字节，不是元素。
  // dispatchThreads          记录一条"启动线程"的命令，参数就是被量对象：
  //                          grid 给线程总数，group 给组内线程数。
  // endEncoding              结束这个 encoder 的记录；提交前可在同一个 command
  //                          buffer 上创建新的 encoder，继续记录后续命令。
  id<MTLCommandBuffer> cb = [q commandBuffer];
  // 第一行创建命令缓冲 `cb`，用于容纳这一次准备提交的命令。
  // 第二行取得编码器 `enc`，用于向 `cb` 记录计算命令。
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

  // 因此，后面的 **`[enc ...]` 都是在调用这个编码器的方法，记录 GPU 应当做的事情。**
  [enc setComputePipelineState:ps];
  [enc setBuffer:buf offset:0 atIndex:0]; // 记录：把数据缓冲 `buf` 绑定到参数槽位 0，对应 `probe` 的 `out [[buffer(0)]]`。

  [enc dispatchThreads:MTLSizeMake(grid, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
  /*
   * 整个方括号内是**一次方法调用**。方法名是 `dispatchThreads:threadsPerThreadgroup:`，两个冒号分别引出两个参数：
   * - `dispatchThreads:` 后面规定整个线程坐标范围。如果创建 12 条线程，全局 x 坐标为 `0` 到 `11`。
   * - `threadsPerThreadgroup:` 后面规定怎样分组。如果每组沿 x 方向安排 4 条线程，于是分为 `0–3`、`4–7`、`8–11` 三组。
   * `MTLSizeMake` 用三个整数构造一个尺寸值。这里 y、z 都为 1，所以只沿 x 方向安排线程。
   */
  [enc endEncoding];

  // commit 把整条缓冲交给队列，到此仍是异步的：主机往下走，GPU 可能还没开始。
  // waitUntilCompleted 阻塞到它执行完。探针必须先等再读缓冲，否则读到的还是初值。
  [cb commit];
  [cb waitUntilCompleted];
  printf("dispatchThreads      grid=%-5u group=%-5u ", grid, group);
  report(tag, cb, buf);
}

static void run_groups(id<MTLDevice> dev, id<MTLCommandQueue> q,
                       id<MTLComputePipelineState> ps, const char* tag,
                       unsigned groups, unsigned group) {
  // 与 run_threads 走同样的六步，唯一差别在 dispatch 一行的语义：
  // dispatchThreadgroups 的第一个参数是**组数**，线程总数 = 组数 × 组内线程数。
  // 因此这里不可能出现"末组不满"——凑不出整除时多出来的线程只能由 kernel 自己挡住。
  id<MTLBuffer> buf = make_buffer(dev);
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:ps];
  [enc setBuffer:buf offset:0 atIndex:0];
  [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  printf("dispatchThreadgroups groups=%-4u group=%-5u ", groups, group);
  report(tag, cb, buf);
}

int main() {
  @autoreleasepool {
    // MTLCreateSystemDefaultDevice() 返回系统默认 GPU；没有独立显卡的 Mac 上就是集显。
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    // 一条队列，下面所有探针共用；同一队列上的命令按提交顺序执行。
    id<MTLCommandQueue> q = [dev newCommandQueue]; // 调用 `dev` 对象的 `newCommandQueue` 方法，把返回的队列赋给 `q`。
    NSError* err = nil;

    // newLibraryWithSource:options:error: 在运行时把一段 MSL 源码编成库，
    // 所以这个探针是一个 .mm 文件，不依赖 metallib 与 CMake。
    // 源码用 C++11 的原始字符串 R"METAL( ... )METAL" 包住，内容里可以有引号与换行。
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                          options:nil
                                            error:&err];
    if (!lib) {
      printf("library: %s\n", [[err description] UTF8String]);
      return 1;
    }
    // 库里按函数名取出 kernel，编成 pipeline state；下面所有 dispatch 都用它。
    id<MTLComputePipelineState> ps =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"probe"] error:&err];

    printf("device=%s maxThreadsPerThreadgroup=%lu\n", [[dev name] UTF8String],
           (unsigned long)dev.maxThreadsPerThreadgroup.width);

    run_threads(dev, q, ps, "整齐", 12, 4);
    run_threads(dev, q, ps, "末组不满", 10, 4);
    run_threads(dev, q, ps, "钳位", 1000, 1024);
    run_threads(dev, q, ps, "单组超限", 4096, 2048);
    run_groups(dev, q, ps, "对照", 3, 4);
    run_groups(dev, q, ps, "单组超限", 2, 2048);
  }
  return 0;
}
