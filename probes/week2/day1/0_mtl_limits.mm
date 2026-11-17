// probes/week2/day1/0_mtl_limits.mm
//
// 派发 GPU 计算前，主机需要为实际执行的函数查询线程组限制。这个程序先准备
// 一个空的 GPU 函数 k，再打印设备与该函数的属性。沿 main 的顺序读：
//
//   dev：系统默认 GPU 的接口对象
//     → 用 dev 编译函数源码，得到函数库 lib
//     → 从 lib 按名字取出 k
//     → 用 dev 和 k 创建计算管线 ps（MTLComputePipelineState）
//     → 从 dev、ps 读取下列属性
//
// kernel 在这里指可从主机派发的 GPU 函数，源码为 [[kernel]] void k() {}。
// ps 是针对这台设备、这个函数准备的执行对象。后续记录计算命令时，主机通过
// setComputePipelineState 选择 ps，再指定数据与派发尺寸。同一个函数多次处理
// 不同数据时，可以复用 ps；数据地址与启动多少线程由各次派发分别指定。
//
// 要读懂输出，先分开两层分组：
// 主机把一次计算的线程划分为 threadgroup（CUDA 称 block）。同一 threadgroup
// 的线程可以共享组内内存、进行组内同步。执行时，每个 threadgroup 的线程又被
// 组织成 SIMD 组（对应 NVIDIA CUDA 的 warp），组内线程执行共同的指令，
// 各自处理自己的数据。
//
// 三行属性分别回答：
//
//   device.maxThreadsPerThreadgroup
//     一组线程可以用三个整数 (a,b,c) 安排坐标，共有 a*b*c 条线程。三个整数
//     是每个坐标分量可取的位置数，可以用于任意计算任务；width、height、depth
//     是 Metal 对这三个分量的命名。设备返回的 MTLSize 分别限制 a、b、c。
//
//   pipeline.maxTotalThreadsPerThreadgroup
//     执行 k 时，一组允许包含的线程总数上限，限制上述乘积 a*b*c。
//     一组的三个尺寸各自满足设备限制后，乘积还须满足这项限制。
//     此值取决于设备与具体函数需要的资源，同一设备上不同 ps 的值可能不同。
//
//   pipeline.threadExecutionWidth
//     这个 ps 的每个 SIMD 组包含多少条线程。它描述 threadgroup 内的执行分组。
//     线程组的容量限制与 SIMD 组宽度，是两个分别查询的属性。
//
// 本机 Apple M4 的一次查询结果：
//   device.maxThreadsPerThreadgroup       = (1024,1024,1024)
//   pipeline.maxTotalThreadsPerThreadgroup = 1024
//   pipeline.threadExecutionWidth         = 32
// 前两个结果合起来要求 a<=1024、b<=1024、c<=1024，且 a*b*c<=1024。
// 这些数约束单个 threadgroup；整台 GPU 某时刻执行多少线程需另行研究硬件调度。
// 最后的 32 表示每个 SIMD 组的线程数。若将来派发一个包含 1024 条线程的组，
// 它会包含 1024/32=32 个 SIMD 组；这里的派发规模是假设示例。
//
// ---------------------------------------------------------------------------
// Objective-C++ 语法：
//
// 消息发送（方括号）：
//
//   [receiver selector]          向对象发一条无参消息
//   [receiver selector:arg]      一个参数；冒号属于选择器名，写作 "selector:"
//   [receiver a:x b:y]           多参数；选择器是 "a:b:"，每个冒号前的词是参数标签
//   obj.name                     等价于 [obj name]，只适用于无参取值，是语法糖
//   nil                          Objective-C 的空对象值
//
// 嵌套：[[err description] UTF8String]
//   先向 err 要 description（得到 NSString），再向这个 NSString 要 UTF8String
//   （得到 char* 给 printf 的 %s）。等价于 C 的 UTF8String(description(err))。
//
// [[kernel]] 与函数名的关系：
//   MSL 源码"[[kernel]] void k() {}" 同时定两件事：函数名 k，以及 [[kernel]] 属性。
//   [[kernel]] 标记计算入口；newFunctionWithName:@"k" 按名字 k 取出该入口。
//   属性与函数名各有职责，字符串 @"k" 必须与源码中的函数名一致。
// ---------------------------------------------------------------------------
//
// 编译（仓库根目录）：
//   clang++ -fobjc-arc -std=c++17 -framework Metal -framework Foundation \
//     probes/week2/day1/0_mtl_limits.mm -o /tmp/0_mtl_limits
// 运行：
//   /tmp/0_mtl_limits

#import <Metal/Metal.h>
#include <cstdio>

int main() {
  // 自动释放池：作用域退出时释放池中登记的临时对象；ARC 管理对象引用的持有与释放。
  @autoreleasepool {
    // MTLCreateSystemDefaultDevice() 是 C 函数，返回系统默认 GPU 的对象；
    // id<MTLDevice> 表示"任何遵守 MTLDevice 协议的对象"。没有独立显卡的 Mac 上是集显。
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();

    // NSError* 接收库编译、管线创建的错误信息。&err 把指针变量的地址传给 API，
    // 使 API 能在失败时写回错误对象。
    NSError* err = nil;

    // newLibraryWithSource:options:error: 在运行时把一段 MSL 源码编成库。
    // 相邻字符串字面量自动拼接，\n 在生成的源码中换行，结束 #include 指令。
    // 三行依次引入标准库、使用 metal 命名空间、定义 GPU 入口 k。
    // k 的函数体为空，足以为它创建 ps 并查询属性。
    // options 传 nil 表示用默认编译选项；error 传地址以便失败时能打印原因。
    id<MTLLibrary> lib = [dev newLibraryWithSource:@"#include <metal_stdlib>\n"
                                                   "using namespace metal;\n"
                                                   "[[kernel]] void k() {}\n"
                                              options:nil
                                                error:&err];
    if (!lib) {
      // [[err description] UTF8String]：先向 NSError 要一句人类可读的描述（NSString），
      // 再把它转成 C 字符串，因为 printf 的 %s 只认 char*。
      printf("library: %s\n", [[err description] UTF8String]);
      return 1;
    }

    // 嵌套调用先执行内层：[lib newFunctionWithName:@"k"] 取出函数 k；
    // 外层把这个函数交给 dev，创建在该设备上执行 k 所需的计算管线对象 ps。
    // ps 可以在后续多次派发 k 时复用，它也保存下面要查询的函数相关属性。
    id<MTLComputePipelineState> ps =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k"] error:&err];

    // dev.name 是 NSString，同样要转 UTF8String 才能交给 %s。
    printf("device.name                          = %s\n", [[dev name] UTF8String]);

    // limit 存的是设备返回的尺寸上限，width/height/depth 分别约束组尺寸的
    // 三个分量。具体组尺寸由后续派发命令选择，并同时受下面的总线程数限制。
    // 显式转 unsigned long 是因为 %lu 要求无符号长整型，而成员是 NSUInteger。
    MTLSize limit = dev.maxThreadsPerThreadgroup;
    printf("device.maxThreadsPerThreadgroup      = (%lu,%lu,%lu)\n",
           (unsigned long)limit.width, (unsigned long)limit.height, (unsigned long)limit.depth);

    // 执行 k 时单个 threadgroup 的线程总数上限。axpby 的 eval_gpu 也查询
    // 这个属性，但查询对象是 axpby 自己的管线，值须从实际使用的管线取得。
    printf("pipeline.maxTotalThreadsPerThreadgroup = %lu\n",
           (unsigned long)ps.maxTotalThreadsPerThreadgroup);

    // 每个 SIMD 组的线程数，对应 CUDA warp 的线程数这一层概念。
    // threadgroup 可以包含多个 SIMD 组；容量上限和执行分组的区别见文件头。
    printf("pipeline.threadExecutionWidth        = %lu\n",
           (unsigned long)ps.threadExecutionWidth);
  }
  return 0;
}
