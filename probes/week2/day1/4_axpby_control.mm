// A/B/C control: dispatch the existing compiled Axpby kernels directly.
// A = contiguous input / contiguous kernel
// B = contiguous input / general kernel
// C = transposed y view / general kernel
//
// All cases share x, y and out buffers, alpha=beta=1, grid=N*N,
// group=1024 and row-major output. B changes only the kernel and its required
// metadata; C changes only y_strides. No MLX graph/allocation time is measured.
// Each sample encodes one dispatch, commits, waits and reads GPU timestamps.
// Six permutations balance the order. Correctness is checked outside timing.
//
// clang++ -O2 -fobjc-arc -std=c++17 \
//   -framework Metal -framework Foundation \
//   probes/week2/day1/4_axpby_control.mm -o /path/to/control
// MTL_CAPTURE_ENABLED=1 /path/to/control \
//   src/extensions/tiny_llm_ext/tiny_llm_ext.metallib /new/output/dir 60
// Each output directory must be new. Produces samples.json and three captures.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

static void require(bool condition, NSString *message) {
  if (!condition) {
    fprintf(stderr, "%s\n", message.UTF8String);
    exit(1);
  }
}

struct Case {
  const char *name;
  id<MTLComputePipelineState> pipeline;
  bool general;
  bool transpose;
};

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    require(argc == 4, @"usage: control METALLIB NEW_OUTPUT_DIR ITERATIONS");
    char *end;
    long iterations = strtol(argv[3], &end, 10);
    require(*end == '\0' && iterations > 0 && iterations % 6 == 0,
            @"ITERATIONS must be positive and divisible by 6");
    NSString *directory = [NSString stringWithUTF8String:argv[2]];
    NSFileManager *fs = NSFileManager.defaultManager;
    require(![fs fileExistsAtPath:directory], @"output directory already exists");
    NSError *error = nil;
    require([fs createDirectoryAtPath:directory withIntermediateDirectories:YES
                          attributes:nil error:&error], error.description);

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    require(device != nil, @"no Metal device");
    NSURL *libraryURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
    id<MTLLibrary> library = [device newLibraryWithURL:libraryURL error:&error];
    require(library != nil, error.description);
    auto makePipeline = [&](NSString *name) {
      id<MTLFunction> function = [library newFunctionWithName:name];
      require(function != nil, name);
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      require(pipeline != nil, error.description);
      require(pipeline.maxTotalThreadsPerThreadgroup >= 1024,
              @"kernel cannot support the requested group size");
      return pipeline;
    };
    auto contiguous = makePipeline(@"axpby_contiguous_float32");
    auto general = makePipeline(@"axpby_general_float32");
    Case cases[] = {
      {"A_contiguous", contiguous, false, false},
      {"B_general_contiguous", general, true, false},
      {"C_general_transpose", general, true, true},
    };
    constexpr int n = 4096;
    constexpr size_t count = size_t(n) * n;
    constexpr size_t bytes = count * sizeof(float);
    id<MTLBuffer> x = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> y = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    require(x && y && out, @"buffer allocation failed");
    x.label = @"x";
    y.label = @"y";
    out.label = @"out";
    float *xp = (float *)x.contents, *yp = (float *)y.contents, *op = (float *)out.contents;
    // Deterministic, bounded and exactly representable data, separate from offsets.
    for (size_t i = 0; i < count; i++) {
      xp[i] = float(int((i * 17 + 3) % 1021) - 510) / 1024;
      yp[i] = float(int((i * 29 + 7) % 1019) - 509) / 1024;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    require(queue != nil, @"command queue creation failed");
    queue.label = @"Axpby controls";
    NSMutableArray *samples = [NSMutableArray array];
    auto dispatch = [&](int which, bool record, long round, int position) {
      const Case &c = cases[which];
      auto begin = std::chrono::steady_clock::now();
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      cb.label = [NSString stringWithUTF8String:c.name];
      id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
      encoder.label = cb.label;
      [encoder setComputePipelineState:c.pipeline];
      [encoder setBuffer:x offset:0 atIndex:0];
      [encoder setBuffer:y offset:0 atIndex:1];
      [encoder setBuffer:out offset:0 atIndex:2];
      float alpha = 1, beta = 1;
      [encoder setBytes:&alpha length:sizeof(alpha) atIndex:3];
      [encoder setBytes:&beta length:sizeof(beta) atIndex:4];
      if (c.general) {
        int shape[] = {n, n};
        int64_t xs[] = {n, 1};
        int64_t ys[] = {c.transpose ? 1 : n, c.transpose ? n : 1};
        int ndim = 2;
        [encoder setBytes:shape length:sizeof(shape) atIndex:5];
        [encoder setBytes:xs length:sizeof(xs) atIndex:6];
        [encoder setBytes:ys length:sizeof(ys) atIndex:7];
        [encoder setBytes:&ndim length:sizeof(ndim) atIndex:8];
      }
      [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
      [encoder endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      auto finish = std::chrono::steady_clock::now();
      require(cb.status == MTLCommandBufferStatusCompleted, cb.error.description);
      require(cb.GPUEndTime > cb.GPUStartTime && cb.GPUStartTime > 0,
              @"missing GPU timestamps");
      if (record) {
        double wall = std::chrono::duration<double, std::milli>(finish - begin).count();
        [samples addObject:@{
          @"case": cb.label, @"round": @(round), @"position": @(position),
          @"wall_ms": @(wall),
          @"gpu_ms": @((cb.GPUEndTime - cb.GPUStartTime) * 1000),
        }];
      }
    };
    auto verify = [&](int which) {
      for (size_t i = 0; i < count; i++) {
        size_t yi = cases[which].transpose ? (i % n) * n + i / n : i;
        if (op[i] != xp[i] + yp[yi]) {
          require(false, [NSString stringWithFormat:@"incorrect output: %s index %zu", cases[which].name, i]);
        }
      }
    };
    for (int which = 0; which < 3; which++) {
      std::fill(op, op + count, NAN);
      for (int warmup = 0; warmup < 3; warmup++) dispatch(which, false, -1, -1);
      verify(which);
    }
    const int orders[][3] = {{0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0}};
    for (long round = 0; round < iterations; round++) {
      for (int position = 0; position < 3; position++) {
        @autoreleasepool {
          dispatch(orders[round % 6][position], true, round, position);
        }
      }
    }
    NSMutableDictionary *summary = [NSMutableDictionary dictionary];
    for (int which = 0; which < 3; which++) {
      NSString *name = [NSString stringWithUTF8String:cases[which].name];
      NSMutableDictionary *stats = [NSMutableDictionary dictionary];
      for (NSString *metric in @[@"gpu_ms", @"wall_ms"]) {
        std::vector<double> values;
        for (NSDictionary *sample in samples) {
          if ([sample[@"case"] isEqual:name]) values.push_back([sample[metric] doubleValue]);
        }
        std::sort(values.begin(), values.end());
        double sum = 0;
        for (double value : values) sum += value;
        size_t size = values.size();
        stats[metric] = @{
          @"median": @((values[(size-1)/2] + values[size/2]) / 2),
          @"mean": @(sum / size), @"min": @(values.front()), @"max": @(values.back()),
          @"p10": @(values[size/10]), @"p90": @(values[size*9/10]),
        };
      }
      summary[name] = stats;
    }
    // Captures contain exactly one dispatch, after benchmark timing is finished.
    MTLCaptureManager *capture = MTLCaptureManager.sharedCaptureManager;
    require([capture supportsDestination:MTLCaptureDestinationGPUTraceDocument],
            @"GPU capture unavailable; launch with MTL_CAPTURE_ENABLED=1");
    for (int which = 0; which < 3; which++) {
      NSString *name = [NSString stringWithUTF8String:cases[which].name];
      MTLCaptureDescriptor *descriptor = [MTLCaptureDescriptor new];
      descriptor.captureObject = queue;
      descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
      descriptor.outputURL = [NSURL fileURLWithPath:
          [directory stringByAppendingPathComponent:[name stringByAppendingString:@".gputrace"]]];
      require([capture startCaptureWithDescriptor:descriptor error:&error], error.description);
      std::fill(op, op + count, NAN);
      dispatch(which, false, -1, -1);
      [capture stopCapture];
      verify(which);
    }
    NSDictionary *result = @{
      @"device": device.name,
      @"os": NSProcessInfo.processInfo.operatingSystemVersionString,
      @"metallib": libraryURL.path, @"size": @(n), @"dtype": @"float32",
      @"iterations_per_case": @(iterations), @"warmup_per_case": @3,
      @"order": @"six permutations, cyclic", @"grid": @[@(count), @1, @1],
      @"group": @[@1024, @1, @1], @"correct_all_elements": @YES,
      @"scope": @"Direct Metal: wall=encode+commit+wait; GPU=command buffer GPU timestamps",
      @"summary": summary, @"samples": samples,
    };
    NSData *json = [NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:&error];
    require(json != nil, error.description);
    require([json writeToFile:[directory stringByAppendingPathComponent:@"samples.json"]
                     options:NSDataWritingAtomic error:&error], error.description);
    NSData *shortJSON = [NSJSONSerialization dataWithJSONObject:summary options:NSJSONWritingPrettyPrinted error:&error];
    puts([[NSString alloc] initWithData:shortJSON encoding:NSUTF8StringEncoding].UTF8String);
  }
}
