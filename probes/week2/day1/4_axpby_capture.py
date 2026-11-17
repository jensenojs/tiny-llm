"""分别捕获一次预热后的 Axpby，供 gpudebug 按 kernel 分析。

仓库根目录运行；两种布局分别使用一个新文件：
  MTL_CAPTURE_ENABLED=1 PYTHONPATH=src/extensions pdm run python \
    probes/week2/day1/4_axpby_capture.py --layout transpose \
    --trace /path/to/new-transpose.gputrace

与 4_axpby_bandwidth.py 保持相同规模、dtype、系数和三次预热。
输入创建与预热在捕获前完成，正确性比较在捕获结束后执行。
trace 保存一次 Axpby 求值；重放的 kernel 成本与普通 bench 墙钟时间
属于不同测量。输出的 expected_kernel 来自源码选路，实际名称由 trace 核对。
"""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("contiguous", "transpose"), required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--size", type=int, default=4096)
    args = parser.parse_args()
    trace = args.trace.resolve()
    metadata = trace.with_suffix(".capture.json")
    if args.size <= 0:
        parser.error("--size must be positive")
    if trace.suffix != ".gputrace":
        parser.error("--trace must end in .gputrace")
    if not trace.parent.is_dir():
        parser.error("--trace parent directory must exist")
    for path in (trace, metadata):
        if path.exists():
            parser.error(f"refusing to overwrite {path}")

    import mlx.core as mx
    import tiny_llm_ext as ext

    mx.random.seed(0)
    x = mx.random.normal((args.size, args.size)).astype(mx.float32)
    y = mx.random.normal((args.size, args.size)).astype(mx.float32)
    yy = y if args.layout == "contiguous" else y.T
    mx.eval(x, y, yy)
    for _ in range(3):
        mx.eval(ext.axpby(x, yy, 1.0, 1.0))

    # MLX 0.32.4 会在上次提交后预先创建下一只 command buffer。
    # 捕获开始后先 synchronize，提交捕获前那只，并在捕获内创建新 buffer。
    # 本机对照验证：省略这步时 trace 只有同步命令；加入后有一个计算 dispatch。
    # mx.eval 等到结果可用，随后 stop_capture 结束这一段 GPU 命令记录。
    mx.metal.start_capture(str(trace))
    try:
        mx.synchronize()
        out = ext.axpby(x, yy, 1.0, 1.0)
        mx.eval(out)
    finally:
        mx.metal.stop_capture()

    # 参考计算发生在捕获外，避免将 multiply/add 或比较 kernel 混入 trace。
    correct = bool(mx.all(out == x + yy).item())
    assert correct, "captured Axpby differs from x + y"
    root = Path(__file__).resolve().parents[3]
    sources = [
        Path(__file__).resolve(),
        root / "src/extensions/src/axpby.cpp",
        root / "src/extensions/src/axpby.metal",
    ]
    report = {
        "layout": args.layout,
        "shape": list(x.shape),
        "dtype": "float32",
        "alpha": 1.0,
        "beta": 1.0,
        "seed": 0,
        "warmup_calls": 3,
        "captured_calls": 1,
        "capture_boundary": "start_capture -> synchronize -> axpby -> eval -> stop_capture",
        "expected_kernel": (
            "axpby_contiguous_float32"
            if args.layout == "contiguous"
            else "axpby_general_float32"
        ),
        "trace": str(trace),
        "correct": correct,
        "mlx_version": version("mlx"),
        "macos": platform.mac_ver()[0],
        "device": mx.device_info(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    }
    metadata.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
