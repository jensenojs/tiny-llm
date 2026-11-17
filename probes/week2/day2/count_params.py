"""统计 Qwen3-0.6B 的稠密等价参数个数: 把每个 U32 打包权重按 8 个 4bit 展开."""
import glob
import json
import os
import struct

paths = sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-MLX-4bit/snapshots/*/*.safetensors"
)))
print("权重文件", len(paths))

dense = 0
stored = 0
for p in paths:
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    for name, info in header.items():
        if name == "__metadata__" or not name.endswith("weight"):
            continue
        count = 1
        for d in info["shape"]:
            count *= d
        stored += count
        dense += count * 8 if info["dtype"] == "U32" else count

print(f"打包存储的元素 {stored:,}")
print(f"稠密等价参数   {dense:,} = {dense / 1e6:.1f}M")
print(f"bf16 字节      {dense * 2 / 2**30:.3f} GiB")
print(f"W4 字节        {dense * 0.53125 / 2**30:.3f} GiB")
