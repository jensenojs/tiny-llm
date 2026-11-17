"""观察 Qwen3-0.6B-MLX-4bit 的 W4 打包结构.

这个脚本不实现解包, 只把打包后的三张表摊开, 让每一步中间结果可见.
跑法:

    .venv/bin/python probes/week2/day2/unpack_lab.py

要拨动的地方都标了 `改这里`. 每个改动都会改变一路输出, 对照着看.
"""

import argparse
import sys

import mlx.core as mx
from mlx_lm import load

MODEL = "Qwen/Qwen3-0.6B-MLX-4bit"

# 默认看第 0 行. 换 --row 看别的 token: 不同行的 scale 量级不同,
# 也有的行全部 scale 为正.
ROW = 0

# checkpoint 的固定参数. 改了它, 拆出的码与真实数据就对不上,
# 只有换到别的 checkpoint 时才动.
BITS = 4
PER_WORD = 32 // BITS


def show_tables(e):
    V, E = 151936, 1024
    print("=== 三张表 ===")
    print(f"weight  {tuple(e.weight.shape)}  {e.weight.dtype}   "
          f"每行 {e.weight.shape[1]} 个 word, 每个 word {PER_WORD} 个码")
    print(f"scales  {tuple(e.scales.shape)}  {e.scales.dtype}   "
          f"每行 {e.scales.shape[1]} 个组")
    print(f"biases  {tuple(e.biases.shape)}  {e.biases.dtype}")
    print(f"group_size {e.group_size}, bits {e.bits}")
    print(f"一行还原出 {e.weight.shape[1] * PER_WORD} 个权重, "
          f"恰好等于 hidden_size {E}")
    print(f"一组 {e.group_size} 个权重, 占 {e.group_size // PER_WORD} 个 word")
    print()


def show_one_word(e, row):
    word = e.weight[row, 0].item()
    print(f"=== 第 {row} 行第 0 个 word ===")
    print(f"0x{word:08X}   {word:032b}")
    print("从低 4 位往上拆:")
    codes = []
    for i in range(PER_WORD):
        code = (word >> (BITS * i)) & ((1 << BITS) - 1)
        codes.append(code)
        print(f"  >> {BITS * i:2d} & 0xF -> {code:2d}")
    print(f"码序列: {codes}")
    print()

    # 改这里: 把 BITS * i 换成 BITS * (PER_WORD - 1 - i), 看整组是否逆序
    return codes


def show_reconstruction(e, row, codes):
    s = e.scales[row, 0].item()
    b = e.biases[row, 0].item()
    print("=== 用第 0 组的 scale 与 bias 还原这 8 个码 ===")
    print(f"scale = {s!r}   bias = {b!r}")
    manual = [c * s + b for c in codes]
    for i, (c, w) in enumerate(zip(codes, manual)):
        print(f"  码[{i}] = {c:2d}   {c} * {s} + {b} = {w: .6f}")
    print()

    # mx.dequantize 要求覆盖一个完整组: group_size / PER_WORD 个 word
    words = e.group_size // PER_WORD
    truth = mx.dequantize(
        e.weight[row:row + 1, :words],
        e.scales[row:row + 1, 0:1],
        e.biases[row:row + 1, 0:1],
        e.group_size, e.bits,
    )
    print("=== 对照 mx.dequantize ===")
    print("前 8 个:", [round(float(x), 6) for x in truth[0, :PER_WORD]])
    print("手工的:", [round(x, 6) for x in manual])
    print("最大差:", max(abs(float(t) - m) for t, m in zip(truth[0, :PER_WORD], manual)))
    print()


def show_group_boundary(e, row):
    words = e.group_size // PER_WORD
    print(f"=== 组边界: 前 {words} 个 word 共用 scales[{row}, 0] ===")
    print("scales 前 4 组:", [round(float(x), 6) for x in e.scales[row, :4]])
    print("biases 前 4 组:", [round(float(x), 6) for x in e.biases[row, :4]])
    print(f"第 {words} 个 word 起换下一组的参数; "
          f"同组内 {words} 个 word 共用一组 scale/bias")
    negative = [i for i, x in enumerate(e.scales[row].tolist()) if x < 0]
    print(f"这一行里 scale 为负的组号: {negative}")
    if negative:
        g = negative[0]
        s = e.scales[row, g].item()
        b = e.biases[row, g].item()
        print(f"  组 {g}: scale = {s!r}, bias = {b!r}")
        print(f"  于是码 0 -> {b:.6f}, 码 15 -> {15 * s + b:.6f}")
        print("  码 0 落在高的一端, 与正 scale 的组相反")
    print()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--row", type=int, default=ROW, help="看词表里哪一行")
    p.add_argument("--repl", action="store_true", help="看完停在交互层")
    return p.parse_args()


def main():
    args = parse_args()
    model, _ = load(args.model)
    e = model.model.embed_tokens
    show_tables(e)
    codes = show_one_word(e, args.row)
    show_reconstruction(e, args.row, codes)
    show_group_boundary(e, args.row)

    q = model.model.layers[0].self_attn.q_proj
    print("=== 同一格式用在投影上 ===")
    print(f"q_proj.weight {tuple(q.weight.shape)} {q.weight.dtype}, "
          f"scales {tuple(q.scales.shape)}")
    print(f"2048 个输出通道, 每个通道 1024 个输入, "
          f"打包成 {q.weight.shape[1]} 个 word")

    if args.repl:
        import code

        code.interact(local=dict(globals(), **locals()), banner="")


if __name__ == "__main__":
    sys.exit(main())
