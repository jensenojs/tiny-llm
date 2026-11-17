"""把 decode 一步切成三段, 看 GPU 到底忙多久.

三段:
  build   调用 model(...), 只往计算图里加节点, 不执行
  eval    mx.eval(logits), 提交给 GPU 并同步等待
  sample  logprobs 与 argmax, 再一次 eval 拿到 token id

跑法 (仓库根目录):

    .venv/bin/python probes/week2/day2/step_timeline.py
    .venv/bin/python probes/week2/day2/step_timeline.py --steps 32 --checkpoint kv-cache

`build + sample` 是 Python 侧的时间, 这期间 GPU 队列里没有新 kernel.
真实的交接点上 GPU 是空的, 所以 `eval / (build + eval + sample)` 是这一步
里 GPU 占的比例.
"""

import argparse
import statistics
import sys
from time import perf_counter

import mlx.core as mx
from mlx_lm import load

from model_names import shortcut_name_to_full_name
from tiny_llm import models

PROMPT = "Give me a short introduction to large language model."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3-0.6b")
    p.add_argument("--checkpoint", default="capacity-cache")
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--steps", type=int, default=16, help="测多少步 decode")
    p.add_argument("--warmup", type=int, default=3, help="前多少步不计入汇总")
    p.add_argument("--repl", action="store_true", help="测完停在交互层")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = shortcut_name_to_full_name(args.model)
    mlx_model, tokenizer = load(model_name)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    print(f"prompt {ids.size} 个 token, checkpoint = {args.checkpoint}")

    with mx.stream(mx.gpu):
        model = models.dispatch_model(
            model_name, mlx_model, week=2, checkpoint=args.checkpoint
        )
        capacity = int(ids.size) + args.steps + args.warmup + 8
        cache = model.create_kv_cache(capacity=capacity)

        # prefill: 单独一次, 形状与 decode 不同, 不计入下面的表
        t0 = perf_counter()
        logits = model(ids[None, :], 0, cache, 1)
        t1 = perf_counter()
        mx.eval(logits)
        t2 = perf_counter()
        nxt = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
        mx.eval(nxt)
        t3 = perf_counter()
        offset = ids.size
        print(
            f"prefill {ids.size} 行: build {(t1-t0)*1e3:7.2f} ms  "
            f"eval {(t2-t1)*1e3:7.2f} ms  sample {(t3-t2)*1e3:6.2f} ms"
        )

        rows = []
        total_steps = args.warmup + args.steps
        for step in range(total_steps):
            y = nxt
            if step == 0:
                print(
                    f"decode 第一步入参: y {tuple(y.shape)} -> {tuple(y[None, :].shape)}, "
                    f"offset {offset}, cache.offset {cache[0].offset}"
                )
            t0 = perf_counter()
            logits = model(y[None, :], offset, cache, 1)
            t1 = perf_counter()
            mx.eval(logits)
            t2 = perf_counter()
            logits = logits[:, -1, :]  # (1,1,V) -> (1,V); 不切掉位置维, nxt 会是 (1,1)
            m = mx.max(logits, axis=-1, keepdims=True)
            logprobs = logits - m - mx.log(
                mx.sum(mx.exp(logits - m), axis=-1, keepdims=True)
            )
            nxt = mx.argmax(logprobs, axis=-1).astype(mx.int32)
            mx.eval(nxt)
            t3 = perf_counter()
            offset += 1
            rows.append((step, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3))

    print()
    print(f"{'步':>3}  {'build':>8}  {'eval':>8}  {'sample':>8}  {'合计':>8}  {'GPU 占比':>8}")
    for step, b, e, s in rows:
        mark = " warmup" if step < args.warmup else ""
        total = b + e + s
        print(
            f"{step:>3}  {b:>8.2f}  {e:>8.2f}  {s:>8.2f}  {total:>8.2f}  "
            f"{e / total * 100:>7.1f}%{mark}"
        )

    kept = rows[args.warmup:]
    if kept:
        b = statistics.median(r[1] for r in kept)
        e = statistics.median(r[2] for r in kept)
        s = statistics.median(r[3] for r in kept)
        total = b + e + s
        print()
        print(f"中位数 (去掉前 {args.warmup} 步):")
        print(f"  build  {b:6.2f} ms")
        print(f"  eval   {e:6.2f} ms")
        print(f"  sample {s:6.2f} ms")
        print(f"  合计   {total:6.2f} ms   ->  {1000 / total:.2f} tok/s")
        print(f"  GPU 占 {e / total * 100:.1f}%, Python 侧 {100 - e / total * 100:.1f}%")
        print(f"  Python 侧里 build 占 {(b + s) and b / (b + s) * 100:.0f}%")

    if args.repl:
        import code

        code.interact(local=dict(globals(), **locals()), banner="")


if __name__ == "__main__":
    sys.exit(main())
