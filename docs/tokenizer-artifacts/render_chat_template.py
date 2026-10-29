#!/usr/bin/env python
"""把 chat_template 从一段转义字符串变成能读的结构，并演示它到 id 数组的整条路。

用法（在仓库根目录）：
    # 1. 只看模板结构：按 Jinja 标签切行，按 if/for 缩进，字符串里的 \\n 原样保留
    .venv/bin/python docs/tokenizer-artifacts/render_chat_template.py

    # 2. 渲染默认 prompt，把特殊 token 和换行标出来
    .venv/bin/python docs/tokenizer-artifacts/render_chat_template.py --render

    # 3. 渲染 + 逐 token 打表：字符串 → id 数组
    .venv/bin/python docs/tokenizer-artifacts/render_chat_template.py --tokens

    # 4. 换 prompt / 换 system / 打开 thinking
    .venv/bin/python docs/tokenizer-artifacts/render_chat_template.py --tokens \
        --prompt "你好" --system "You are terse." --thinking

默认 prompt 与 main.py 的默认值一致，所以 --tokens 的 n 能和站 2 的日志对上。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

LABEL = {"if": "IF", "for": "FOR", "else": "ELSE", "elif": "ELIF"}


def resolve_snapshot(model_id: str | None, snapshot: str | None) -> pathlib.Path:
    if snapshot:
        return pathlib.Path(snapshot).expanduser().resolve()
    from huggingface_hub import snapshot_download

    return pathlib.Path(snapshot_download(model_id, local_files_only=True)).resolve()


def load_template(snap: pathlib.Path) -> str:
    return json.loads((snap / "tokenizer_config.json").read_text())["chat_template"]


def show_structure(src: str) -> None:
    """按 Jinja 标签切行并缩进。

    两类换行必须分开：
      - 模板结构里的换行（源码排版）：丢掉，用缩进表达
      - 字符串字面量里的 \\n（渲染结果的一部分）：原样保留成两个字符
    """
    depth = 0
    for part in re.split(r"(\{%-?.*?-?%\}|\{\{-?.*?-?\}\})", src, flags=re.S):
        if not part:
            continue
        if part.startswith("{%"):
            kw = part[2:-2].strip("- ").strip().split()[0]
            if kw.startswith("end") or kw in ("else", "elif"):
                depth = max(depth - 1, 0)
            print("  " * depth + part)
            if kw in ("if", "for", "block", "else", "elif"):
                depth += 1
        elif part.startswith("{{"):
            print("  " * depth + part)
        elif part.strip():
            print("  " * depth + repr(part))


def annotate(text: str, specials: list[str]) -> str:
    """把特殊 token 用 «» 括起来，换行打成 ⏎。只做显示，不改内容。"""
    pat = "|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True))
    marked = re.sub(f"({pat})", r"«\1»", text) if pat else text
    return marked.replace("\n", "⏎\n")


def build_messages(system: str, prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-0.6B-MLX-4bit")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--render", action="store_true", help="渲染 prompt 并标注")
    ap.add_argument("--tokens", action="store_true", help="渲染后再逐 token 打表")
    ap.add_argument("--thinking", action="store_true", help="enable_thinking=True")
    ap.add_argument("--prompt", default="Give me a short introduction to large language model.")
    ap.add_argument("--system", default="You are a helpful assistant.")
    args = ap.parse_args()

    snap = resolve_snapshot(args.model_id, args.snapshot)
    src = load_template(snap)

    if not (args.render or args.tokens):
        print(f"# chat_template（{len(src)} 字符，来自 {snap.name}/tokenizer_config.json）")
        print()
        show_structure(src)
        return

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(snap))
    messages = build_messages(args.system, args.prompt)
    text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.thinking,
    )

    print(f"# messages（enable_thinking={args.thinking}）")
    for m in messages:
        print(f"  {m['role']:<9} {m['content']!r}")
    print()
    print(f"# 渲染结果（{len(text)} 字符；«» 标出特殊 token，⏎ 标出换行）")
    # 标的是 added_vocab ∪ all_special_tokens。这一模型里两者不重合：
    # all_special_tokens 只有 <|im_end|> 和 <|endoftext|>，而 <|im_start|> / <think>
    # 这些是靠 added_vocab 保证不被 BPE 切碎的，没被标 special。
    marks = list(dict.fromkeys(tok.all_special_tokens + list(tok.get_added_vocab())))
    print(annotate(text, marks))
    print()

    if not args.tokens:
        return

    ids = tok.encode(text, add_special_tokens=False)
    print(f"# encode（add_special_tokens=False）→ n={len(ids)} 个 id")
    print(f"  {'idx':>4}  {'id':>7}  token")
    for i, tid in enumerate(ids):
        piece = tok.convert_ids_to_tokens([tid])[0]
        print(f"  {i:>4}  {tid:>7}  {piece!r}")
    print()
    print(f"  n = {len(ids)}（站 2 的硬判据；和 main.py 默认 prompt 的日志对得上）")


if __name__ == "__main__":
    main()
