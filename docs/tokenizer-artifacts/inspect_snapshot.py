#!/usr/bin/env python
"""盘点一个 HF 模型快照目录：每个文件是什么，词表有几份，运行时谁在读谁。

用法（在仓库根目录）：
    .venv/bin/python docs/tokenizer-artifacts/inspect_snapshot.py
    .venv/bin/python docs/tokenizer-artifacts/inspect_snapshot.py --snapshot <快照目录>
    .venv/bin/python docs/tokenizer-artifacts/inspect_snapshot.py --model-id Qwen/Qwen3-0.6B-MLX-4bit

脚本只读不写。所有数字都是当场读出来的，不写死。
"""
from __future__ import annotations

import argparse
import json
import pathlib

# 目录里每个文件的角色。不在表里的文件按扩展名猜，猜不出的标 "?"。
ROLES = {
    "config.json": "模型结构（层数、头数、hidden size、vocab_size）",
    "tokenizer_config.json": "分词器行为与 chat_template（一段 Jinja 源码）",
    "tokenizer.json": "fast 路径的词表：vocab + merges + 预分词/解码配置，全打包",
    "vocab.json": "slow 路径的 token→id 字典",
    "merges.txt": "slow 路径的 BPE 合并规则",
    "model.safetensors": "权重",
    "model.safetensors.index.json": "权重分片索引",
    "README.md": "模型卡",
    "LICENSE": "许可",
    ".gitattributes": "git 属性（LFS 等）",
}


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return str(n)


def resolve_snapshot(model_id: str | None, snapshot: str | None) -> pathlib.Path:
    if snapshot:
        return pathlib.Path(snapshot).expanduser().resolve()
    from huggingface_hub import snapshot_download

    return pathlib.Path(
        snapshot_download(model_id, local_files_only=True)
    ).resolve()


def section(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


def show_files(snap: pathlib.Path) -> None:
    section("文件清单")
    files = sorted(p for p in snap.iterdir() if p.is_file())
    width = max(len(p.name) for p in files)
    for p in files:
        print(f"  {human(p.stat().st_size):>9}  {p.name:<{width}}  {ROLES.get(p.name, '?')}")


def show_tokenizer_config(snap: pathlib.Path) -> dict:
    section("tokenizer_config.json")
    cfg = json.loads((snap / "tokenizer_config.json").read_text())
    tpl = cfg.get("chat_template")
    print(f"  tokenizer_class  {cfg.get('tokenizer_class')}")
    print(f"  chat_template    {len(tpl)} 字符，开头 {tpl[:20]!r}")
    print(f"  chat_template_type  {cfg.get('chat_template_type', '(缺省)')}")
    print(f"  add_bos_token    {cfg.get('add_bos_token')}")
    for k in ("bos_token", "eos_token", "pad_token"):
        print(f"  {k:<16} {cfg.get(k)!r}")
    added = cfg.get("added_tokens_decoder") or {}
    print(f"  added_tokens     {len(added)} 个")
    print(f"  model_max_length {cfg.get('model_max_length')}")
    return cfg


def show_vocab_sizes(snap: pathlib.Path, cfg: dict) -> None:
    section("词表有几份，各多少条")
    vocab_json = json.loads((snap / "vocab.json").read_text()) if (snap / "vocab.json").exists() else {}
    merges = (snap / "merges.txt").read_text().splitlines() if (snap / "merges.txt").exists() else []
    merges = [ln for ln in merges if ln and not ln.startswith("#")]
    tj = json.loads((snap / "tokenizer.json").read_text()) if (snap / "tokenizer.json").exists() else {}
    tj_vocab = tj.get("model", {}).get("vocab", {})
    tj_merges = tj.get("model", {}).get("merges", [])

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(snap))
    print(f"  vocab.json        {len(vocab_json):>7} 条（slow 路径）")
    print(f"  merges.txt        {len(merges):>7} 条（slow 路径）")
    print(f"  tokenizer.json    {len(tj_vocab):>7} 条 vocab / {len(tj_merges)} 条 merges（fast 路径）")
    print(f"  len(tokenizer)    {len(tok):>7} 条（含 added tokens）")
    print(f"  运行时类名        {type(tok).__name__}  is_fast={tok.is_fast}")
    print(f"  新增 token 数     {len(tok.get_added_vocab())}")


def show_config(snap: pathlib.Path) -> None:
    section("config.json")
    cfg = json.loads((snap / "config.json").read_text())
    for k in (
        "model_type",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "tie_word_embeddings",
    ):
        if k in cfg:
            print(f"  {k:<24} {cfg[k]}")


def show_checks(snap: pathlib.Path) -> None:
    section("交叉校验")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(snap))
    cfg = json.loads((snap / "config.json").read_text())
    vocab_json = json.loads((snap / "vocab.json").read_text())
    tj = json.loads((snap / "tokenizer.json").read_text())
    tj_vocab = tj.get("model", {}).get("vocab", {})

    n_tok, n_vocab, n_tj = len(tok), len(vocab_json), len(tj_vocab)
    n_cfg = cfg.get("vocab_size")

    def judge(ok: bool) -> str:
        return "一致" if ok else "不一致"

    print(f"  vocab.json == tokenizer.json vocab ?  {judge(n_vocab == n_tj)}  ({n_vocab} vs {n_tj})")
    print(f"  len(tokenizer) == vocab.json       ?  {judge(n_tok == n_vocab)}  ({n_tok} vs {n_vocab})")
    print(f"  len(tokenizer) == config.vocab_size?  {judge(n_tok == n_cfg)}  ({n_tok} vs {n_cfg})")
    if n_cfg and n_tok != n_cfg:
        print()
        print(f"  config.vocab_size={n_cfg} 是 embedding 的行数，len(tokenizer)={n_tok} 是能产出的 id 上界。")
        print("  两者不等说明 embedding 末尾留了 tokenizer 用不到的 padding 行；")
        print("  反向（len > vocab_size）则说明有 token 没有对应的 embedding，模型会越界。")
    print()
    print(f"  合法 id 范围: 0 .. {n_tok - 1}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-0.6B-MLX-4bit")
    ap.add_argument("--snapshot", default=None)
    args = ap.parse_args()

    snap = resolve_snapshot(args.model_id, args.snapshot)
    print(f"快照目录: {snap}")
    show_files(snap)
    cfg = show_tokenizer_config(snap)
    show_vocab_sizes(snap, cfg)
    show_config(snap)
    show_checks(snap)


if __name__ == "__main__":
    main()
