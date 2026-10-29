import inspect
import os
import sys

_path = os.environ.get("TINY_LLM_TRACE")
_fh = None
_seen = set()


def _write(line):
    global _fh
    print(line, file=sys.stderr)
    if _path:
        if _fh is None:
            _fh = open(_path, "w", buffering=1)
        _fh.write(line + "\n")


def _emit(where, left):
    path = where.f_code.co_filename
    rel = os.path.relpath(path)
    if rel.startswith(".."):
        rel = path
    loc = f"@ {rel}:{where.f_lineno}"
    line = f"{left:<64}  {loc}" if len(left) < 64 else f"{left}  {loc}"
    _write(line)


def stage(no, name, body=""):
    """站级一行: `[no name]  内容  ...  @ 路径:行号`

    编号沿用生成链的顺序 (prompt → encode → model → logits → softmax →
    sampler → new id → detokenizer → screen), 日志从上往下读就是一次前向。
    行尾的 `@ 路径:行号` 给 quickfix 用, 配
    `set errorformat=%m\\ @\\ %f:%l`；每行都带位置，所以列表里没有空条目。
    """
    where = inspect.currentframe().f_back
    _emit(where, f"{f'[{no} {name}]':<16}{body}")


def stage_once(no, name, body=""):
    """同一节里只出一行, 给"每层都会经过"的观测点用.

    rule() 划一次 forward 的分节并清空记录, 所以这里的粒度就是"每次前向一行".
    28 层都造掩码, 但读者只需要看第一次造出来的那张.
    """
    key = (no, name)
    if key in _seen:
        return
    _seen.add(key)
    where = inspect.currentframe().f_back
    _emit(where, f"{f'[{no} {name}]':<16}{body}")


def rule(text):
    """分节横线，同样带位置，标出一次 forward 的起止."""
    _seen.clear()
    where = inspect.currentframe().f_back
    head = f"── {text} "
    _emit(where, head + "─" * max(0, 60 - len(head)))
