from mlx_lm import load
from model_names import shortcut_name_to_full_name

name = shortcut_name_to_full_name("qwen3-0.6b")
_, tokenizer = load(name)

d = tokenizer.detokenizer
d.reset()
for tid in (32, 3460, 4128):
    d.add_token(tid)
    a = d.last_segment
    b = d.last_segment
    print(f"token={tid} first_read={a!r} second_read={b!r}", flush=True)
d.finalize()
print("after finalize first=", repr(d.last_segment), "second=", repr(d.last_segment), flush=True)
print("type:", type(d).__name__, flush=True)
