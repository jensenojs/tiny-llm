import mlx.core as mx
from tiny_llm.attention import causal_mask

for dt in (mx.int32, mx.int64, mx.uint32):
    m = causal_mask(4, 4, dt)
    print(f"dtype={dt} -> {m.dtype}", flush=True)
    print(m.tolist(), flush=True)

# is the corruption from the mask alone?
scores = mx.array([[1.0, 2.0, 3.0, 4.0]] * 4, dtype=mx.float32)
for dt in (mx.int32, mx.int64):
    m = causal_mask(4, 4, dt)
    out = scores + m
    print(f"dt={dt} scores+mask dtype={out.dtype} last_row={out[-1].tolist()}", flush=True)
