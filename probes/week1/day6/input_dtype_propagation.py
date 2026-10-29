import mlx.core as mx
from mlx_lm import load
from model_names import shortcut_name_to_full_name
from tiny_llm import models

name = shortcut_name_to_full_name("qwen3-0.6b")
mlx_model, tokenizer = load(name)
tiny = models.dispatch_model(name, mlx_model, week=1)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Give me a short introduction to large language model."},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
base = tokenizer.encode(prompt, add_special_tokens=False) + [32]

with mx.stream(mx.gpu):
    for dt in (mx.int32, mx.int64, mx.uint32):
        y = mx.array(base).astype(dt)[None, :]
        lg = tiny(y)[0, -1]
        print(f"dtype={dt} shape={y.shape} top5={mx.argsort(-lg)[:5].tolist()}",
              f"sum={mx.sum(lg).item():.4f}", flush=True)
    print("mx.array(list).dtype =", mx.array([1, 2, 3]).dtype, flush=True)
    a = mx.array(base)
    n = mx.argmax(mx.zeros((1, 5)), axis=-1)
    print("argmax dtype =", n.dtype, "| concat(int32,uint32) =",
          mx.concatenate([a, n]).dtype, flush=True)
    # embedding alone
    emb = tiny.embed_tokens
    for dt in (mx.int32, mx.int64):
        idx = mx.array(base[:4]).astype(dt)
        print(f"embed dtype={dt} row0[:3]={emb(idx)[0][:3].tolist()}", flush=True)
