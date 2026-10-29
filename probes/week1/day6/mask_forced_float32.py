import mlx.core as mx
from mlx_lm import load
from model_names import shortcut_name_to_full_name
import tiny_llm.qwen3_week1 as q1
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
toks = tokenizer

orig = q1.causal_mask


def patched(L, S, dtype):
    return orig(L, S, mx.float32)


with mx.stream(mx.gpu):
    for label in ("original", "mask_forced_f32"):
        if label == "mask_forced_f32":
            q1.causal_mask = patched
        for dt in (mx.int32, mx.int64):
            y = mx.array(base).astype(dt)[None, :]
            lg = tiny(y)[0, -1]
            top = mx.argsort(-lg)[:5].tolist()
            print(f"{label:16s} input_dtype={dt} top5={top} "
                  f"text={[toks.decode([i]) for i in top]}", flush=True)
