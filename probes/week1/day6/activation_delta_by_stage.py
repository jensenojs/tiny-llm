# int32 输入和 int64 输入, 逐级比对激活值的 max|Δ|.
# 关键在"零件"这一级怎么跑: 28 层拿到的是字符串 "causal", attention 内部按
# query.dtype 自己造掩码 (attention.py:154-155), 所以输入的整数类型碰不到掩码.
# 顶层 __call__ 不一样, 它在 qwen3_week1.py:379 按 inputs.dtype 造掩码再传下去.
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


def stages(dt):
    y = mx.array(base).astype(dt)[None, :]
    out = {}
    h = tiny.embed_tokens(y)
    out["embed_tokens"] = h
    for layer in tiny.layers:
        h = layer(h, "causal")
    out['28 层 (mask="causal")'] = h
    out["norm + lm_head"] = tiny.lm_head.as_linear(tiny.norm(h))
    out["顶层 __call__(inputs)"] = tiny(y)
    return out


with mx.stream(mx.gpu):
    a, b = stages(mx.int32), stages(mx.int64)

print(f"\n序列长度 {len(base)}", flush=True)
for k in a:
    d = mx.max(mx.abs(a[k].astype(mx.float32) - b[k].astype(mx.float32))).item()
    print(f"{k:26s} max|Δ| = {d:.4g}", flush=True)

# 采样只用最后一个位置, 单独再比一次
last = mx.max(
    mx.abs(
        a["顶层 __call__(inputs)"][0, -1].astype(mx.float32)
        - b["顶层 __call__(inputs)"][0, -1].astype(mx.float32)
    )
).item()
print(f'{"其中 logits[:, -1, :]":26s} max|Δ| = {last:.4g}', flush=True)
