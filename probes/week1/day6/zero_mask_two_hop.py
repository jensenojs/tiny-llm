# "最后一行掩码本来就全 0, 为什么全零掩码还会改变最后一个位置?"
# 两版只差掩码内容, dtype 都是 float32, 所以这里测的纯粹是"掩码全零"这件事.
# 每层之后把末尾位置 (S-1) 和前序位置 (0..S-2) 分开比.
import mlx.core as mx
from mlx_lm import load
from model_names import shortcut_name_to_full_name
from tiny_llm import models
from tiny_llm.attention import causal_mask

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


def d(x, y):
    return mx.max(mx.abs(x.astype(mx.float32) - y.astype(mx.float32))).item()


with mx.stream(mx.gpu):
    y = mx.array(base, dtype=mx.int32)[None, :]
    S = y.shape[1]
    good = causal_mask(S, S, mx.float32)
    zero = mx.zeros((S, S), dtype=mx.float32)

    hg = hz = tiny.embed_tokens(y)
    print(f"S={S}")
    print(f"{'layer':>5s}{'last (S-1)':>18s}{'prefix (0..S-2)':>20s}", flush=True)
    one_layer = None
    for n, layer in enumerate(tiny.layers):
        hg, hz = layer(hg, good), layer(hz, zero)
        print(
            f"{n:>5d}{d(hg[0, -1], hz[0, -1]):>18.6e}{d(hg[0, :-1], hz[0, :-1]):>20.6e}",
            flush=True,
        )
        if n == 0:
            # 只有一层时就收口: 末尾位置的 logits 差多少
            lg = tiny.lm_head.as_linear(tiny.norm(hg))
            lz = tiny.lm_head.as_linear(tiny.norm(hz))
            one_layer = d(lg[0, -1], lz[0, -1])

    lg = tiny.lm_head.as_linear(tiny.norm(hg))
    lz = tiny.lm_head.as_linear(tiny.norm(hz))
    print(f"\n只跑第 0 层    logits[:, -1, :] max|Δ| = {one_layer:.6e}", flush=True)
    print(f"全 28 层跑完   logits[:, -1, :] max|Δ| = {d(lg[0, -1], lz[0, -1]):.4g}", flush=True)
