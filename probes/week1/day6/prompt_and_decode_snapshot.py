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
print("PROMPT REPR:", repr(prompt), flush=True)
tokens = tokenizer.encode(prompt, add_special_tokens=False)
print("N TOKENS:", len(tokens), flush=True)
print("LAST 12 TOKENS:", tokens[-12:], flush=True)
print("EOS ID:", tokenizer.eos_token_id, flush=True)
print("DECODED TAIL:", repr(tokenizer.decode(tokens[-6:])), flush=True)

with mx.stream(mx.gpu):
    y = mx.array(tokens)[None, :]
    logits = tiny(y)
    last = logits[:, -1, :]
    m = mx.max(last, axis=-1, keepdims=True)
    lp = last - m - mx.log(mx.sum(mx.exp(last - m), axis=-1, keepdims=True))
    ids = mx.argsort(-lp[0])[:8].tolist()
    print("TOP8 ids:", ids, flush=True)
    print("TOP8 text:", [tokenizer.decode([i]) for i in ids], flush=True)
    print("TOP8 logprob:", lp[0][mx.array(ids)].tolist(), flush=True)
    print("ARGMAX:", ids[0], "== eos?", ids[0] == tokenizer.eos_token_id, flush=True)

    # step a few tokens to see raw stream
    seq = list(tokens)
    for step in range(8):
        yy = mx.array(seq)[None, :]
        lg = tiny(yy)[:, -1, :]
        mm = mx.max(lg, axis=-1, keepdims=True)
        lpp = lg - mm - mx.log(mx.sum(mx.exp(lg - mm), axis=-1, keepdims=True))
        nid = int(mx.argmax(lpp, axis=-1).item())
        seq.append(nid)
        print(f"STEP {step} -> id={nid} text={tokenizer.decode([nid])!r}", flush=True)

# detokenizer behaviour
detok = tokenizer.detokenizer
detok.reset()
for i in ids:
    detok.add_token(i)
    print("DETOK", i, "segment=", repr(detok.last_segment), flush=True)
detok.finalize()
print("DETOK final=", repr(detok.last_segment), flush=True)
