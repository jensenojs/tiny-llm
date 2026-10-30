"""交互装配脚本：把 generate 路径的对象绑进 REPL，供手工实验。

用法 (在仓库根目录; 不需要 -i, 脚本末尾自己停在 >>>):
    .venv/bin/python debug_repl.py
    .venv/bin/python debug_repl.py --model qwen3-0.6b --week 1

落进交互层后可用: mlx_model / tokenizer / model / prompt / messages /
sampler / stream / simple_generate。脚本只做组装，不跑生成——跑不跑、
怎么跑由使用者决定。

各变量由哪一步造出来 (sys.settrace 实跑):
mlx_lm/utils.py:453  load(model_name)  → mlx_model, tokenizer
  ├─ mlx_lm/utils.py:282  load_model  读 config.json 与 model.safetensors, 建 mlx_lm/models/qwen3.py:163 Model
  │  (tie_word_embeddings=True, 所以 mlx_model 没有 lm_head 属性)
  └─ mlx_lm/utils.py:429  load_tokenizer
     └─ mlx_lm/tokenizer_utils.py:578  load  选 BPEStreamingDetokenizer, 包成 TokenizerWrapper
        (模板与词表怎么挂上去见 main.py 与 src/tiny_llm/generate.py 里的树)
mlx_lm/tokenizer_utils.py:335  tokenizer.apply_chat_template  → prompt (str, 180 字符)
src/tiny_llm/sampler.py:5  make_sampler  → sampler (temp=0 时是 argmax)
src/tiny_llm/models.py:8  dispatch_model  → model
  └─ src/tiny_llm/qwen3_week1.py:294  Qwen3ModelWeek1.__init__  从 mlx_model 取权重, 反量化成 bf16
     ├─ src/tiny_llm/qwen3_week1.py:301  Qwen3TransformerBlock × 28 (src/tiny_llm/qwen3_week1.py:177)
     ├─ src/tiny_llm/qwen3_week1.py:326  embed_tokens = Embedding(vocab, E, ...)  (src/tiny_llm/embedding.py:6)
     ├─ src/tiny_llm/qwen3_week1.py:332  norm
     └─ src/tiny_llm/qwen3_week1.py:342  tie=True → lm_head 就是 embed_tokens, 同一张 (vocab, E) 表
model(ids) 走 src/tiny_llm/qwen3_week1.py:352 __call__:
  :368 embed_tokens 查表 → :374-375 28 层 → :377 norm → :378 lm_head.as_linear 出 logits

.venv/bin/python debug_repl.py
ids = mx.array(tokenizer.encode(prompt, add_special_tokens=False))[None, :]
logits = model(ids)
nxt = mx.argmax(logits[0, -1]).item()
print(f"输入 ids 形状      : {ids.shape}  {ids.dtype}")
print(f"logits 形状        : {logits.shape}  (batch, 序列长, 词表)")
print(f"lm_head 与嵌入同表 : {model.lm_head is model.embed_tokens}")
print(f"最后位置 argmax    : {nxt} = {tokenizer.decode([nxt])!r}")
"""

import argparse
import code
import sys

import mlx.core as mx
from mlx_lm import load

from model_names import shortcut_name_to_full_name
from tiny_llm import models, simple_generate
from tiny_llm import sampler as sampler_module

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="qwen3-0.6b")
parser.add_argument("--week", type=int, default=1)
parser.add_argument(
    "--prompt",
    type=str,
    default="Give me a short introduction to large language model.",
)
parser.add_argument("--sampler-temp", type=float, default=0)
parser.add_argument("--sampler-top-p", type=float, default=None)
parser.add_argument("--sampler-top-k", type=int, default=None)
args = parser.parse_args()

model_name = shortcut_name_to_full_name(args.model)
mlx_model, tokenizer = load(model_name)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": args.prompt},
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # 与 main.py 默认一致, 注释里的例子才对得上
)

sampler = sampler_module.make_sampler(
    args.sampler_temp, top_p=args.sampler_top_p, top_k=args.sampler_top_k
)

stream = mx.stream(mx.gpu)
with stream:
    model = models.dispatch_model(model_name, mlx_model, week=args.week)

print(f"[debug_repl] loaded {model_name}, week={args.week}")
print("[debug_repl] bound: mlx_model tokenizer model prompt messages sampler stream simple_generate")

# 不加 -i 直接跑时也停在交互层, 而不是执行完就退出
if not sys.flags.interactive:
    code.interact(local=globals(), banner="")
