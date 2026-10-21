"""交互装配脚本：把 generate 路径的对象绑进 REPL，供手工实验。

用法:
    pdm run python -i debug_repl.py --model qwen3-0.6b --week 1

落进交互层后可用: mlx_model / tokenizer / model / prompt / messages /
sampler / stream / simple_generate。脚本只做组装，不跑生成——跑不跑、
怎么跑由使用者决定。
"""

import argparse

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
)

sampler = sampler_module.make_sampler(
    args.sampler_temp, top_p=args.sampler_top_p, top_k=args.sampler_top_k
)

stream = mx.stream(mx.gpu)
with stream:
    model = models.dispatch_model(model_name, mlx_model, week=args.week)

print(f"[debug_repl] loaded {model_name}, week={args.week}")
print("[debug_repl] bound: mlx_model tokenizer model prompt messages sampler stream simple_generate")
