from mlx_lm import load
import mlx_lm
import mlx.core as mx
import argparse

import mlx_lm.sample_utils
from model_names import shortcut_name_to_full_name

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="qwen3-0.6b")
parser.add_argument("--draft-model", type=str, default=None)
parser.add_argument(
    "--prompt",
    type=str,
    default="Give me a short introduction to large language model.",
)
parser.add_argument("--solution", type=str, default="tiny_llm")
parser.add_argument("--loader", type=str, default="week1")
parser.add_argument("--device", type=str, default="gpu")
parser.add_argument("--sampler-temp", type=float, default=0)
parser.add_argument("--sampler-top-p", type=float, default=None)
parser.add_argument("--sampler-top-k", type=int, default=None)
parser.add_argument(
    "--max-tokens",
    type=int,
    default=256,
    help="maximum number of newly emitted non-EOS tokens",
)
parser.add_argument("--enable-thinking", action="store_true")
parser.add_argument(
    "--disable-paged-attention",
    action="store_true",
    help="run the Week 3 Day 4 dense-gather compatibility checkpoint",
)
parser.add_argument(
    "--week2-checkpoint",
    choices=(
        "kv-cache",
        "capacity-cache",
        "quantized-matvec",
        "simd-matmul",
        "rmsnorm",
        "rope",
        "swiglu",
        "tiled-prefill",
        "selected",
    ),
    help="run one cumulative Week 2 model checkpoint",
)

args = parser.parse_args()

if args.week2_checkpoint is not None and args.loader != "week2":
    parser.error("--week2-checkpoint requires --loader week2")
if args.week2_checkpoint is not None and args.solution == "mlx":
    parser.error("--week2-checkpoint is not supported with --solution mlx")
if (
    args.solution != "mlx"
    and args.device != "gpu"
    and (
        args.loader == "week3"
        or (
            args.loader == "week2"
            and (args.week2_checkpoint or "selected")
            not in ("kv-cache", "capacity-cache")
        )
    )
):
    parser.error(
        "Week 3 and the Week 2 packed-weight checkpoints require GPU; "
        "the Day 1 Week 2 checkpoints use the readable path"
    )
if args.disable_paged_attention and args.loader != "week3":
    parser.error("--disable-paged-attention requires --loader week3")
if args.disable_paged_attention and args.solution == "mlx":
    parser.error("--disable-paged-attention is not supported with --solution mlx")
if args.max_tokens < 0:
    parser.error("--max-tokens must be non-negative")
if args.draft_model and (
    args.sampler_temp != 0
    or args.sampler_top_p is not None
    or args.sampler_top_k is not None
):
    parser.error("--draft-model supports greedy decoding only; remove sampler options")
if args.draft_model and args.loader == "week1":
    parser.error("--draft-model is not supported with --loader week1")

use_mlx = False
if args.solution == "tiny_llm":
    print("Using your tiny_llm solution")
    from tiny_llm import (
        models,
        simple_generate,
        simple_generate_with_kv_cache,
        speculative_generate,
        sampler,
    )

elif args.solution == "tiny_llm_ref" or args.solution == "ref":
    print("Using tiny_llm_ref solution")
    from tiny_llm_ref import (
        models,
        simple_generate,
        simple_generate_with_kv_cache,
        speculative_generate,
        sampler,
    )

elif args.solution == "mlx":
    use_mlx = True
    from mlx_lm.generate import stream_generate

    print("Using the original mlx model")
else:
    raise ValueError(f"Solution {args.solution} not supported")

if args.max_tokens == 0:
    raise SystemExit(0)

args.model = shortcut_name_to_full_name(args.model)
mlx_model, tokenizer = load(args.model)

if args.draft_model:
    args.draft_model = shortcut_name_to_full_name(args.draft_model)
    draft_mlx_model, draft_tokenizer = load(args.draft_model)
else:
    draft_mlx_model = None
    draft_tokenizer = None

with mx.stream(mx.gpu if args.device == "gpu" else mx.cpu):
    if use_mlx:
        tiny_llm_model = mlx_model
    else:
        if args.loader == "week1":
            print(f"Using week1 loader for {args.model}")
            tiny_llm_model = models.dispatch_model(args.model, mlx_model, week=1)
        elif args.loader == "week2":
            print(
                f"Using week2 loader with thinking={args.enable_thinking} for {args.model}"
            )
            dispatch_kwargs = {}
            if args.week2_checkpoint is not None:
                dispatch_kwargs["checkpoint"] = args.week2_checkpoint
            tiny_llm_model = models.dispatch_model(
                args.model, mlx_model, week=2, **dispatch_kwargs
            )
            if draft_mlx_model is not None:
                print(f"Using draft model {args.draft_model}")
                draft_tiny_llm_model = models.dispatch_model(
                    args.draft_model,
                    draft_mlx_model,
                    week=2,
                    **dispatch_kwargs,
                )
            else:
                draft_tiny_llm_model = None
        elif args.loader == "week3":
            print(
                f"Using week3 loader with paged_attention={not args.disable_paged_attention} "
                f"thinking={args.enable_thinking} for {args.model}"
            )
            tiny_llm_model = models.dispatch_model(
                args.model,
                mlx_model,
                week=3,
                enable_paged_attention=not args.disable_paged_attention,
            )
            if draft_mlx_model is not None:
                print(f"Using draft model {args.draft_model}")
                draft_tiny_llm_model = models.dispatch_model(
                    args.draft_model,
                    draft_mlx_model,
                    week=3,
                    enable_paged_attention=not args.disable_paged_attention,
                )
            else:
                draft_tiny_llm_model = None
        else:
            raise ValueError(f"Loader {args.loader} not supported")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]
    # 这一行的调用链, 按调用顺序. 光标停在路径上, gF / gd 直接跳.
    # tokenize=False: 只渲染, 编码留给下游的 simple_generate.
    #
    # mlx_lm/tokenizer_utils.py:335  TokenizerWrapper.apply_chat_template
    #   ├─ mlx_lm/tokenizer_utils.py:336  enable_thinking 已传入, 不覆盖默认
    #   ├─ mlx_lm/tokenizer_utils.py:339  self._chat_template is None → 跳过 mlx 自带实现
    #   └─ mlx_lm/tokenizer_utils.py:346  转给 HF
    #      └─ transformers/tokenization_utils_base.py:3002  PreTrainedTokenizerBase.apply_chat_template
    #         ├─ transformers/tokenization_utils_base.py:3098  get_chat_template
    #         │  └─ transformers/tokenization_utils_base.py:3281  ← self.chat_template, 来自 JSON 的 chat_template 字段
    #         │     实体: <tok.name_or_path>/tokenizer_config.json 的 "chat_template"
    #         │     tok.name_or_path = ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-MLX-4bit/snapshots/
    #         ├─ transformers/tokenization_utils_base.py:3121  render_jinja_template
    #         │  └─ transformers/utils/chat_template_utils.py:498  render_jinja_template
    #         │     ├─ transformers/utils/chat_template_utils.py:419  _compile_jinja_template
    #         │     │  └─ transformers/utils/chat_template_utils.py:425  _cached_compile_jinja_template
    #         │     │     ├─ jinja2/environment.py:618   Environment._parse → AST
    #         │     │     ├─ jinja2/environment.py:1092  Environment.from_string → Template
    #         │     │     └─ jinja2/environment.py:682   Environment._generate
    #         │     │        └─ jinja2/compiler.py:300   CodeGenerator → Python 源码 → exec
    #         │     └─ transformers/utils/chat_template_utils.py:581  compiled_template.render
    #         │        └─ jinja2/environment.py:1275  Template.render
    #         │           └─ jinja2/environment.py:1291  concat(self.root_render_func(ctx))  ← 真干活
    #         ├─ transformers/tokenization_utils_base.py:3135  if tokenize → False, 跳过
    #         └─ transformers/tokenization_utils_base.py:3176  return rendered_chat
    #
    # root_render_func 是上面 exec 出来的函数, 逐段 yield 字面量. 想看它:
    #   ImmutableSandboxedEnvironment().compile(template, raw=True)
    # <|im_start|>/<|im_end|> 是模板里的字面量, 渲染只负责原样吐出.
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    if not use_mlx:
        sampler = sampler.make_sampler(
            args.sampler_temp, top_p=args.sampler_top_p, top_k=args.sampler_top_k
        )
        if args.loader == "week1":
            simple_generate(
                tiny_llm_model,
                tokenizer,
                prompt,
                sampler=sampler,
                max_tokens=args.max_tokens,
            )
        elif args.loader in ("week2", "week3"):
            if draft_tiny_llm_model is not None:
                speculative_generate(
                    draft_tiny_llm_model,
                    tiny_llm_model,
                    draft_tokenizer,
                    tokenizer,
                    prompt,
                    max_tokens=args.max_tokens,
                )
            else:
                simple_generate_with_kv_cache(
                    tiny_llm_model,
                    tokenizer,
                    prompt,
                    max_tokens=args.max_tokens,
                )
    else:
        sampler = mlx_lm.sample_utils.make_sampler(
            args.sampler_temp, top_p=args.sampler_top_p, top_k=args.sampler_top_k
        )
        for resp in stream_generate(
            tiny_llm_model,
            tokenizer,
            prompt,
            sampler=sampler,
            max_tokens=args.max_tokens,
        ):
            print(resp.text, end="", flush=True)
