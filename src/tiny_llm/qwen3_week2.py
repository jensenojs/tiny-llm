from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .attention import scaled_dot_product_attention_grouped
from .positional_encoding import RoPE
from .basics import linear, silu
from .layer_norm import RMSNorm

import mlx.core as mx

from .embedding import Embedding  # noqa: F401 - learner checkpoint dependency
from .kv_cache import TinyKvCache, TinyKvFullCache
from .quantize import QuantizedWeights, dequantize_linear  # noqa: F401
from .week2_kernels import (
    FastRMSNorm,  # noqa: F401 - learner checkpoint dependency
    FastRoPE,  # noqa: F401 - learner checkpoint dependency
    scaled_dot_product_attention,  # noqa: F401 - learner checkpoint dependency
    swiglu,  # noqa: F401 - learner checkpoint dependency
)


@dataclass(frozen=True)
class Week2CheckpointFeatures:
    bounded_kv_capacity: bool = False
    quantized_weights: bool = False
    fast_rms_norm: bool = False
    fast_rope: bool = False
    fast_swiglu: bool = False
    simdgroup_matmul: bool = False
    tiled_prefill_attention: bool = False


WEEK2_CHECKPOINT_FEATURES = MappingProxyType(
    {
        "kv-cache": Week2CheckpointFeatures(),
        "capacity-cache": Week2CheckpointFeatures(bounded_kv_capacity=True),
        "quantized-matvec": Week2CheckpointFeatures(
            bounded_kv_capacity=True, quantized_weights=True
        ),
        "simd-matmul": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
        ),
        "rmsnorm": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
        ),
        "rope": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
            fast_rope=True,
        ),
        "swiglu": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
        ),
        "tiled-prefill": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
            tiled_prefill_attention=True,
        ),
        "selected": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
            tiled_prefill_attention=True,
        ),
    }
)
WEEK2_CHECKPOINTS = tuple(WEEK2_CHECKPOINT_FEATURES)


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array | QuantizedWeights,
        wk: mx.array | QuantizedWeights,
        wv: mx.array | QuantizedWeights,
        wo: mx.array | QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
        use_fast_rms_norm: bool = True,
        use_fast_rope: bool = True,
        use_tiled_prefill_attention: bool = False,
    ):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.theta = theta
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope = RoPE(
            dims=head_dim, seq_len=max_seq_len, base=theta, traditional=False
        )
        self.use_fast_rms_norm = use_fast_rms_norm
        self.use_fast_rope = use_fast_rope
        self.use_tiled_prefill_attention = use_tiled_prefill_attention

    def __call__(
        self,
        x: mx.array,
        offsets: int | list[int] | mx.array,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q = linear(x, self.wq).reshape(B, L, self.num_heads, self.head_dim)
        k = linear(x, self.wk).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = linear(x, self.wv).reshape(B, L, self.num_kv_heads, self.head_dim)

        q = mx.fast.rms_norm(q, self.q_norm, eps=self.rms_norm_eps)
        k = mx.fast.rms_norm(k, self.k_norm, eps=self.rms_norm_eps)

        if not isinstance(offsets, int):
            raise NotImplementedError("per-request offsets are a Week 3 concern")
        pos = slice(offsets, offsets + L)

        q = self.rope(q, pos)
        k = self.rope(k, pos)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        cache_k, cache_v, seq_len, mask = cache.update_and_fetch(k, v, mask=mask)

        out = scaled_dot_product_attention_grouped(
            q.astype(mx.float32),
            cache_k.astype(mx.float32),
            cache_v.astype(mx.float32),
            mask=mask,
        ).astype(x.dtype)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim)
        return linear(out, self.wo)


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array | QuantizedWeights,
        w_up: mx.array | QuantizedWeights,
        w_down: mx.array | QuantizedWeights,
        use_fast_swiglu: bool = True,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.use_fast_swiglu = use_fast_swiglu

    def __call__(self, x: mx.array) -> mx.array:
        gate = silu(linear(x, self.w_gate))
        up = linear(x, self.w_up)
        if self.use_fast_swiglu:
            h = swiglu(gate, up)
        else:
            h = gate * up
        return linear(h, self.w_down)


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array | QuantizedWeights,
        wk: mx.array | QuantizedWeights,
        wv: mx.array | QuantizedWeights,
        wo: mx.array | QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array | QuantizedWeights,
        w_up: mx.array | QuantizedWeights,
        w_down: mx.array | QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_fast_rms_norm: bool = True,
        use_fast_rope: bool = True,
        use_fast_swiglu: bool = True,
        use_tiled_prefill_attention: bool = False,
        use_mlx_quantized_linear: bool = False,
    ):
        self.use_fast_rms_norm = use_fast_rms_norm
        self.use_fast_rope = use_fast_rope
        self.use_tiled_prefill_attention = use_tiled_prefill_attention
        self.use_mlx_quantized_linear = use_mlx_quantized_linear
        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm)
        self.post_attention_layernorm = RMSNorm(hidden_size, w_post_attention_layernorm)
        self.multiHeadAttention = Qwen3MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            q_norm=q_norm,
            k_norm=k_norm,
            max_seq_len=max_seq_len,
            theta=theta,
            rms_norm_eps=rms_norm_eps,
            use_fast_rms_norm=use_fast_rms_norm,
            use_fast_rope=use_fast_rope,
            use_tiled_prefill_attention=use_tiled_prefill_attention,
        )
        self.mlp = Qwen3MLP(
            dim=hidden_size,
            hidden_dim=intermediate_size,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
            use_fast_swiglu=use_fast_swiglu,
        )

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        x_input_layernorm = self.input_layernorm(x)
        x_multiHeadAttention = self.multiHeadAttention(
            x_input_layernorm, offset, cache, mask=mask
        )
        x_delta1 = x_multiHeadAttention + x

        x_post_attention_layernorm = self.post_attention_layernorm(x_delta1)
        x_mlp = self.mlp(x_post_attention_layernorm)
        return x_mlp + x_delta1


class Qwen3ModelWeek2:
    def __init__(
        self,
        mlx_model: Any,
        checkpoint: str = "selected",
        use_mlx_quantized_linear: bool = False,
        use_bounded_kv_capacity: bool | None = None,
        use_register_cached_rms_norm: bool | None = None,
        use_tiled_prefill_attention: bool | None = None,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.use_mlx_quantized_linear = use_mlx_quantized_linear
        self.use_register_cached_rms_norm = use_register_cached_rms_norm

        args = mlx_model.args
        features = WEEK2_CHECKPOINT_FEATURES[checkpoint]
        if use_bounded_kv_capacity is None:
            use_bounded_kv_capacity = features.bounded_kv_capacity
        if use_tiled_prefill_attention is None:
            use_tiled_prefill_attention = features.tiled_prefill_attention
        self.use_tiled_prefill_attention = use_tiled_prefill_attention
        self.use_bounded_kv_capacity = use_bounded_kv_capacity

        layers = []
        for layer in mlx_model.model.layers:
            sa = layer.self_attn
            mlp = layer.mlp
            layers.append(
                Qwen3TransformerBlock(
                    num_attention_heads=args.num_attention_heads,
                    num_kv_heads=args.num_key_value_heads,
                    hidden_size=args.hidden_size,
                    head_dim=args.head_dim,
                    intermediate_size=args.intermediate_size,
                    rms_norm_eps=args.rms_norm_eps,
                    wq=dequantize_linear(sa.q_proj),
                    wk=dequantize_linear(sa.k_proj),
                    wv=dequantize_linear(sa.v_proj),
                    wo=dequantize_linear(sa.o_proj),
                    q_norm=sa.q_norm.weight,
                    k_norm=sa.k_norm.weight,
                    w_gate=dequantize_linear(mlp.gate_proj),
                    w_up=dequantize_linear(mlp.up_proj),
                    w_down=dequantize_linear(mlp.down_proj),
                    w_input_layernorm=layer.input_layernorm.weight,
                    w_post_attention_layernorm=layer.post_attention_layernorm.weight,
                    max_seq_len=getattr(args, "max_position_embeddings", 32768),
                    theta=getattr(args, "rope_theta", 1000000),
                    use_fast_rms_norm=features.fast_rms_norm,
                    use_fast_rope=features.fast_rope,
                    use_fast_swiglu=features.fast_swiglu,
                    use_tiled_prefill_attention=use_tiled_prefill_attention,
                    use_mlx_quantized_linear=use_mlx_quantized_linear,
                )
            )
        self.layers = layers

        self.embed_tokens = Embedding(
            args.vocab_size,
            args.hidden_size,
            dequantize_linear(mlx_model.model.embed_tokens),
        )
        self.norm = RMSNorm(args.hidden_size, mlx_model.model.norm.weight)
        if not args.tie_word_embeddings:
            self.lm_head = Embedding(
                args.vocab_size,
                args.hidden_size,
                dequantize_linear(mlx_model.lm_head),
            )
        else:
            self.lm_head = self.embed_tokens

    def create_kv_cache(self, capacity: int | None = None) -> list[TinyKvCache]:
        return [
            TinyKvFullCache(capacity=capacity) for _ in range(self.num_hidden_layers)
        ]

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvCache],
        logits_to_keep: int | None = None,
    ) -> mx.array:
        # ── 为什么位置要从外面传 ────────────────────────────────────
        #
        # 一个整数在这里承担了两个语义:
        #
        #   position   这是序列里第几个 token, 决定 RoPE 的旋转角度
        #   slot       这个 K/V 写到数组的第几个格子
        #
        # dense 缓存里 position == slot 恒成立, 因为 dense 的定义就是
        # 逻辑位置等于物理下标. 所以两件事一直共用一个数. 单请求连续
        # 追加时, 它恰好等于 cache[0].offset, 这个参数看起来是冗余的.
        #
        # 位置的 owner 是请求, 不属于 cache:
        #
        #   模型 (权重)   无状态, 每步读同一份
        #   推理引擎      持有请求队列, 决定谁先跑
        #   请求          持有自己的 position 和已生成的 token
        #   cache         持有 K/V, 以及一份 slot (已存长度)
        #
        # cache 之所以要自己推断 slot, 是因为 update_and_fetch 的入参
        # 只有 (k, v), 没有地址. 它只能假设「接在已有的后面」. 这个
        # 假设在四种情形下破裂:
        #   1. 批处理: 各请求长度不同, 这一步的位置分别是 5、2、0.
        #   2. 前缀缓存: 共享 K/V, 各请求的逻辑位置不同.
        #   3. 投机解码: 写入 3 个草稿, 验证后 rewind 掉 2 个.
        #   4. 跳号验证: 只补第 7 个位置, 缓存里已有 5 个.
        #
        # 位置也无法由 cache 派生: RoPE 要在写入之前用 position 旋转
        # 新的 q、k, 那一刻 cache 里还是旧值. 让 cache 自己算, 就得把
        # RoPE 搬进 cache, 存储层从此依赖位置编码的频率与 base.
        #
        # 代价是这个参数可以传错, 所以入口先做一致性检查. 不一致时
        # RoPE 按一个 position 旋转、cache 按一个 slot 写入, 输出错误
        # 但不抛异常, 只是生成质量下降. 28 层必须同步, 查第 0 层等价
        # 于查全部, 失败得也比第 17 层早.
        #
        # 三种写法把位置的副本依次减少:
        #   A. 拆成 prefill/decode, 位置从 cache 派生; 只覆盖连续追加.
        #   B. 传 positions (B, L) 代替 offset; cache 仍持有私有 slot,
        #      两者仍可能不一致.
        #   C. 把 slots 也传给 cache, 它不再持有私有 slot; dense 要求
        #      slots 连续, 有空洞就要分页.
        # 完整讨论见 docs/kv-cache-position-ownership.md.
        if cache[0].offset != offset:
            raise ValueError(
                f"cache offset {cache[0].offset} disagrees with position {offset}"
            )

        h = self.embed_tokens(inputs)
        for layer, layer_cache in zip(self.layers, cache):
            h = layer(h, offset, layer_cache, mask="causal")

        h = self.norm(h)
        # 只需要最后一个位置的 logits: 预测下一个 token 用的是最后一个
        # 位置的输出, 前面的位置不参与.
        #
        # 为什么能切. 每个位置的 h 在 28 层里有两次消费:
        #   1. 变成这个位置的 k、v, 被后面的位置读到 (attention 内部);
        #   2. 变成这个位置的输出, 作为下一层的输入.
        # 第 28 层结束时这两次都用完了, 只剩 lm_head 还没用. 而 lm_head
        # 只关心最后一个位置. 所以前面的 h 必须算 (它们的 k、v 被后面
        # 的位置读), 但它们的 logits 不用算.
        #
        # 错误写法 (先算完再切):
        #   logits = self.lm_head.as_linear(h)        # (B, L, V)
        #   logits = logits[:, -logits_to_keep:, :]
        # lm_head 是 (L, E) @ (E, V), 上面给全部 L 个位置各乘了一次,
        # 只有最后几行被留下, 前面 L - logits_to_keep 行的矩阵乘白做.
        #
        # 正确写法 (先切再乘):
        #   h = h[:, -logits_to_keep:, :]             # (B, k, E)
        #   return self.lm_head.as_linear(h)          # (B, k, V)
        # 乘加次数从 L·E·V 降到 k·E·V, k = logits_to_keep.
        #
        # 代一组数. E 与 V 取自 self.lm_head 的形状 (Qwen3-0.6B 是
        # 1024 与 151936); L 取 prefill 的 token 数, 下面假设 33;
        # k = logits_to_keep = 1:
        #   错误   33 × 1024 × 151936 ≈ 5.13e9 次乘加
        #   正确    1 × 1024 × 151936 ≈ 1.56e8 次乘加
        #   比值 33, 等于 L / k.
        if logits_to_keep is not None:
            h = h[:, -logits_to_keep:, :]
        return self.lm_head.as_linear(h)
