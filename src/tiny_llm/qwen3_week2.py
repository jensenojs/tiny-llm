from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

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
        pass

    def __call__(
        self,
        x: mx.array,
        offsets: int | list[int] | mx.array,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


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
        pass

    def __call__(self, x: mx.array) -> mx.array:
        pass


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
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


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
        pass

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
        pass
