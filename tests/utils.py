# =============================================================================
# 测试基建 (test infrastructure) — 学习注释
#
# 本文件是 tests/ 的公共测试工具, 所有 test_week_*_day_*.py 经
# `from .utils import *` 引入。关键约定:
#
# - AVAILABLE_STREAMS / PRECISIONS 与 pytest.mark.parametrize 组合成笛卡尔积,
#   让每个测试函数在 [cpu, gpu] x [f32, f16] 上重复运行, 验证实现不依赖
#   特定设备或精度。一次测试函数 = 4 次实际运行 (Week 1 attention 再乘
#   batch_dimension 4 种 = 16 次)。
# - assert_allclose 是唯一的数值比较器: bfloat16 先升到 float32 再比;
#   f32 用严格容差 (rtol=1e-5, atol=1e-6), f16/bfloat16 用宽松容差
#   (rtol=5e-2) —— float16 尾数 10 bit / bfloat16 尾数 7 bit, 严格比较必然
#   误报。形状不一致或超差时打印逐点 diff 辅助定位。
# - tiny_qwen3_mlx_model / qwen3_*_model_exists: Week 3+ 集成测试用的小模型
#   构造器与真实模型存在性检查 (本地无模型时相关测试自动 skip)。
#
# 注意: 本文件不在 tests_refsol 的复制范围内, 不会被 `pdm run test` 覆盖;
# 而 tests/test_week_*_day_*.py 每次都会被 refsol force 复制, 不要往那里
# 写持久注释。
# =============================================================================

import numpy as np
import mlx.core as mx
import huggingface_hub
from types import SimpleNamespace

AVAILABLE_STREAMS = [mx.cpu, mx.gpu]
AVAILABLE_STREAMS_IDS = ["cpu", "gpu"]
PRECISIONS = [mx.float32, mx.float16]
PRECISION_IDS = ["f32", "f16"]


def tiny_qwen3_mlx_model(
    num_hidden_layers: int = 1, *, head_dim: int = 64
) -> SimpleNamespace:
    """Build a small MLX-shaped Qwen3 model for integration tests."""
    hidden_size = 2 * head_dim

    def quantized_layer(out_dim: int, in_dim: int) -> SimpleNamespace:
        weight = mx.random.normal((out_dim, in_dim)).astype(mx.bfloat16)
        packed, scales, biases = mx.quantize(weight, group_size=128, bits=4)
        return SimpleNamespace(
            weight=packed,
            scales=scales,
            biases=biases,
            group_size=128,
            bits=4,
        )

    args = SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        vocab_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=head_dim,
        intermediate_size=hidden_size,
        rms_norm_eps=1e-5,
        max_position_embeddings=256,
        rope_theta=10000,
        tie_word_embeddings=True,
    )
    layers = []
    for _ in range(num_hidden_layers):
        layers.append(
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    q_proj=quantized_layer(hidden_size, hidden_size),
                    k_proj=quantized_layer(head_dim, hidden_size),
                    v_proj=quantized_layer(head_dim, hidden_size),
                    o_proj=quantized_layer(hidden_size, hidden_size),
                    q_norm=SimpleNamespace(weight=mx.ones((head_dim,), mx.bfloat16)),
                    k_norm=SimpleNamespace(weight=mx.ones((head_dim,), mx.bfloat16)),
                ),
                mlp=SimpleNamespace(
                    gate_proj=quantized_layer(hidden_size, hidden_size),
                    up_proj=quantized_layer(hidden_size, hidden_size),
                    down_proj=quantized_layer(hidden_size, hidden_size),
                ),
                input_layernorm=SimpleNamespace(
                    weight=mx.ones((hidden_size,), mx.bfloat16)
                ),
                post_attention_layernorm=SimpleNamespace(
                    weight=mx.ones((hidden_size,), mx.bfloat16)
                ),
            )
        )
    return SimpleNamespace(
        args=args,
        model=SimpleNamespace(
            embed_tokens=quantized_layer(128, hidden_size),
            layers=layers,
            norm=SimpleNamespace(weight=mx.ones((hidden_size,), mx.bfloat16)),
        ),
    )


def assert_allclose(
    a: mx.array,
    b: mx.array,
    precision: mx.Dtype,
    rtol: float | None = None,
    atol: float | None = None,
    message: str | None = None,
):
    if a.dtype == mx.bfloat16:
        a = a.astype(mx.float32)
    if b.dtype == mx.bfloat16:
        b = b.astype(mx.float32)
    a = np.array(a)
    b = np.array(b)
    if precision == mx.float32:
        rtol = rtol or 1.0e-5
        atol = atol or 1.0e-6
    elif precision == mx.float16:
        rtol = rtol or 5.0e-2
        atol = atol or 1.0e-3
    elif precision == mx.bfloat16:
        rtol = rtol or 5.0e-2
        atol = atol or 1.0e-2
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        diff = np.invert(np.isclose(a, b, rtol=rtol, atol=atol))
        with np.printoptions(precision=3, suppress=True):
            print("a=", a)
            print("b=", b)
            print("diff_a=", a * diff)
            print("diff_b=", b * diff)
            print("diff_a_val=", a[diff])
            print("diff_b_val=", b[diff])
            assert False, f"result mismatch: {message}"


def np_type_to_mx_type(np_type: np.dtype) -> mx.Dtype:
    if np_type == np.float32:
        return mx.float32
    elif np_type == np.float16:
        return mx.float16
    else:
        raise ValueError(f"Unsupported numpy type: {np_type}")


def qwen3_0_6b_model_exists() -> bool:
    try:
        huggingface_hub.snapshot_download(
            "Qwen/Qwen3-0.6B-MLX-4bit", local_files_only=True
        )
        return True
    except Exception as e:
        print(f"Cannot find the Qwen3-0.6B-4bit model: {e}")
        return False


def qwen3_1_7b_model_exists() -> bool:
    try:
        huggingface_hub.snapshot_download(
            "Qwen/Qwen3-1.7B-MLX-4bit", local_files_only=True
        )
        return True
    except Exception as e:
        print(f"Cannot find the Qwen3-1.7B-4bit model: {e}")
        return False


def qwen3_4b_model_exists() -> bool:
    try:
        huggingface_hub.snapshot_download(
            "Qwen/Qwen3-4B-MLX-4bit", local_files_only=True
        )
        return True
    except Exception as e:
        print(f"Cannot find the Qwen3-4B-4bit model: {e}")
        return False
