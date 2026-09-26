import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
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

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # x: (B, L, E) —— 每个 token 一条 E 维向量
        B, L, _ = x.shape

        # ── ① 投影 + 拆头 ──────────────────────────────────────────
        # linear: (B,L,E) @ (E, H_q*D) -> (B,L,H_q*D)       ← 运算(改数值)
        #   输出还是"所有头拼成的一条长向量", H_q 和 D 还粘在一起
        # reshape: (B,L,H_q*D) -> (B,L,H_q,D)              ← 重排(不改数值)
        #   按行优先把长向量切成 H_q 段、每段 D
        #   切完 H_q(头数) 和 D(头内维) 才变成两个独立的轴
        # K/V 同理, 但宽度是 H*D (H = num_kv_heads), 不是 H_q*D
        q = linear(x, self.wq).reshape(B, L, self.num_heads, self.head_dim)
        k = linear(x, self.wk).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = linear(x, self.wv).reshape(B, L, self.num_kv_heads, self.head_dim)

        # ── ② 只对 Q/K 归一化 ──────────────────────────────────────
        # mx.fast.rms_norm 沿"最后一维"归一化 —— 现在最后一维是 D
        #   即: 每个 (b,l,h) 位置上的 D 维向量各自算均方根、各自缩放
        #   q_norm 的长度 = D, 是每头共用的一组缩放权重
        # 为什么必须等 reshape 之后:
        #   若在拆头前做, 最后一维是 H_q*D, 会把所有头当成一整条向量归一化
        q = mx.fast.rms_norm(q, self.q_norm, eps=self.rms_norm_eps)
        k = mx.fast.rms_norm(k, self.k_norm, eps=self.rms_norm_eps)

        # ── ③ 只对 Q/K 旋转 (RoPE) ─────────────────────────────────
        # self.rope 期望 (N,L,H,D), 其中:
        #   第二维 L 是"位置维": 按位置取 cos/sin 表的第 0..L-1 行
        #   最后一维 D 是"被旋转的维": 把 D 按对分组, 每对转一个角度
        # 形状不变: (B,L,H_q,D) -> (B,L,H_q,D)  ← 旋转是逐对坐标的替换
        # 为什么必须现在做:
        #   若挪到 transpose 之后 (B,H_q,L,D), 第二维变成 H_q,
        #   rope 会把"头数"当成序列长度去取表
        q = self.rope(q)
        k = self.rope(k)

        # ── ④ 换轴 ────────────────────────────────────────────────
        # (B,L,H,D) -> (B,H,L,D)                            ← 重排
        #   L 和 H 交换. 目的: attention 要算每个头内部的 (L,D)@(D,S),
        #   头必须提到前面才能各头独立并行; L 留在倒数第二维,
        #   才能和 K 的 S 维构成 (L,D)@(D,S)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # ── ⑤ 分组注意力 ──────────────────────────────────────────
        # (B,H_q,L,D) -> (B,H_q,L,D)                        ← 运算
        # 内部: 每个 KV 头被 n_repeats = H_q // H 个 Q 头共享
        out = scaled_dot_product_attention_grouped(q, k, v, mask=mask)

        # ── ⑥ 换轴回 + 合头 ────────────────────────────────────────
        # (B,H_q,L,D) -> (B,L,H_q,D) -> (B,L,H_q*D)         ← 重排
        # 先换回来, 再把 H_q 和 D 拼回一条长向量
        # (输出投影 w_o 的输入是"每 token 一条 H_q*D")
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim)

        # ── ⑦ 输出投影 ────────────────────────────────────────────
        # (B,L,H_q*D) @ (H_q*D, E) -> (B,L,E)               ← 运算
        return linear(out, self.wo)


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
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
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen3ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
