import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear

# ── 记号表 (全文件通用, 各图中的缩写都查这里) ─────────────────────
#   B:  batch, 批大小
#   L:  序列长度 (query 侧的 token 数)
#   S:  key/value 侧的序列长度 (prefill 时 S=L; decode 时 L=1, S=已生成历史)
#   E:  hidden_size, 模型统一的表示宽度 (= 嵌入维度, 层间恒定)
#   H_q: num_heads, query 头数
#   H:  num_kv_heads, KV 头数 (GQA: H ≤ H_q, 每个 KV 头被 H_q//H 个 Q 头共享)
#   D:  head_dim, 每个头的维数 (独立配置, 不由 E/H_q 推出)
#   I:  intermediate_size, FFN 中间宽度 (Qwen3 约为 4E)


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
        # ── 鸟瞰: 三条支路, ①-④ 都是备战(重排/加工), 只有 ⑤ 真正跨位置计算 ──
        #
        #   x (B,L,E)
        #   ├──► wq ─► 拆 H_q 头─► RMSNorm(q_norm) ─► RoPE ─┐
        #   ├──► wk ─► 拆 H  头 ─► RMSNorm(k_norm) ─► RoPE ─┼─► ⑤ GQA 打分加权 ─► 合头 ─► wo ─► out
        #   └──► wv ─► 拆 H  头 ────────────────────────────┘
        #   (Q/K 加工进"内容+位置"; V 是载荷, 只换轴不加工)
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
        # 打分与 softmax 提升到 float32: bfloat16 尾数太短,
        # 点积和指数运算的误差会累积 (oracle 同样在进注意力前转 float32)
        out = scaled_dot_product_attention_grouped(
            q.astype(mx.float32),
            k.astype(mx.float32),
            v.astype(mx.float32),
            mask=mask,
        ).astype(x.dtype)

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
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        # FFN 逐位置独立: (B, L, E) 的每个位置单独走一遍下面的流程, 位置之间不通信
        # (通信是 attention 的职责, FFN 只做逐位置加工).
        #
        # big idea: 升维 -> 非线性 -> 降维.
        #   1. W_gate / W_up 把每个位置从 E 维升到 I≈4E 维: 在更宽的空间里给非线性腾出施展余地;
        #   2. SiLU(W_gate x) 造出 I 维门控, 逐通道决定 W_up x 的对应分量放行多少 ——
        #      两条支路逐元素相乘, 这个乘法是整个 FFN 非线性的主要来源;
        #   3. W_down 把 I 维压回 E 维: 残差连接 x = x + FFN(x) 要求进出口同宽.
        #
        #        x (B,L,E) —— 每个位置独立
        #        │
        #        ├──► W_gate (E→I) ──► SiLU ──────┐
        #        │                                ⊙ ──► W_down (I→E) ──► out (B,L,E)
        #        └──► W_up   (E→I) ───────────────┘
        #
        # 形状流 (单个位置):
        #   x:    (E)
        #   gate = SiLU(x @ W_gate.T)   (E) -> (I)   # 非线性在这里进入
        #   up   = x @ W_up.T           (E) -> (I)   # 线性支路
        #   h    = gate * up            (I) 逐元素乘, 形状不变
        #   out  = h @ W_down.T         (I) -> (E)
        gate = silu(linear(x, self.w_gate))
        up = linear(x, self.w_up)
        return linear(gate * up, self.w_down)


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
        )
        self.mlp = Qwen3MLP(
            dim=hidden_size,
            hidden_dim=intermediate_size,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # ── Block 全景 (Pre-LN: 归一化在旁路, 残差加法是主干) ─────────
        #
        #            x (B,L,E)
        #            │
        #            ├──────────────────► 残差1 ──┐
        #            ▼                            │
        #      input_layernorm                    │
        #            ▼                            │
        #      multiHeadAttention                 │
        #            ▼                            │
        #            + ◄──────────────────────────┘    x = x + attn(ln₁(x))
        #            │
        #            ├──────────────────► 残差2 ──┐
        #            ▼                            │
        #      post_attention_layernorm           │
        #            ▼                            │
        #           mlp                           │
        #            ▼                            │
        #            + ◄──────────────────────────┘    x = x + mlp(ln₂(x))
        #            │
        #            ▼
        #        out (B,L,E)
        x_input_layernorm = self.input_layernorm(x)
        x_multiHeadAttention = self.multiHeadAttention(x_input_layernorm, mask=mask)
        x_delta1 = x_multiHeadAttention + x

        x_post_attention_layernorm = self.post_attention_layernorm(x_delta1)
        x_mlp = self.mlp(x_post_attention_layernorm)
        x_delta2 = x_mlp + x_delta1
        return x_delta2


class Qwen3ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
