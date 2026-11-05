import mlx.core as mx
from .basics import linear, silu
from .trace import stage
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
#   I:  intermediate_size, FFN 中间宽度 (Qwen3 是 3E)


class Qwen3MultiHeadAttention:
    """本模块的渲染与调试入口, 对应框架那一侧 layers[0].self_attn.

    .venv/bin/python debug_repl.py
    attn = mlx_model.model.layers[0].self_attn
    print(attn)
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        p = getattr(attn, name)
        print(f"{name:7} 打包 {str(tuple(p.weight.shape)):13} scales {str(tuple(p.scales.shape)):12} 还原 {p.weight.shape[0]}×{p.weight.shape[1]*8}")
    print(f"{'q_norm':7} γ {tuple(attn.q_norm.weight.shape)}  只在打分支路")
    print(f"{'rope':7} {attn.rope}")

        渲染:
        Attention(
          (q_proj): QuantizedLinear(input_dims=1024, output_dims=2048, bias=False, group_size=128, bits=4, mode=affine)
          (k_proj): QuantizedLinear(input_dims=1024, output_dims=1024, bias=False, group_size=128, bits=4, mode=affine)
          (v_proj): QuantizedLinear(input_dims=1024, output_dims=1024, bias=False, group_size=128, bits=4, mode=affine)
          (o_proj): QuantizedLinear(input_dims=2048, output_dims=1024, bias=False, group_size=128, bits=4, mode=affine)
          (q_norm): RMSNorm(128, eps=1e-06)
          (k_norm): RMSNorm(128, eps=1e-06)
          (rope): RoPE(128, traditional=False)
        )
        q_proj  打包 (2048, 128)   scales (2048, 8)    还原 2048×1024
        k_proj  打包 (1024, 128)   scales (1024, 8)    还原 1024×1024
        v_proj  打包 (1024, 128)   scales (1024, 8)    还原 1024×1024
        o_proj  打包 (1024, 256)   scales (1024, 16)   还原 1024×2048
        q_norm  γ (128,)  只在打分支路
        rope    RoPE(128, traditional=False)

        判据, 每条对上 __init__ 的一个参数:
        q_proj 还原成 (2048,1024) = Hq·D, 正是 o_proj 的输入宽; k/v_proj 是 q/o 的一半宽,
        GQA 写在形状里 (16 个 q 头共用 8 组 kv);
        o_proj 打包 (1024,256): 输入 2048 = 8 列一组 × 256 组, scales 16 列 = 2048/128;
        q_norm/k_norm 长 D=128, eps 要与 rms_norm_eps 传成一致 (见 Qwen3ModelWeek1 对账段);
        rope 只吃 D=128, 角度表高 max_seq_len=32768, base=theta=1000000.

        四行形状的顺序读法:
        1024 -q_proj→ 2048 -切成 16 段×128→ 打分加权 -拼回 2048 -o_proj→ 1024
        2048 不等于 E, 所以头不是在 1024 上均分 (那样每头 64), 而是在投影后的
        2048 上切; q_norm 与 rope 的形状都是 128, 正是切完每段的宽度 —— 归一化
        与旋转都在每段内部独立做. 第 h 个 Q 头配第 h//2 组 K/V.
        k_proj 输出 1024 = 8×128, 宽度是 q_proj 的一半, 所以只切 8 段.

        维度约束, 三条各管一处:
        q 的第 h 个头与 k 的第 g 组必须等长 —— 点积逐位相乘求和, 两边都是 D=128;
        v 的 S 行之间必须等宽 —— 加权求和要把它们叠起来, 宽度记 D_v;
        v 与 q/k 之间没有约束 —— 打分产出 (L,S) 的标量表, 权重个数由 S 定,
        与 D_v 无关.
        本项目三支共用同一个 D (attention.py 的
        scaled_dot_product_attention_grouped 里 D 只从 query 取), 所以 D_v≠D 会报
        reshape 错. 那是实现假设: 数学上 v 的宽度可任取, 唯一要闭合的是出口
        H·D_v = o_proj 的输入宽; K 侧另有一把尺子 KV·D = k_proj 的输出宽,
        两者互不影响. 2048 与 1024 的差是头数 16 对 8, 每头仍是 128.

        缓存 2·KV·D·2 B = 2×8×128×2 = 4 KiB 每 token 每层; KV 若也取 16 组
        则是 8 KiB. 省下的一半是相对同 D 的 MHA, 与经典 64 维头的 MHA 相比
        字节数相同 —— 省的是组数, 不是每组的宽度.

        DeepSeek-V3 的 MLA 动的是另一处: 一次压到低秩, cache 只存
        kv_lora_rank 512 + qk_rope_head_dim 64 = 576 个数, 与头数无关.
        MHA 的 2·H·D 随头数增长, 压缩之后这条依赖消失. K 与 V 在解压时才出现:
        embed_q 把 512 还原成每头的 128 维 (K 的内容部分), unembed_out 还原成
        每头的 v_head_dim (V). 那 64 维位置键单独一条不走压缩 —— 它要过 RoPE,
        而 RoPE 作用在头维度上, 压缩表示没有头维度. decode 时连解压都省:
        把 K 侧的 W_k 用结合律推到 Q 侧、V 侧推到输出侧, 直接拿 512 维 latent
        打分. 两条路径见 mlx_lm/models/deepseek_v3.py:118-169.

        __call__ 内的鸟瞰图与 ①-⑦ 分步见方法内注释. 接缝:
        输入 (B,L,E) 输出 (B,L,E); 输出交给 Qwen3TransformerBlock 的残差加法
        x + attn(input_layernorm(x)). 这里出错, 表现是每一层之后整句都偏.
    """

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
    """本模块的渲染与调试入口, 对应框架那一侧 layers[0].mlp.
        直观解释大语言模型如何储存事实 | 【深度学习第7章】
        Transformer Feed-Forward Layers Are Key-Value Memories

    .venv/bin/python debug_repl.py
    mlp = mlx_model.model.layers[0].mlp
    print(mlp)
    for name in ["gate_proj", "up_proj", "down_proj"]:
        p = getattr(mlp, name)
        print(f"{name:10} 打包 {str(tuple(p.weight.shape)):13} scales {str(tuple(p.scales.shape)):12} 还原 {p.weight.shape[0]}×{p.weight.shape[1]*8}")

        渲染:
        MLP(
          (gate_proj): QuantizedLinear(input_dims=1024, output_dims=3072, bias=False, group_size=128, bits=4, mode=affine)
          (down_proj): QuantizedLinear(input_dims=3072, output_dims=1024, bias=False, group_size=128, bits=4, mode=affine)
          (up_proj): QuantizedLinear(input_dims=1024, output_dims=3072, bias=False, group_size=128, bits=4, mode=affine)
        )
        gate_proj  打包 (3072, 128)   scales (3072, 8)    还原 3072×1024
        up_proj    打包 (3072, 128)   scales (3072, 8)    还原 3072×1024
        down_proj  打包 (1024, 384)   scales (1024, 24)   还原 1024×3072

        判据: gate/up 还原成 (3072,1024) = I×E, down 还原成 (1024,3072) = E×I;
        down 打包 (1024,384), 3072/8=384 组. 参数量 3×E×I, 每层 9.44M, 全模型 44% 都在这里:
        I 是"这层花在逐位置加工上的预算", 中段每个坐标对应一个模式的判断与写入;
        模式越多, 单层能记住的事实越多. (Qwen3 取 I=3E; SwiGLU 有三张矩阵, 传统 MLP 两张,
        Shazeer 2020 建议宽度取 2/3×4E ≈ 2.67E, LLaMA 取 8/3E, 都是配平参数量的选择.)

        ── 简要版 ────────────────────────────────────────────────────
        MLP 做的是有条件的查表, 每个位置单独查一次:
          升维: x 与 W_up 的 3072 行逐一点积, 得到 3072 个条件各自的匹配得分.
          开关: 非线性把得分低的条件压到 0 (SiLU 是压到接近 0), 这些条件不写入.
          降维: 通过的条件按得分, 把 W_down 里各自那一列写入向量加权相加, 作为增量加回 x.

        为什么要升维, 要从残差流怎样存放特征说起. 1024 维空间里两两严格正交的方向最多
        1024 个; 只要求两两近似正交 (夹角接近 90 度), 能放下的方向数随维度指数增长.
        模型利用这一点, 用近似正交的方向在 1024 维残差流里存放远多于 1024 个特征, 这种
        存法称为叠加 (superposition; Elhage et al. 2022, Toy Models of Superposition,
        arXiv 2209.10652). 代价是特征之间不完全正交: 用一个特征的方向去做点积, 读数里
        混着其他特征的小量串扰. 由此推出三件事:
          轴没有含义. 特征是方向, 不和坐标轴对齐, 一根轴同时参与很多特征; 在原坐标上
            压掉一根轴, 会同时削弱所有用到它的特征.
          开关必须放在升维之后. 只有沿特征方向点积得到的读数才对应一个条件; 非线性在
            这些读数上逐个压放, 同时把低于阈值的串扰当作 0 截掉.
          I 是预算, 不是特征总数. 3072 个条件同样装不下全部特征, 中间层的神经元也是叠加
            的: 同一个神经元会对几个不相关的特征都有反应 (多义性, 上述论文的研究对象).
            I 越大, 一层里能干净区分的条件越多.

        ── 为什么需要这个模块 ────────────────────────────────────────
        整个模型里, 每个位置只携带一条 E = 1024 维的向量: embed_tokens 查表得到它,
        28 个 block 依次读它、把各自的增量加回去, lm_head 最后用同一张表对它打分.
        层与层之间只传这一条向量, 宽度锁死在 1024: 残差加法要求进出同宽, 出口复用
        嵌入表也要求 1024. 任何子层的产出, 最终都只能写成这 1024 个数上的增量.

        往这条向量上写东西的子层有两个, 信息来源不同. attention 只能搬运上下文里已有
        的信息, 它的输出由各位置的 v 加权平均得到. "Michael Jordan"后面该接篮球, 这条知识不在
        输入文本里, 在参数里; 把参数里的知识写进来的是 MLP. 写入是有条件的: 先判断
        当前内容满足哪些条件, 再写入这些条件对应的信息.

        有条件就需要非线性: 满足就写, 不满足就不写. 这里的非线性 (SiLU, 传统 MLP 用
        ReLU) 逐分量作用, 每次只压或放一个坐标. 逐分量开关有意义的前提, 是每个坐标
        恰好对应一个条件. 1024 维向量的坐标不满足这个前提: 特征以叠加的方式存放 (见简要版),
        轴本身没有含义, 一个概念摊在很多根轴上, 一根轴也参与很多概念; 在这里关掉一根轴, 会同时削弱所有用到
        它的概念. 三步代码由此推出:

          升维 up = x @ W_up.T: W_up 的 I 行各是一个模式的检测方向, 点积把内容换到一套
            新坐标里, 第 i 个坐标就是模式 i 的得分. 要识别的模式远多于 1024 个,
            每个模式占一个独立坐标, 所以 I > E (Qwen3 取 3072).
          开关 h = gate * up: 在新坐标里逐分量压放, 等于逐个模式判断在场与否.
          降维 out = h @ W_down.T: 3072 维的 h 出不了这一层. W_down 的第 i 列是模式 i
            在场时要写入的 1024 维信息, h 按各自强度把这些列加总成一份增量.

        中间的开关不能省. 去掉它, 升维再降维合成一个 1024×1024 矩阵, 两张表可以并成
        一张, I 开多大都没有意义. 宽度锁定逼出降维, 逐分量开关逼出升维, 模式数多于 E
        决定了升到多宽.

        下面先讲矩阵乘法的两种读法 (升维和降维各用一种), 再用一个玩具把三步走一遍.

        ── 线性代数: 一次乘法的两种组合 ────
        矩阵乘法 C = A x B，两种“组合”到底在组合谁

        取 2×2 的例子：

        A = [ a  b ]        B = [ e  f ]
            [ c  d ]            [ g  h ]

        C = A x B = [ ae+bg   af+bh ]
                    [ ce+dg   cf+dh ]

        按行看：C 的每一行，是 B 的行的加权和

        C 第1行 = a × [e f] + b × [g h]
                = [ae+bg, af+bh]

        C 第2行 = c × [e f] + d × [g h]
                = [ce+dg, cf+dh]

        权重来自 A 的对应行：A 第 1 行是 [a, b]，就拿 a 乘 B 第 1 行、b 乘 B 第 2 行，加起来。
        按列看：C 的每一列，是 A 的列的加权和

        C 第1列 = e × [a] + g × [b] = [ae+bg]
                      [c]       [d]   [ce+dg]

        C 第2列 = f × [a] + h × [b] = [af+bh]
                      [c]       [d]   [cf+dh]

        权重来自 B 的对应列：B 第 1 列是 [e, g]，就拿 e 乘 A 第 1 列、g 乘 A 第 2 列，加起来。

        小结 : C = A @ B 时：
        - 按行组合：系数来自 A 的行，被组合的是 B 的行。
        - 按列组合：系数来自 B 的列，被组合的是 A 的列。

        乘法的定义只有一个双重求和, 这两种读法只是记账的先后顺序不同

        用升维的形状: x 是 1×E 的行向量, U 是 E×I 的矩阵, y = x @ U 是 1×I.

        取 E=2, I=3:
          x = [2, 1]       U = [ 1  0  2 ]        y = x @ U = [2, 3, 5]
                               [ 0  3  1 ]

        按列组合 (打分): U 的每一列与 x 做一次点积, 点积的得数就是 y 的一个分量;
        U 有三列, y 就有三个分量. 每个分量只看自己那一列, 互相不知道别人.

              列0      列1      列2       (U 的三列, 每列 2 个分量)
             [ 1]     [ 0]     [ 2]
        x ·  [ 0]     [ 3]     [ 1]       x = [2, 1]
              │        │        │
              ▼        ▼        ▼
            y[0]=2   y[1]=3   y[2]=5      一次点积产出一个分量

        按行组合 (混合): x 的每个分量给自己那行发放大倍率 (x[0]=2 把行0 放大 2 倍),
        放大后的各行逐列相加, 加完就是 y.

          x[0]=2 ──×──► 行0 [1, 0, 2] ──► [2, 0, 4] ──┐
                                                      +──► y = [2, 3, 5]
          x[1]=1 ──×──► 行1 [0, 3, 1] ──► [0, 3, 1] ──┘

        ── 代入玩具: 6 根字母轴, 10 个模式 ────────────────────────────
        残差流的六根轴记作字母 A B C D E F. 玩具为了能读, 让 A B C 只在名字特征出现时
        活跃 (A: Michael, B: Jordan, C: Phelps), D E F 只在属性特征出现时活跃
        (D: 篮球, E: 游泳, F: 国家). 真实模型里一根轴不对应单一特征, 见本节末尾.

        输入 x 是 1×6 的行向量, 是 "Michael Jordan" 里 Jordan 这个位置走到这一层的残差流:
                  A  B  C  D  E  F
             x = [1, 1, 0, 0, 0, 0]
        前面的层已经把 Michael 的特征搬到了这个位置 (那是 attention 的工作), 所以 A、B 都是 1.
        D E F 全是 0: 这个位置只知道名字, 还不知道这个人是做什么的.

        这一层要补什么, 取决于字母的组合. A = 1 在 Michael Jordan 和 Michael Phelps 里都成立,
        只看 A 分不出该补篮球还是游泳; B = 1 在 Michael Jordan 和国家约旦里都成立, 只看 B
        也分不出. 能决定的是组合, 例如"A 和 B 都在, C 不在". 这样的组合在 x 里没有自己的
        坐标, 在 x 上逐分量开关只能整根关掉 A 或 B; 所以先升维, 给每个组合一个坐标.
        6 个字母能组成的条件远多于 6 个: 玩具取 10 个模式, 真实模型 1024 根轴对 3072 个模式.

        十个模式: p0 Michael Jordan; p1 Michael Phelps; p2 约旦 (有 Jordan, 没有 Michael);
        p3 别的 Michael (Jordan、Phelps 都不在); p4 Phelps (没有 Michael); p5 姓 Jordan 的人
        (只看 Jordan); p6 篮球语境; p7 游泳语境; p8 国家语境; p9 奥运 (游泳与国家都在).

        升维: 代码里存的是 w_up, 10×6, 每一行是一个模式的检测方向,
        1 表示"这个字母应该在", -1 表示"这个字母不应该在". 下图框出 p0 这一行:

                       A   B   C   D   E   F
                ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓
             p0 ┃    1   1  -1   0   0   0 ┃
                ┗━━━━━━━━━━━━━━━━━━━━━━━━━━┛
             p1 [    1  -1   1   0   0   0 ]
             p2 [   -1   1   0   0   0   0 ]
             p3 [    1  -1  -1   0   0   0 ]
              ⋮ [    ⋮   ⋮   ⋮   ⋮   ⋮   ⋮ ]
             p9 [    0   0   0   0   1   1 ]

              矩阵的总行数 可以看作被提问的数量

        linear(x, w_up) 算的是 x @ w_up.T. 转置后 U = w_up.T 是 6×10, w_up 的 p0 行
        变成 U 的 p0 列:

                                           ┏━━━━┓
                                           ┃ p0 ┃  p1  p2  p3  p4  p5  p6  p7  p8  p9
                                       A [ ┃  1 ┃   1  -1   1  -1   0   0   0   0   0 ]
                                       B [ ┃  1 ┃  -1   1  -1   0   1   0   0   0   0 ]
            x = [ 1  1  0  0  0  0 ] @ C [ ┃ -1 ┃   1   0  -1   1   0   0   0   0   0 ]
                  A  B  C  D  E  F     D [ ┃  0 ┃   0   0   0   0   0   1   0   0   0 ]
                                       E [ ┃  0 ┃   0   0   0   0   0   0   1   0   1 ]
                                       F [ ┃  0 ┃   0   0   0   0   0   0   0   1   1 ]
                                           ┗━┳━━┛
                                             ▼
                                 up = [       2     0   0   0  -1   1   0   0   0   0 ]

        升维用按列组合: 系数是框出的那一列 (p0 的检测方向), 被组合的是 x 的六个分量:
          up[p0] =  1·x[A] + 1·x[B] - 1·x[C] + 0·x[D] + 0·x[E] + 0·x[F] =  2
        其余九列各算一次, 得到 up 的其余九个数. 每个数只由一个模式的检测方向算出,
        所以 up[p0] = 2 读作"这里是 Michael Jordan". 再看一列: up[p1] = 1 - 1 + 0 = 0,
        p1 要求 B 不在, 而 B 在, 它的 -1 抵消了 A 的 +1.

        升维的成品 up 是 1×10, 下标与模式一一对应:
             up = [2, 0, 0, 0, -1, 1, 0, 0, 0, 0]
        up[i] 的含义: 模式 i 的条件在当前内容里满足到什么程度. x 回答"哪些字母在",
        up 回答"哪些组合成立": 同一份内容, 坐标从 6 个字母换成了 10 个模式.
        读这一行: p0 Michael Jordan 得 2, 条件完全满足; p1 Phelps 和 p2 约旦都是 0,
        因为各自要求不在的字母 (B、A) 在场, 被 -1 抵消; p5 姓 Jordan 得 1, 部分满足;
        p4 得 -1, 条件相反. 这一步只做了判断, 残差流还没有任何变化.

        开关: 玩具用传统 MLP 的 h = relu(up), 负数和 0 都变成 0. h 仍是 1×10:
             h  = [2, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        h[i] 的含义: 模式 i 获准写入的强度. 每个坐标只属于一个模式, 压掉 h[4] 只关掉 p4,
        不碰别的模式. 这个逐模式的开关, 只有在升维之后的坐标上才做得到.
        (Qwen3 用 h = SiLU(gate) * up, 开关值来自另一张检测方向表 w_gate; 作用见下文.)

        为什么还要降维: 一是宽度. h 是 10 维的, 层与层之间只传 6 维的残差流 (真实模型 1024 维),
        h 出不了这一层. 二是内容. h 只说明"哪些组合成立、各有多强", 没说明"该补什么";
        补什么是这一层存下的知识, 放在 W_down 里. 降维把两者接起来: 拿 h 的 10 个强度,
        去取同一下标的 10 个写入向量.

        降维: 代码里存的是 w_down, 6×10, 每一列是一个模式的写入向量, 写的是属性字母
        D E F, 不再写名字字母. 下图框出 h 不为 0 的 p0、p5 两列 (p6~p9 读的是属性字母,
        留给后面的层, 写入向量略):

               ┏━━━━━┓                        ┏━━━━━┓
               ┃  p0 ┃   p1    p2    p3    p4 ┃  p5 ┃   p6    p7    p8    p9
            A [┃   0 ┃    0     0     0     0 ┃   0 ┃    ⋮     ⋮     ⋮     ⋮ ]
            B [┃   0 ┃    0     0     0     0 ┃   0 ┃    ⋮     ⋮     ⋮     ⋮ ]
            C [┃   0 ┃    0     0     0     0 ┃   0 ┃    ⋮     ⋮     ⋮     ⋮ ]
            D [┃   1 ┃    0     0     0     0 ┃ 0.3 ┃    ⋮     ⋮     ⋮     ⋮ ]
            E [┃   0 ┃    1     0     0   0.5 ┃   0 ┃    ⋮     ⋮     ⋮     ⋮ ]
            F [┃   0 ┃    0     1     0     0 ┃   0 ┃    ⋮     ⋮     ⋮     ⋮ ]
               ┗━━━━━┛                        ┗━━━━━┛

              矩阵的总列数 可以看作被回答的数量, 和 W_up 的总行数相等: 一个检测方向配一个写入向量.

        linear(h, w_down) 算的是 h @ w_down.T. 转置后 D = w_down.T 是 10×6, w_down 的
        p0、p5 两列变成 D 的 p0、p5 两行:

                                                               A    B    C    D    E    F
                                                         ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                                                      p0 ┃     0    0    0    1    0    0 ┃ ──► 2 × p0 的写入向量
                                                         ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                                      p1 [     0    0    0    0    1    0 ]
            h = [ 2  0  0  0  0  1  0  0  0  0 ] @    p2 [     0    0    0    0    0    1 ]
                  p0 p1 p2 p3 p4 p5 p6 p7 p8 p9       p3 [     0    0    0    0    0    0 ]
                                                      p4 [     0    0    0    0  0.5    0 ]
                                                         ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                                                      p5 ┃     0    0    0  0.3    0    0 ┃ ──► 1 × p5 的写入向量
                                                         ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                                   p6~p9 [     ⋮    ⋮    ⋮    ⋮    ⋮    ⋮ ]

        降维用按行组合: 系数是 h 的十个分量, 被组合的是 D 的十行 (十个写入向量).
        h 里只有 p0、p5 不为 0, 所以只有框出的两行参与:
          out = 2 × [ 0   0   0   1     0   0 ]    p0 Michael Jordan 写入的内容
              + 1 × [ 0   0   0   0.3   0   0 ]    p5 姓 Jordan 写入的内容
              = [ 0   0   0   2.3   0   0 ]
        out 的每一项都能指认是哪个模式写的. 换句话说, 十个写入向量是基, h 是 out 在这组
        基上的坐标.

             out = [0, 0, 0, 2.3, 0, 0]

        out 的含义: 这一层根据"哪些组合成立"给残差流算出的增量. 它回到了字母坐标上, 和 x
        是同一套坐标, 所以能直接加回 x. 读这一行: D (篮球) 是 0 → 2.3, 这个值在 x 里不存在,
        是这一层从参数里取出来的事实, 主要由 p0 写入, p5 补了 0.3. E (游泳)、F (国家)
        保持 0: Phelps、约旦的写入向量都存在 W_down 里, 但它们的开关是 0, 没有写.
        残差加回 x + out = [1, 1, 0, 2.3, 0, 0]: 下一层再读这个位置, 名字和"篮球"都在了.

        检测方向与写入向量是两套组合: 检测方向读名字字母 A B C, 写入向量写属性字母 D E F. 这就是 key 与 value
        的分工: key 决定在什么条件下被取用, value 决定取用之后交出什么, 两者描述的不是同一件事.

        中间的开关为什么不能省, 在玩具里也能直接看到. 去掉 relu, h 就等于 up, p4 的 -1
        会照样乘它的写入向量, 往游泳 E 写入 -0.5: 一个不成立的条件也改动了残差流.
        有了开关, 不成立的条件一律写 0.

        w_gate 不在这套概念里: 它给每个模式再配一个检测方向, 让开关值与得分来自两张
        表, 这是 SwiGLU 的实现选择; 玩具用的传统 h = relu(up) 与它概念相同.

        玩具与真实模型的差别: 玩具让一根轴对应一个特征, 检测方向只用 1 / -1 / 0, 为的是能读.
        真实模型里没有人规定轴的含义, 字母表与全部检测方向、写入向量都是训练出来的; 一个特征摊在很多
        根轴上, 一根轴参与很多特征, 每个检测方向是 1024 个实数. "读一组特征的组合, 写另一组特征"
        这个分工不变.
        玩具里每根轴只对应一个特征, 所以看不出叠加: 真实模型里一根轴同时承载多个特征,
        检测方向的点积会混进其他特征的小量串扰, 开关把低于阈值的串扰压成 0.
        __call__ 的数据流 (升维 → SiLU 门控 → 降维) 见方法内注释. 接缝:
        输出交给 Qwen3TransformerBlock 的第二个残差加法 x + mlp(post_attention_layernorm(x)).
        逐位置独立, 位置之间不通信; 出错时每个位置错法相同, 这是和 attention 区分的观察点.
        MLP 与 attention 同为"打分 → 非线性 → 混合", 两者在键值来源、是否通信、权重是否
        竞争上的差别, 见 Qwen3TransformerBlock.__call__ 注释的同名一节.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        # 三张表: w_gate / w_up 同形状 (hidden_dim, dim), 是传统 MLP 那张 W1 的两份
        # (一份管开关, 一份管内容); w_down 是 W2, 形状反过来 (dim, hidden_dim).
        # 全部 bias=False, 对应 3B1B 那张"矩阵×向量"图里没有 Bias 向量的版本.
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        # 逐位置独立: (B, L, E) 的每个位置单独走下面四行, 位置之间不通信
        # (通信归 attention). 为什么是 升维 → 开关 → 降维, 见类 docstring
        # 「为什么需要这个模块」; 这里只对上数据流与形状.
        #
        #        x (B,L,E)
        #        │
        #        ├──► W_gate (E→I) ──► SiLU ── gate ──┐
        #        │                                    ⊙ ── h ──► W_down (I→E) ──► out (B,L,E)
        #        └──► W_up   (E→I) ──────────  up  ───┘
        #        (out 交给 TransformerBlock 做残差加法 x = x + mlp(ln₂(x)); 加法不归本模块)
        #
        # 形状流 (单个位置, E=1024, I=3072; batch 维照抄不参与, 以 (1,1024) 为例):
        #   x        (1, 1024)      当前位置的内容
        #   w_up     (3072, 1024)   3072 行 = 3072 个模式的检测方向 (总行数 = 被提问的数量)
        #   up       (1, 3072)      x @ w_up.T, 按列组合: 每个模式一个得分
        #   gate     (1, 3072)      silu(x @ w_gate.T), 每个模式一个开关值
        #   h        (1, 3072)      gate * up, 逐元素: 开关作用在每个模式自己的坐标上
        #   w_down   (1024, 3072)   3072 列 = 3072 个模式的写入向量 (总列数 = 被回答的数量)
        #   out      (1, 1024)      h @ w_down.T, 按行组合: 在场模式的写入向量按 h 加总
        #
        # 玩具 (docstring 里的 6 根字母轴, 10 个模式) 对上这四行:
        #   up  = [2, 0, 0, 0, -1, 1, 0, 0, 0, 0]   按列组合: p0 Michael Jordan 条件完全满足
        #   h   = [2, 0, 0, 0,  0, 1, 0, 0, 0, 0]   开关: 不成立的条件写 0
        #   out = 2 × p0 写入向量 + 1 × p5 写入向量 = [0, 0, 0, 2.3, 0, 0]
        #         按行组合: 只有开关放行的两行参与相加, 篮球 D 从 0 写到 2.3
        # 下标 i 把两张表连起来: up[i] 来自 w_up 第 i 行, h[i] 乘的是 w_down 第 i 列.
        gate = silu(linear(x, self.w_gate))  # 开关值: 每个模式一个, 过 SiLU
        up = linear(x, self.w_up)  # 升维: 每个模式的得分
        h = gate * up  # 开关: 逐个模式压放 (relu 的替身)
        return linear(h, self.w_down)  # 降维: 在场模式的写入向量按 h 加总, 回到 E 维
        # 逐行对上 (传统两行 → 这里四行; 3B1B 视频画的是传统版):
        #   传统 x @ W1.T → 拆成打分① + 打分② (W1 复制成 w_gate/w_up 两张表)
        #   传统 relu     → h = gate * up 这一步 (曲线固定无参数, 可学习的是喂进来的分数)
        #   传统 h @ W2.T → 混合行 (视频里 C_i 列按 n_i 加权再加 B; 我们无 B, n_i 就是 h)
        # 四行里可学习的只有三张表; 转置、SiLU、逐元素乘全是固定的机械, 没有参数.


class Qwen3TransformerBlock:
    """单层研究: 框架那一侧 TransformerBlock 的手写版. 研究一个 block 从打印框架的那份开始.

    .venv/bin/python debug_repl.py
    print(mlx_model.model.layers[0])

        渲染:
        TransformerBlock(
          (self_attn): Attention(
            (q_proj): QuantizedLinear(input_dims=1024, output_dims=2048, bias=False, group_size=128, bits=4, mode=affine)
            (k_proj): QuantizedLinear(input_dims=1024, output_dims=1024, bias=False, group_size=128, bits=4, mode=affine)
            (v_proj): QuantizedLinear(input_dims=1024, output_dims=1024, bias=False, group_size=128, bits=4, mode=affine)
            (o_proj): QuantizedLinear(input_dims=2048, output_dims=1024, bias=False, group_size=128, bits=4, mode=affine)
            (q_norm): RMSNorm(128, eps=1e-06)
            (k_norm): RMSNorm(128, eps=1e-06)
            (rope): RoPE(128, traditional=False)
          )
          (mlp): MLP(
            (gate_proj): QuantizedLinear(input_dims=1024, output_dims=3072, bias=False, group_size=128, bits=4, mode=affine)
            (down_proj): QuantizedLinear(input_dims=3072, output_dims=1024, bias=False, group_size=128, bits=4, mode=affine)
            (up_proj): QuantizedLinear(input_dims=1024, output_dims=3072, bias=False, group_size=128, bits=4, mode=affine)
          )
          (input_layernorm): RMSNorm(1024, eps=1e-06)
          (post_attention_layernorm): RMSNorm(1024, eps=1e-06)
        )

        打印出来的这份清单就是本类 __init__ 要装进去的东西的清单:
        self_attn → Qwen3MultiHeadAttention(…), mlp → Qwen3MLP(…),
        input_layernorm / post_attention_layernorm → RMSNorm(1024, ·).
        print(m) 是同一份内容重复 28 层, 510 行; 看一层就够了.

        keys() 是外科手术用的, 只在要确认"这一个模块里面有哪些数组"时才用:

    .venv/bin/python debug_repl.py
    print(list(mlx_model.model.layers[0].self_attn.q_proj.keys()),
            list(mlx_model.model.layers[0].self_attn.q_norm.keys()))
    q = mlx_model.model.layers[0].self_attn.q_proj
    print(q.weight.shape, q.scales.shape, q.biases.shape, q.group_size, q.bits)

        渲染:
        ['weight', 'scales', 'biases'] ['weight']
        (2048, 128) (2048, 8) (2048, 8) 128 4

        量化与否看 keys() 里有没有 scales, 判据只有这一条:

          Linear / Embedding 家族  config.json 有 "quantization" 块时, mlx_lm 在加载阶段
                                   把它们整类换成 QuantizedLinear / QuantizedEmbedding
                                   (替换逻辑在 mlx_lm/models/base.py). 这类模块 keys() 列出
                                   ['weight','scales','biases'], group_size 与 bits 是普通属性,
                                   不在 keys() 里. 整个模块交给 dequantize_linear, 它内部自己取
                                   (src/tiny_llm/quantize.py).

          RMSNorm 家族             γ 向量从不量化, 替换名单里没有它. keys() 只有 ['weight'],
                                   直接取 .weight 传给 RMSNorm, 不过 dequantize_linear.

        ── 整模型看一条事实: 答案随人名变 ─────────────────────────────

    .venv/bin/python debug_repl.py
    b = tokenizer.encode(" basketball", add_special_tokens=False)[0]
    s = tokenizer.encode(" swimming", add_special_tokens=False)[0]

    def ask(sent):
        ids = tokenizer.encode(sent, add_special_tokens=False)
        p = mx.softmax(model(mx.array([ids], dtype=mx.int32))[0, -1].astype(mx.float32))
        print(f"{sent:34}  basketball {float(p[b]):.3f}  swimming {float(p[s]):.3f}  第一名 {tokenizer.decode([int(mx.argmax(p))])!r}")

    ask("Michael Jordan plays the sport of")
    ask("Michael Phelps plays the sport of")
    ask("Maria Jordan plays the sport of")
    ask("Jordan plays the sport of")

        渲染:
        Michael Jordan plays the sport of   basketball 0.218  swimming 0.023  第一名 ' basketball'
        Michael Phelps plays the sport of   basketball 0.014  swimming 0.272  第一名 ' swimming'
        Maria Jordan plays the sport of     basketball 0.038  swimming 0.009  第一名 ' the'
        Jordan plays the sport of           basketball 0.250  swimming 0.012  第一名 ' basketball'

        - Michael Jordan：迈克尔·乔丹，打篮球。
        - Michael Phelps：菲尔普斯，游泳运动员。
        - Jordan：单独出现时只是一个姓「乔丹」，也是国家「约旦」的英文名。
        - Maria：随便取的名字，不对应任何名人。

        三句只差人名, 句子里都没有 basketball 或 swimming, 回答却跟着人名变: 这条知识
        来自参数. 只给 "Jordan plays the sport of" 时 basketball 是 0.250, 篮球主要挂在
        Jordan 这个词上; 前一个位置的名字能改掉它: Maria 压到 0.038, Phelps 换成游泳.
        别的位置的信息怎样进到预测位置, 篮球又是在哪一层写进去的, 是本类两个子层分工的问题,
        见 __call__ 注释「MLP 与 attention」一节.

        ── 切掉 attention 的一列: 内容只能经 attention 换位置 ─────────────

        实验在 28 层里同时禁止后面的位置读某一个位置, 别的都不动: 被切的位置自己照常走完
        28 层, 它的 MLP 照常运行, 只是它的内容送不到后面. 以 cut=1 (切 Jordan) 为例, 掩码表
        的行是读的人, 列是被读的人, 格子表示行的词做加权平均时能不能用到列的词:

               \  被读的人   0 Michael  1 Jordan  2 plays  3 the  4 sport  5 of
        读的人
        0 Michael              ·          —         —        —       —       —
        1 Jordan               ·          ·         —        —       —       —
        2 plays                ·          ×         ·        —       —       —
        3 the                  ·          ×         ·        ·       —       —
        4 sport                ·          ×         ·        ·       ·       —
        5 of                   ·          ×         ·        ·       ·       ·

          ·  允许读, 权重由内容现算
          —  因果掩码本来就禁止 (读不了后面的词)
          ×  这一刀新禁止的格子: Jordan 之后的每个词都读不到 Jordan

        Jordan 自己那一行没有 ×: 它照常读 Michael, 照常跑 MLP, 只是后面的词看不见它.
        cut=0 把这四个 × 换到 Michael 那一列, 并多出一格 (Jordan 读 Michael).

    .venv/bin/python debug_repl.py
    b = tokenizer.encode(" basketball", add_special_tokens=False)[0]
    s = tokenizer.encode(" swimming", add_special_tokens=False)[0]

    def run(sent, cut=None):
        ids = tokenizer.encode(sent, add_special_tokens=False)
        L = len(ids)
        m = mx.tril(mx.ones((L, L)))
        if cut is not None:
            col = (mx.arange(L) == cut)[None, :] & (mx.arange(L) > cut)[:, None]
            m = mx.where(col, 0.0, m)
        mask = mx.where(m > 0, 0.0, -mx.inf)
        h = model.embed_tokens(mx.array([ids], dtype=mx.int32))
        for blk in model.layers:
            h = h + blk.multiHeadAttention(blk.input_layernorm(h), mask=mask.astype(h.dtype))
            h = h + blk.mlp(blk.post_attention_layernorm(h))
        p = mx.softmax(model.lm_head.as_linear(model.norm(h))[0, -1].astype(mx.float32))
        print(f"{sent:34} 切断位置={str(cut):4}  basketball {float(p[b]):.3f}  swimming {float(p[s]):.3f}  第一名 {tokenizer.decode([int(mx.argmax(p))])!r}")

    for sent in ["Michael Jordan plays the sport of", "Michael Phelps plays the sport of"]:
        for cut in (None, 0, 1, 2, 3, 4):
            run(sent, cut)

        渲染:
        Michael Jordan plays the sport of  切断位置=None  basketball 0.218  swimming 0.023  第一名 ' basketball'
        Michael Jordan plays the sport of  切断位置=0     basketball 0.105  swimming 0.013  第一名 ' basketball'
        Michael Jordan plays the sport of  切断位置=1     basketball 0.020  swimming 0.012  第一名 ' his'
        Michael Jordan plays the sport of  切断位置=2     basketball 0.008  swimming 0.007  第一名 ' the'
        Michael Jordan plays the sport of  切断位置=3     basketball 0.028  swimming 0.004  第一名 ' which'
        Michael Jordan plays the sport of  切断位置=4     basketball 0.004  swimming 0.000  第一名 ' the'
        Michael Phelps plays the sport of  切断位置=None  basketball 0.014  swimming 0.272  第一名 ' swimming'
        Michael Phelps plays the sport of  切断位置=0     basketball 0.069  swimming 0.019  第一名 ' baseball'
        Michael Phelps plays the sport of  切断位置=1     basketball 0.020  swimming 0.012  第一名 ' his'
        Michael Phelps plays the sport of  切断位置=2     basketball 0.001  swimming 0.247  第一名 ' swimming'
        Michael Phelps plays the sport of  切断位置=3     basketball 0.005  swimming 0.193  第一名 ' choice'
        Michael Phelps plays the sport of  切断位置=4     basketball 0.002  swimming 0.001  第一名 ' the'

        切断位置 1, 两句都退成 ' his': 位置 1 的 MLP 一直在算, 算出的东西却到不了 of. 但切断
        不带名字的位置也会坏: Jordan 句切 plays (2) 或 sport (4), 篮球同样塌掉. 所以这一刀
        说不出被搬运的是名字的内容; 它只演示了 of 的输出依赖经 attention 读别的位置.
        "内容换位置只有 attention 一条路" 由架构决定 (见下一节 ②), 这里是演示, 不是证明.
        切断位置 0 还另有一层: 从第 3 层起, of 位置约 0.5~0.85 的注意力权重 (16 头平均) 落在
        位置 0 上, 切掉它同时拿掉了 Michael 的内容和这个注意力落点, 两者在这一刀里分不开.
        名字带来的差别经过哪一层、哪个位置写进残差流, 由下面的替换实验回答.

        ── 替换: 名字带来的差别经过哪里的 MLP 输出 ───────────────────────

        干净句与对照句只差开头的名字 (Michael Phelps / Maria Phelps). 跑对照句, 只把某一层、
        某一位置的 MLP 输出换成干净句同层同位置的那行, 看目标词的概率补回多少. 28 层 × 6 个
        位置全部扫一遍打表, 另有自检、上限、累加曲线、必要性与增量投词表几步. 实现、跑法和
        各步读法都在 src/tiny_llm/mlp_patching.py:

    .venv/bin/python debug_repl.py
    from tiny_llm.mlp_patching import run_patching
    run_patching(model, tokenizer, "Michael Phelps plays the sport of", "Maria Phelps plays the sport of", " swimming")
    run_patching(model, tokenizer, "Michael Jordan plays the sport of", "Maria Jordan plays the sport of", " basketball")

        两句的结果形状不同, 读表时对照着看:
          Phelps 句: 上限 99.0%; 层 2 × Phelps 位置一格补回 86.6%; 干净句里只换掉这一格,
                     swimming 从 0.272 掉到 0.047. 这一格既够用, 也缺不了.
          Jordan 句: 上限 94.3%; 最大一格 层 3 × Jordan 位置 44.2%, 其余分散在名字两个位置
                     的前几层; 干净句换掉最大一格 basketball 还剩 98.8%. 有冗余.
          两句最大一格的增量直接投到词表, 目标词都排在一万名以后: 这份增量不直接指向答案词,
          答案词是后面的层算出来的 (也可能是这种投影读不出早期层的内容).
        这说明名字带来的差别主要经名字位置前几层的 MLP 写进残差流. 它说明不了写进去的是
        什么, 也没有测它怎样从名字位置到达 of: 后者可以用同一套替换, 把对象换成 attention
        输出来测, 尚未做.
        Jordan 一词单独就给 basketball 0.250 (见上一节), Jordan 例子不满足 Qwen3MLP 玩具里
        "Michael 和 Jordan 同时在" 的组合条件; Phelps 例子满足.

        ── 同一个 MLP, 三种输入接法 ───────────────────────────────────

        既然 MLP 能逼近任意函数, 为什么它不能自己从别的位置取信息? 差别在算 of 这个位置时,
        输入接进来的是什么. 同一张读/被读表, 三种接法:

                    ① attention             ② 逐位置 MLP            ③ 摊平整句的 MLP
        读\被读     M  J  p  t  s  o        M  J  p  t  s  o        M  J  p  t  s  o
        M           ·                       ●                       ▣  ▣  ▣  ▣  ▣  ▣
        J           ·  ·                       ●                    ▣  ▣  ▣  ▣  ▣  ▣
        p           ·  ·  ·                       ●                 ▣  ▣  ▣  ▣  ▣  ▣
        t           ·  ·  ·  ·                       ●              ▣  ▣  ▣  ▣  ▣  ▣
        s           ·  ·  ·  ·  ·                       ●           ▣  ▣  ▣  ▣  ▣  ▣
        o           ·  ·  ·  ·  ·  ·                       ●        ▣  ▣  ▣  ▣  ▣  ▣

        格子的权重  q·k 现场算                只有自己                 每格一份独立参数
        规则        一套 W_Q/W_K/W_V          一套 W 全位置共用        每个槽位各自一块
        of 的输入   全句经同一个 V 加权       只有 x[of]               x0..x5 各经自己的 W0..W5

          ·  能读, 权重由两边内容算出      ●  只读自己      ▣  能读, 权重是绑在 (行, 列) 上的参数
          ① 的右上角空白是因果掩码禁止; ② 对角线以外的空白不是被禁止, 而是公式
          out[t] = f(x[t]) 里根本没有别的位置这个下标.

        ② 是本类 self.mlp 的真实接法. 输入 (1, 6, 1024), linear 只沿最后一维乘, 6 行各自乘
        同一张 w_up (3072, 1024), 行与行之间没有任何计算:

        x[0] (1024) ──w_up──► (3072) ──开关──► ──w_down──► out[0]
        x[1] (1024) ──w_up──► (3072) ──开关──► ──w_down──► out[1]
         ...
        x[5] (1024) ──w_up──► (3072) ──开关──► ──w_down──► out[5]   of 的输出只用到 x[5]

        ③ 是假想的接法, Qwen3 里没有. 6 行首尾相接成一条 6×1024 = 6144 维的长向量, 喂给一张
        w_up (3072, 6144). W0..W5 不是 6 张矩阵, 是这张大 w_up 按列切成的 6 块:

        长向量     [  x[0]  |  x[1]  |  x[2]  |  x[3]  |  x[4]  |  x[5]  ]   6144 个数
        w_up 的列   0~1023  1024~2047 2048~3071 3072~4095 4096~5119 5120~6143
                     = W0     = W1     = W2     = W3     = W4     = W5
        长向量 (6144) ──w_up──► (3072) ──开关──► ──w_down──► 一个输出

        Jordan 坐在槽 1, 读它的是 W1 那一块; 坐在槽 3, 读它的是 W3. 句子长到 7 个词,
        长向量放不进 6144 列.

        ② 里 Jordan 不在 of 的输入里, 逼近能力再强也没有东西可逼近: 只输入 "of" 时
        basketball 是 0.0000. ③ 里 Jordan 在输入里, 原则上学得会, 但 Jordan 在槽 1 经过
        W1, 挪到槽 3 经过 W3, W1 学到的 W3 不知道, 每个槽要单独用数据教 (固定位置混合在
        没见过的槽位上只有随机水平, 见笔记 [[固定位置混合与按内容现算的混合]] 的 numpy
        玩具实验, 不是本模型测的). ① 每个槽都经过同一个 V, 权重由内容现算, Jordan 挪到
        哪个槽规则都照用. Transformer 把两步拆开: attention 先把别的位置的内容搬进 of 的
        向量, MLP 再只处理这一个已经带着全句信息的向量.

        Jordan 换槽位, 同一套权重:

    .venv/bin/python debug_repl.py
    b = tokenizer.encode(" basketball", add_special_tokens=False)[0]

    def slot(sent):
        ids = tokenizer.encode(sent, add_special_tokens=False)
        toks = [tokenizer.decode([i]) for i in ids]
        j = toks.index(" Jordan") if " Jordan" in toks else "-"
        p = mx.softmax(model(mx.array([ids], dtype=mx.int32))[0, -1].astype(mx.float32))
        print(f"{sent:50} Jordan 在槽 {j}  basketball {float(p[b]):.4f}")

    slot("Michael Jordan plays the sport of")
    slot("Everyone knows Michael Jordan plays the sport of")
    slot("In 1990, Michael Jordan plays the sport of")
    slot("of")

        渲染:
        Michael Jordan plays the sport of                  Jordan 在槽 1  basketball 0.2181
        Everyone knows Michael Jordan plays the sport of   Jordan 在槽 3  basketball 0.8973
        In 1990, Michael Jordan plays the sport of         Jordan 在槽 8  basketball 0.4186
        of                                                 Jordan 在槽 -  basketball 0.0000

        Jordan 在槽 1、3、8, 三句都答篮球, 用的是同一套 W_Q/W_K/W_V. 概率高低随句子变,
        这里只看它在每个槽位上都还在.

        ── 逐位置独立 vs 跨位置 ────────────────────────────────────────
        三个位置里只改位置 0 的输入, 看两个子层的输出在每个位置上变了多少 (最大差值):

    .venv/bin/python debug_repl.py
    blk = model.layers[0]
    mx.random.seed(0)
    x = mx.random.normal((1, 3, 1024)).astype(mx.bfloat16)
    x2 = mx.concatenate([mx.random.normal((1, 1, 1024)).astype(mx.bfloat16), x[:, 1:, :]], axis=1)
    print("输入 ", mx.abs(x - x2).max(axis=-1)[0].tolist())
    print("mlp  ", mx.abs(blk.mlp(x) - blk.mlp(x2)).max(axis=-1)[0].tolist())
    print("attn ", mx.abs(blk.multiHeadAttention(x, mask="causal") - blk.multiHeadAttention(x2, mask="causal")).max(axis=-1)[0].tolist())

        渲染:
        输入  [4.46875, 0.0, 0.0]
        mlp   [1.296875, 0.0, 0.0]
        attn  [6.3125, 1.375, 1.875]

        mlp 只在位置 0 变, 位置 1、2 精确为 0: linear 只沿最后一维 E 做乘法, L 那一维原样
        保留, 每个位置拿同一张表各算各的. 全连接的"稠密"指同一位置内 1024 个分量与 3072 个
        中间分量两两相连, 不同位置之间没有连线. attention 三个位置全变: 因果掩码下位置 1、2
        都看得到位置 0, 位置 0 的内容经打分和加权进入它们.
    """

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
        #
        # ── 两个单元, 一个结构 ─────────────────────────────────────────
        # 归一化没有矩阵, 不学习, 不算"层". block 就是两个同构单元串联:
        # 「归一化 → 子层 → 加回」, 区别只在子层是谁:
        #   attention  每个位置按相关性从别的位置搬信息 (位置之间的通信)
        #   mlp        每个位置单独过, 按当前内容读出该补的量 (位置之内的加工)
        # 模型宽度 E=1024 从进来到出去不变, 子层只出增量; 28 层重复这个单元.
        #
        # ── 为什么加回 (残差与增量是同一件事的两面) ────────────────────
        # 子层的输出与 x 相加, 而不是替换 x. 这一个决定同时给出两件事:
        #   梯度通路: 反向传播时 +x 这条路不经过任何权重, 梯度直通浅层,
        #             深层堆叠不衰减 (对照 RNN: 梯度要穿每一步的雅可比连乘).
        #   增量语义: 子层只学"该补什么". 学不动的层 delta≈0, 等于空转,
        #             已有信息原样保留. 28 层里允许部分层贡献很小.
        # 从输入侧看叫残差 (输入被加回来), 从输出侧看叫增量 (子层只产 delta),
        # 说法不同, 加法只有一次.
        #
        # ── MLP 与 attention: 步骤相同, 被加起来的表不同 ──────────────
        # 两个子层的步骤相同: 点积打分 → 非线性定权重 → 按权重把一组向量加起来.
        # 区别只在那组向量从哪来. 以 "Michael Jordan" 里 Jordan 这个位置为例:
        #
        #              拿什么打分   和谁打分                   加起来的是谁                 表有几行
        #   mlp        x            w_up 的每一行 (检测方向)   w_down 的每一列 (写入向量)   3072, 写在参数里, 换一句话不变
        #   attention  q_Jordan     每个位置的 k               每个位置的 v                 S, 一个 token 一行, 换一句话就换
        #                           k_Michael, k_Jordan        v_Michael, v_Jordan
        #
        # attention 写的是这句话里已有的东西. q_Jordan 给 k_Michael、k_Jordan 打分, softmax
        # 定出权重, v_Michael、v_Jordan 按权重相加, Michael 的特征就这样进入 Jordan 的位置
        # (类 Qwen3MLP 的玩具里, x 的 A = 1 来自这一步). softmax 的权重和为 1, 结果的每个
        # 分量都夹在这几个 v 之间, 句子里没有的信息写不出来.
        #
        # mlp 写的是参数里存的东西. 玩具里 x 给 10 个检测方向打分, p0 得 2, p0 的写入向量
        # 被写进来, D (篮球) 从 0 变成 2.3; 整句话里没有一个 token 带着"篮球". h 不归一,
        # 2.3 超过任何一个写入向量在 D 上的值. mlp 的输入只有当前位置的 x, 读不到前一个
        # 位置的 Michael (实测见类 docstring「逐位置独立 vs 跨位置」).
        #
        # 两者因此是接力: attention 先把 Michael 搬到 Jordan 的位置, mlp 才能认出
        # "Michael 和 Jordan 同时在"这个组合, 再从参数里取出"篮球". 这是玩具的讲法;
        # 真实模型里 Jordan 单独就能给出篮球, 更贴合这个讲法的是 Phelps 句, 实测见类
        # docstring「替换」一节.
        #
        # attention 也有参数 (W_Q/W_K/W_V/W_O), 它们决定每个 token 怎样变成 q、k、v;
        # 被加起来的表仍是一个 token 一行, 由当前这句话现算. mlp 的表就是 w_down 本身,
        # 与输入是哪句话无关. Geva et al. 2021 (EMNLP, arXiv 2012.14913) 据此把 FFN 层
        # 称为键值记忆: 检测方向对应训练语料里的文本模式, 写入向量推高这些模式之后
        # 常见的下一个 token.
        x_input_layernorm = self.input_layernorm(x)
        x_multiHeadAttention = self.multiHeadAttention(x_input_layernorm, mask=mask)
        x_delta1 = x_multiHeadAttention + x

        x_post_attention_layernorm = self.post_attention_layernorm(x_delta1)
        x_mlp = self.mlp(x_post_attention_layernorm)
        x_delta2 = x_mlp + x_delta1
        return x_delta2


class Qwen3ModelWeek1:
    """从 mlx_lm 加载好的模型里把权重拆出来, 装进手写的纯函数实现.

        单层 (重复单元) 的研究在 Qwen3TransformerBlock 的 docstring 里. 这里管装配:
        数层数 → 整棵树现场渲染 → 宏观汇总 → 两份模型对账.

        ── 多少层: 28 份同构 ──────────────────────────────────────────

        .venv/bin/python debug_repl.py
        print(len(mlx_model.model.layers))
        print({type(layer).__name__ for layer in mlx_model.model.layers})

            渲染:
            28
            {'TransformerBlock'}

            28 层同一类型, 整棵树因此只展开一份.

        .venv/bin/python debug_repl.py
        def tree(mod, name, pad=""):
            if isinstance(mod, mx.array):
                print(f"{pad}{name}  {tuple(mod.shape)} {mod.dtype}")
                return
            kids = list(mod.keys()) if hasattr(mod, "keys") else []
            extra = ""
            if hasattr(mod, "group_size"): extra = f"  group_size={mod.group_size} bits={mod.bits}"
            elif hasattr(mod, "eps"): extra = f"  eps={mod.eps}"
            print(f"{pad}{name}  {type(mod).__name__}{extra}")
            for k in kids:
                v = getattr(mod, k)
                last = k == kids[-1]
                if k == "layers" and isinstance(v, list):
                    print(f"{pad}├── layers[{len(v)}]  ×{len(v)} 同构, 只展开第一层")
                    tree(v[0], "", pad + "│   ")
                else:
                    tree(v, ("└── " if last else "├── ") + k, pad + ("    " if last else "│   "))

        a = mlx_model.args
        print(f"{'E':6} = {a.hidden_size:<7} 表示宽度, 层间恒定, 也就是查表的行宽")
        print(f"{'V':6} = {a.vocab_size:<7} 词表行数, 出口打分的宽度")
        print(f"{'layers':6} = {a.num_hidden_layers:<7} block 个数, 内部见 Qwen3TransformerBlock docstring")
        print(f"{'Hq':6} = {a.num_attention_heads:<7} Q 头数, 每头 D={a.head_dim}, 16×128=2048 是 q_proj 的输出宽")
        print(f"{'H':6} = {a.num_key_value_heads:<7} KV 头数, Hq 的一半, GQA: 每个 KV 头被 2 个 Q 头共用")
        print(f"{'I':6} = {a.intermediate_size:<7} MLP 中间宽度, 3E")
        print(f"{'tie':6} = {str(a.tie_word_embeddings):<7} 出口复用查表, 框架里没有 lm_head")
        tree(mlx_model, "mlx_model")

                渲染:
                E      = 1024    表示宽度, 层间恒定, 也就是查表的行宽
                V      = 151936  词表行数, 出口打分的宽度
                layers = 28      block 个数, 内部见 Qwen3TransformerBlock docstring
                Hq     = 16      Q 头数, 每头 D=128, 16×128=2048 是 q_proj 的输出宽
                H      = 8       KV 头数, Hq 的一半, GQA: 每个 KV 头被 2 个 Q 头共用
                I      = 3072    MLP 中间宽度, 3E
                tie    = True    出口复用查表, 框架里没有 lm_head
                mlx_model  Model
                    └── model  Qwen3Model
                    │   ├── embed_tokens  QuantizedEmbedding  group_size=128 bits=4
                    │   │   ├── weight  (151936, 128) mlx.core.uint32
                    │   │   ├── scales  (151936, 8) mlx.core.bfloat16
                    │   │       └── biases  (151936, 8) mlx.core.bfloat16
                    ├── layers[28]  ×28 同构, 只展开第一层
                    │     TransformerBlock
                    │   │   ├── self_attn  Attention
                    │   │   │   ├── q_proj  QuantizedLinear  group_size=128 bits=4
                    │   │   │   │   ├── weight  (2048, 128) mlx.core.uint32
                    │   │   │   │   ├── scales  (2048, 8) mlx.core.bfloat16
                    │   │   │   │       └── biases  (2048, 8) mlx.core.bfloat16
                    │   │   │   ├── k_proj  QuantizedLinear  group_size=128 bits=4
                    │   │   │   │   ├── weight  (1024, 128) mlx.core.uint32
                    │   │   │   │   ├── scales  (1024, 8) mlx.core.bfloat16
                    │   │   │   │       └── biases  (1024, 8) mlx.core.bfloat16
                    │   │   │   ├── v_proj  QuantizedLinear  group_size=128 bits=4
                    │   │   │   │   ├── weight  (1024, 128) mlx.core.uint32
                    │   │   │   │   ├── scales  (1024, 8) mlx.core.bfloat16
                    │   │   │   │       └── biases  (1024, 8) mlx.core.bfloat16
                    │   │   │   ├── o_proj  QuantizedLinear  group_size=128 bits=4
                    │   │   │   │   ├── weight  (1024, 256) mlx.core.uint32
                    │   │   │   │   ├── scales  (1024, 16) mlx.core.bfloat16
                    │   │   │   │       └── biases  (1024, 16) mlx.core.bfloat16
                    │   │   │   ├── q_norm  RMSNorm  eps=1e-06
                    │   │   │   │   └── weight  (128,) mlx.core.bfloat16
                    │   │   │   ├── k_norm  RMSNorm  eps=1e-06
                    │   │   │   │   └── weight  (128,) mlx.core.bfloat16
                    │   │   │       └── rope  RoPE
                    │   │   ├── mlp  MLP
                    │   │   │   ├── gate_proj  QuantizedLinear  group_size=128 bits=4
                    │   │   │   │   ├── weight  (3072, 128) mlx.core.uint32
                    │   │   │   │   ├── scales  (3072, 8) mlx.core.bfloat16
                    │   │   │   │       └── biases  (3072, 8) mlx.core.bfloat16
                    │   │   │   ├── down_proj  QuantizedLinear  group_size=128 bits=4
                    │   │   │   │   ├── weight  (1024, 384) mlx.core.uint32
                    │   │   │   │   ├── scales  (1024, 24) mlx.core.bfloat16
                    │   │   │   │       └── biases  (1024, 24) mlx.core.bfloat16
                    │   │   │       └── up_proj  QuantizedLinear  group_size=128 bits=4
                    │   │   │       │   ├── weight  (3072, 128) mlx.core.uint32
                    │   │   │       │   ├── scales  (3072, 8) mlx.core.bfloat16
                    │   │   │       │       └── biases  (3072, 8) mlx.core.bfloat16
                    │   │   ├── input_layernorm  RMSNorm  eps=1e-06
                    │   │       └── weight  (1024,) mlx.core.bfloat16
                    │       └── post_attention_layernorm  RMSNorm  eps=1e-06
                    │           └── weight  (1024,) mlx.core.bfloat16
                        └── norm  RMSNorm  eps=1e-06
                            └── weight  (1024,) mlx.core.bfloat16

                lm_head 不在这棵树里: mlx_model.keys() 列不出它, hasattr(mlx_model, "lm_head") 是 False,
                tie=True, 出口复用 embed_tokens 同一张表.

                ── 宏观: 0.6B 从哪来 ──────────────────────────────────────────
                先说单位. 模型尺寸说的都是"参数个数", 一个参数就是权重矩阵里的一个数:
                  K = 10^3 (千)   M = 10^6 (百万)   B = 10^9 (十亿) = 1000 M
                  行业里的 M/B 是英文 million / billion 的缩写, "0.6B" = 6 亿 = 600 M.
                  一个参数按 bf16 存是 2 字节, 596M 个参数 ≈ 1.2 GB; 磁盘上按 4bit 打包
                  (一个 uint32 装 8 个数, 外加每 128 个数一组 scales/biases) 后约缩到 1/4.

                一层 15.73M, 每张矩阵的参数个数 = 行数 × 列数 (每个交叉点一个数):
                  attn 6.29M = q_proj 2048×1024 = 2.10M
                             + k_proj 1024×1024 = 1.05M
                             + v_proj 1024×1024 = 1.05M
                             + o_proj 1024×2048 = 2.10M
                  mlp  9.44M = gate/up/down 各 3072×1024 = 3.15M, ×3
                28 × 15.73M = 440.4M, 加 embed_tokens 151936×1024 = 155.6M (查表与出口打分共用一张表)
                合计 596.0M, "0.6B" 的 0.6 就是它 (596M ≈ 0.6B, 四舍五入到一位小数).
                norm 的 γ 和 rope 没有矩阵, 不进这个表.
                每层之内 attn : mlp = 6.29 : 9.44 ≈ 0.67 : 1; k/v 宽度是 q/o 的一半, GQA 直接写在参数形状里
                (树上 q_proj (2048, ·) 对 k/v_proj (1024, ·), 16 个 q 头共用 8 组 kv).

                装配完可以把两份模型的 logits 摆在一起, 差异应当只剩数值误差:

    .venv/bin/python debug_repl.py
    ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    a = model(ids).astype(mx.float32)
    b = mlx_model(ids).astype(mx.float32)
    print(float(mx.abs(a - b).max()), int(mx.argmax(a[0, -1])), int(mx.argmax(b[0, -1])))

                渲染:
                0.34375 3 3

                block 里两个 RMSNorm 建的时候没传 eps, 用的是默认 1e-5, 而模型要求 1e-6
                (args.rms_norm_eps); q_norm/k_norm 那两处传了, 所以同一层里两种 eps 并存.
                把这两个 norm 的 eps 改成 1e-6 后实测差值 0.34375 → 0.30078: eps 只解释一小部分,
                剩下的来源没有查.
    """

    def __init__(self, mlx_model: Any):
        args = mlx_model.args

        layers = []
        for layer in mlx_model.model.layers:
            sa = layer.self_attn
            mlp = layer.mlp
            block = Qwen3TransformerBlock(
                # 数字
                num_attention_heads=args.num_attention_heads,
                num_kv_heads=args.num_key_value_heads,
                hidden_size=args.hidden_size,
                head_dim=args.head_dim,
                intermediate_size=args.intermediate_size,
                rms_norm_eps=args.rms_norm_eps,
                # 权重
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
            )
            layers.append(block)

        self.layers = layers
        # 查表权重 (vocab, E): QuantizedEmbedding 五件套齐全, 还原成数组供 __call__ 查表
        self.embed_tokens = Embedding(
            args.vocab_size,
            args.hidden_size,
            dequantize_linear(mlx_model.model.embed_tokens),
        )
        # final norm γ: 所有 block 跑完后、出口投影前的最后一道 RMSNorm (与 layers 平级, 不是某层内部)
        self.norm = RMSNorm(args.hidden_size, mlx_model.model.norm.weight)
        # 出口投影 (vocab, E): tie=True 时框架里根本没有 lm_head 模块 (0.6B),
        # 出口复用嵌入矩阵 — 同一份 (vocab, E) 权重, 查表和 @ .T 两用
        if not args.tie_word_embeddings:
            # if hasattr(mlx_model, "lm_head"): same
            # self.lm_head = dequantize_linear(mlx_model.lm_head)
            self.lm_head = Embedding(
                args.vocab_size, args.hidden_size, dequantize_linear(mlx_model.lm_head)
            )
        else:
            self.lm_head = self.embed_tokens

        stage(
            0,
            "args",
            f"E={args.hidden_size} V={args.vocab_size} layers={len(layers)}"
            f" Hq={args.num_attention_heads} H={args.num_key_value_heads}"
            f" D={args.head_dim} I={args.intermediate_size} tie={args.tie_word_embeddings}",
        )

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        # ── 这一次调用里各个量的形状与字节数 ────────────────────────────
        # 常量 (取值来自快照 config.json 的对应字段):
        #   B  batch                   1       本次几条序列
        #   L  query 方向 token 数             本步发问几个; 这一版恒等于 S
        #   S  key/value 方向 token 数         能看见多少历史; KV 版 decode 时 L=1
        #   E  hidden_size             1024    残差流宽度: 每个 token 一条 E 维向量
        #   H  num_attention_heads     16      Q 的头数
        #   KV num_key_value_heads     8       K/V 的头数; 16 个 Q 头共用 8 组 (GQA)
        #   D  head_dim                128     每头维度
        #   I  intermediate_size       3072    MLP 中段宽度; I = 3E
        #   V  vocab_size              151936  词表行数, 也是 lm_head 的输出宽
        # 两个容易看错的宽度: H·D = 2048 (q_proj 输出, 比 E 宽一倍);
        #                    KV·D = 1024 (= E, k/v_proj 输出与残差流同宽)
        # 字节数: bf16 每元素 2 字节, ids 是 int32 每元素 4 字节
        #
        # 量                    每token   S=22      S=32768
        # ids (1,S) int32       4         88 B      128 KiB
        # h  查表 (1,S,E)       2048      44 KiB    64 MiB
        # q  (1,S,H·D)          4096      88 KiB    128 MiB
        # k, v 各 (1,S,KV·D)    2048      44 KiB    64 MiB
        # 加权和 (1,S,H·D)      4096      88 KiB    128 MiB
        # wo 输出 (1,S,E)       2048      44 KiB    64 MiB
        # gate, up 各 (1,S,I)   6144      132 KiB   192 MiB
        # h=gate*up (1,S,I)     6144      132 KiB   192 MiB
        # mlp 输出 (1,S,E)      2048      44 KiB    64 MiB
        # logits (1,S,V)        303872    6 MiB     9.3 GiB
        #
        # 除 logits 外合计 2048+4096+2048+2048+4096+2048+6144+6144+6144+2048
        #                 = 36,864 字节/token = 36 KiB/token
        #
        # scores 不在本函数里 (在 attention 内部), 峰值由它决定:
        #   (1,H,S,S) fp32 = 64·S² 字节
        #   S=22    30 KiB        S=2048   256 MiB        S=32768   64 GiB
        #   对 S 求导 = 128·S: S=32768 时每多一个 token 多 4 MiB, 是 36 KiB 的 114 倍.
        #   二次项起来之后, 其余每一项都可以忽略.
        #
        # 一个 token 的向量是 E × 2 B = 2 KiB; 64 MiB 是 32768 个 token 的总量.
        #
        # 数据流:
        #   inputs    (B, L)        int32, 行号
        #   → 查表    (B, L, E)     每个行号换成一行向量
        #   → 28 层   (B, L, E)     形状不变, 数值被反复改写
        #   → norm    (B, L, E)     最后一次 RMSNorm
        #   → lm_head (B, L, V)     每个位置对词表打分
        h = self.embed_tokens(inputs)

        # 掩码不在这里造. 它要加进的是 attention 内部的 scores, 而 scores 的
        # dtype 只有造它的那一侧知道 —— 这里传 "causal", 由 attention.py 现造.
        # (出口形状: (L, S), 0 允许看, -inf 禁止; 行是"谁在看", 列是"能看谁"。
        #  这一版 L = S = 句子长度, 所以是方阵, 第 i 行放行 i+1 列。)
        for layer in self.layers:
            h = layer(h, "causal")  # (B, L, E) -> (B, L, E)

        x = self.norm(h)  # (B, L, E) -> (B, L, E)
        return self.lm_head.as_linear(x)  # (B, L, E) -> (B, L, V)
