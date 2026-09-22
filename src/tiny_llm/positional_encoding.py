import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        # 维度和位置是两回事——这正是位置编码存在的原因
        # cos 表（(seq_len, M)）：
        #             频率 0    频率 1    频率 2    频率 3
        # 位置 0  [cos(0·θ₀) cos(0·θ₁) cos(0·θ₂) cos(0·θ₃)]
        # 位置 1  [cos(1·θ₀) cos(1·θ₁) cos(1·θ₂) cos(1·θ₃)]
        # 位置 2  [cos(2·θ₀) ...                          ]
        #
        # sin 表（(seq_len, M)）：
        #             频率 0    频率 1    频率 2    频率 3
        # 位置 0  [sin(0·θ₀) sin(0·θ₁) sin(0·θ₂) sin(0·θ₃)]
        # 位置 1  [sin(1·θ₀) sin(1·θ₁) sin(1·θ₂) sin(1·θ₃)]
        # 位置 2  [sin(2·θ₀) ...                          ]
        # 位置 seq_len-1                                      ← 最后一行
        # - 行 = 位置（高度 = seq_len）：每个 token 位置一行。
        #        这就是为什么表的高度由 seq_len 决定——它要为每个可能出现的位置准备好旋转角度
        # - 列 = 频率（宽度 = M）：$M = D/2$ 个频率，每个频率管一对坐标
        # 这正好是主笔记外推性讨论的工程版：RoPE 的公式是连续的，理论上任意位置都能现算（所以有外推潜力）——
        # 但 Day 2 的实现是预计算表，表的高度 seq_len 就把位置范围钉死了。超出就得现算或插值，这就是后来长上下文要处理的问题。
        self.dims = dims  # D 是单头维度
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        self.M = self.dims // 2

        # 角速度，用于旋转角度计算
        # angular_rate = 1.0 / (self.base ** (mx.arange(self.M) / self.M))
        angular_rate = mx.power(self.base, -mx.arange(self.M) / self.M)
        position = mx.arange(self.seq_len)
        # 在每个位置上，计算每个频率的旋转角度
        angle = mx.outer(position, angular_rate)
        self.cos_freqs = mx.cos(angle)
        self.sin_freqs = mx.sin(angle)

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        # ① 选位置行: offset=None → 位置 0..L-1, 取表的前 L 行
        # x: (N, L, H, D)
        # cos/sin_freqs: (MAX_SEQ_LEN, D // 2)
        L = x.shape[1]
        cos = self.cos_freqs[:L]
        sin = self.sin_freqs[:L]

        # ② 广播: (L, M) → (1, L, 1, M), 让 MLX 自动配到 (N, L, H, M)

        cos = self.cos_freqs[offset] if offset is not None else self.cos_freqs[:L]
        sin = self.sin_freqs[offset] if offset is not None else self.sin_freqs[:L]

        cos = cos.astype(x.dtype)
        sin = sin.astype(x.dtype)

        cos = mx.reshape(cos, (1, L, 1, self.M))
        sin = mx.reshape(sin, (1, L, 1, self.M))

        # N 是 batch 维: 每个样本独立旋转, 但共用同一套表——
        # 位置→角度是全局规则, 和哪个样本无关。
        # 广播: 尺寸 1 的轴自动扩展到 N 和 H, 每批每头拿同一行 cos/sin。

        # ③ 分拣配对: 两种配对的拆法不同, 旋转公式共用
        if self.traditional:
            # 相邻配对: reshape 把最后一维 D 按顺序切成 (M, 2): [0,1], [2,3], [4,5]...
            # 相邻配对免费获得, 不用手动搬
            x_pairs = mx.reshape(x, (*x.shape[:-1], self.M, 2))
            x_a = x_pairs[..., 0]  # 每对的第一个坐标 = 极坐标里的 cos 坐标 (r·cosα)
            x_b = x_pairs[..., 1]  # 每对的第二个坐标 = 极坐标里的 sin 坐标 (r·sinα)
        else:
            # 半维配对: 配对是 (x_i, x_{i+M}), 把 D 切成两半
            #   x1 = x[..., :M], x2 = x[..., M:]   # 各 (N, L, H, M)
            # 然后 (x1, x2) 配对, 旋转公式不变
            x_a = x[..., : self.M]
            x_b = x[..., self.M :]

        # ④ 旋转公式（两角和展开的代入，见主笔记 note）：
        #   out_a: x' = x·cosθ − y·sinθ    (cos 公式 cc − ss)
        #   out_b: y' = y·cosθ + x·sinθ    (sin 公式 sc + cs)
        #   sin 公式里 s(sinα = x_b) 排在前头、c(cosα = x_a) 排在后头——配对是反的
        out_a = x_a * cos - x_b * sin
        out_b = x_b * cos + x_a * sin

        # ⑤ 拼回: 两种配对的拼回方式不同
        if self.traditional:
            # 相邻配对: stack 交错 [a0,b0,a1,b1,...] → (N, L, H, M, 2) → reshape 回 (N, L, H, D)
            x_stack = mx.stack([out_a, out_b], axis=-1)
            x_recover = mx.reshape(x_stack, (*x.shape[:-1], self.dims))
        else:
            # 半维配对: out_a 是旋转后的前半、out_b 是旋转后的后半, 直接拼接
            x_recover = mx.concatenate([out_a, out_b], axis=-1)
        return x_recover
