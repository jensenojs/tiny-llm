# 知识写在哪里, 靠什么搬运: 切掩码与激活替换

> 语言模型读 "Michael Phelps plays the sport of" 时, 下一个词是 swimming 的概率为 0.287; 把 Michael 换成 Maria, 只剩 0.073. 两句只差开头的名字, swimming 这个词又不在输入里, 所以这条知识只能来自模型的参数. 本文用两组实验追问两个后续问题. 第一个问题是名字的信息经过什么通路影响句末的预测, 用切掩码回答. 第二个问题是名字带来的差别经过哪一层、哪个位置的 MLP 输出写进模型, 用激活替换回答. 两组实验遵守同一条规则: 一次只改一处, 其余保持原样. 这套方法的来历与已知边界见 [activation-patching-lineage-notes.md](activation-patching-lineage-notes.md).

## 环境与装配

全文的代码只依赖 mlx 与 mlx_lm, 模型是 Hugging Face 上的 `Qwen/Qwen3-0.6B-MLX-4bit`, 即权重量化为 4 bit 的 Qwen3 0.6B, 首次运行时自动下载. 读数是 2026-11-03 在 Apple Silicon (arm64) 上用 Python 3.10.18、mlx 0.32.0、mlx_lm 0.31.3 测得的. 代码块要按顺序运行, 后面的块会用到前面的块定义的函数和变量; 可以依次贴进同一个 Python 交互会话, 也可以拼成一个文件. 整套在这台机器上运行约 7 秒.

先说模型的结构, 后面的实验都要在这个结构上动手. 一句话分词后得到 L 个 token, 每个 token 先在嵌入表里查出一条 1024 维的向量. 这 L 条向量叫残差流, 形状是 (1, L, 1024), 之后 28 个 block 依次处理它. 每个 block 做两步, 每一步的输出都加回残差流:

```text
resid = resid + attention(norm(resid))   # 每个位置从前面的位置读取信息
resid = resid + mlp(norm(resid))         # 每个位置只用自己的向量, 算出一份增量
```

28 个 block 之后, 最后一个位置的向量再过一次 norm, 与嵌入表的每一行做点积, 得到词表里 151936 个词各自的分数 (logits); 经 softmax 后就是下一个词的概率. 这个模型的出口复用嵌入表 (tie_word_embeddings), 所以「与嵌入表每一行做点积」就是出口的打分.

实验要在这个流程中间动手: 换 attention 的掩码, 存下每层 MLP 的输出, 把某一层某个位置的 MLP 输出换成别的值. mlx_lm 的 `model(ids)` 不提供这些入口, 所以下面照它的结构把 28 层的循环手写一遍. 每个子层调用的仍是 mlx_lm 自己的模块, 手写的只有外面这层循环; 块末尾的自检比较手写版与 `model(ids)` 的输出.

```python
import mlx.core as mx
from mlx_lm import load

model, tokenizer = load("Qwen/Qwen3-0.6B-MLX-4bit")
inner = model.model    # 嵌入表 embed_tokens、28 个 block、出口前的 norm 都挂在这里
layers = inner.layers  # 28 个 TransformerBlock


def ids_of(text):
    """文本 → (1, L) 的 token id."""
    return mx.array([tokenizer.encode(text, add_special_tokens=False)])


def token_id(word):
    """目标词 → 它的 id. 目标词必须恰好是一个 token, 否则读不出一个概率."""
    ids = tokenizer.encode(word, add_special_tokens=False)
    assert len(ids) == 1, f"{word!r} 不是单个 token: {ids}"
    return ids[0]


def forward(ids, mask="causal", record=None, plan=None):
    """把模型的前向手写一遍, 多出三个口子: mask 换掩码, record 存下每层 MLP 输出, plan 替换 MLP 输出.

    plan 是 {(层号, 位置): 1024 个数}: 跑到这一层时, 把这个位置的 MLP 输出换成给定的值.
    返回最后一个位置的 logits, 形状 (词表大小,).
    """
    resid = inner.embed_tokens(ids)  # 残差流, (1, L, 1024)
    L = ids.shape[1]
    for i, layer in enumerate(layers):
        resid = resid + layer.self_attn(layer.input_layernorm(resid), mask=mask)
        mlp_out = layer.mlp(layer.post_attention_layernorm(resid))  # (1, L, 1024)
        if record is not None:
            record.append(mlp_out)
        for (at_layer, pos), value in (plan or {}).items():
            if at_layer == i:
                is_pos = (mx.arange(L) == pos)[None, :, None]   # (1, L, 1), 只在位置 pos 为真
                mlp_out = mx.where(is_pos, value, mlp_out)      # 位置 pos 那一行换成 value
        resid = resid + mlp_out
    return inner.embed_tokens.as_linear(inner.norm(resid))[0, -1].astype(mx.float32)


def prob(logits, tid):
    return mx.softmax(logits, axis=-1)[tid].item()


def top1(logits):
    return tokenizer.decode([int(mx.argmax(logits))])


x = ids_of("Maria Phelps plays the sport of")
print("手写 forward 与 model() 的 logits 最大差:", mx.abs(forward(x) - model(x)[0, -1].astype(mx.float32)).max().item())
```

```text
手写 forward 与 model() 的 logits 最大差: 0.0
```

差值为 0.0, 说明手写的循环与模型自己的前向逐位一致. 之后所有读数里出现的变化, 都来自实验有意做的改动.

## 出发点: 答案随名字变

先确认这个现象: 同一个句式换不同的名字, 看 basketball 和 swimming 的概率怎样变.

```python
B, S = token_id(" basketball"), token_id(" swimming")
for s in ["Michael Jordan plays the sport of", "Michael Phelps plays the sport of",
          "Maria Jordan plays the sport of", "Maria Phelps plays the sport of",
          "Jordan plays the sport of", "plays the sport of", "of"]:
    lg = forward(ids_of(s))
    print(f"{s:34} basketball {prob(lg, B):.3f}  swimming {prob(lg, S):.3f}  第一名 {top1(lg)!r}")
```

```text
Michael Jordan plays the sport of  basketball 0.237  swimming 0.024  第一名 ' basketball'
Michael Phelps plays the sport of  basketball 0.016  swimming 0.287  第一名 ' swimming'
Maria Jordan plays the sport of    basketball 0.035  swimming 0.009  第一名 ' the'
Maria Phelps plays the sport of    basketball 0.037  swimming 0.073  第一名 ' tennis'
Jordan plays the sport of          basketball 0.250  swimming 0.012  第一名 ' basketball'
plays the sport of                 basketball 0.007  swimming 0.009  第一名 ' the'
of                                 basketball 0.000  swimming 0.000  第一名 'itude'
```

目标词前面的空格不能省. 这个分词器把词前的空格并进 token, `" swimming"` 是一个 token, `"swimming"` 是另一个; 句子里 of 后面接的是带空格的那个.

这些读数确立两件事. 第一, 答案词不在输入里, 它的概率只能来自参数. 第二, 权重没有改动, 答案却随句中的名字变化, 所以参数里的知识要由输入的内容触发. 两个名字触发的方式也不同. 只给姓氏 Jordan, basketball 就有 0.250, 比加上 Michael 时的 0.237 还高; 把 Michael 换成 Maria, basketball 降到 0.035. 所以 Jordan 一词本身就足以引出篮球, Maria 则把这个联想压了下去. Phelps 句里, Michael 换成 Maria 后 swimming 从 0.287 降到 0.073. 后面的激活替换以 Phelps 句为主例, Jordan 句作对照.

剩下的问题是, 名字的信息经过网络的哪些部分、在哪里起作用. 下面两组实验各回答一部分.

## 切掩码: 断开对一个位置的读取

### 做法

attention 每层算一张 L×L 的表: 行是读的一方 (正在更新自己向量的位置), 列是被读的一方 (被加权平均引用的位置). 模型接收一个加性掩码, 逐格加到分数上: 加 0 表示允许读, 加 -inf 后这一格经 softmax 的权重为 0, 表示禁止读. 平时的 `mask="causal"` 是一张下三角为 0 的表, 每个位置都读不到后面的位置.

实验在这张表上再禁止一列: 把选定位置那一列在对角线以下的格子全设为 -inf. 以切 Jordan (cut=1) 为例, 句子是 Michael(0) Jordan(1) plays(2) the(3) sport(4) of(5):

```
读的人 \ 被读的人    0 Michael   1 Jordan   2 plays   3 the   4 sport   5 of
0 Michael              ·           —          —         —        —         —
1 Jordan               ·           ·          —         —        —         —
2 plays                ·           ×          ·         —        —         —
3 the                  ·           ×          ·         ·        —         —
4 sport                ·           ×          ·         ·        ·         —
5 of                   ·           ×          ·         ·        ·         ·
```

- `·` 允许读, 权重由两个词的内容现算; `—` 因果掩码本来就禁止; `×` 这次新禁止的格子: Jordan 之后的每个词都读不到 Jordan.
- Jordan 自己那一行没有 ×. 它照常读 Michael, 照常走完 28 层, 它的 MLP 照常运行, 只是它的输出到不了后面的任何位置.
- 这张掩码在 28 层里每层都生效. 如果只在最后一层禁止, Jordan 的信息可以先在前面的层传到 plays, 再由 plays 传给 of; 每层都禁止, 就没有这种间接路线.

代码里, 掩码是一张 (L, L) 的表, 允许的格子填 0, 禁止的格子填 -inf; 把它作为 `mask` 交给每一层的 attention:

```python
def cut_mask(L, cut):
    """因果掩码, 再禁止所有后面的位置读位置 cut. 0 = 允许, -inf = 禁止."""
    allow = mx.tril(mx.ones((L, L)))  # 行 = 读的一方, 列 = 被读的一方
    if cut is not None:
        col = (mx.arange(L) == cut)[None, :] & (mx.arange(L) > cut)[:, None]
        allow = mx.where(col, 0.0, allow)
    return mx.where(allow > 0, 0.0, -mx.inf).astype(mx.bfloat16)


for s in ["Michael Jordan plays the sport of", "Michael Phelps plays the sport of"]:
    ids = ids_of(s)
    for cut in (None, 0, 1, 2, 3, 4):
        lg = forward(ids, mask=cut_mask(ids.shape[1], cut))
        print(f"{s:34} 切断={str(cut):4}  basketball {prob(lg, B):.3f}  swimming {prob(lg, S):.3f}  第一名 {top1(lg)!r}")
```

`col` 是一张布尔表: 列号等于 cut, 且行号大于 cut 的格子为真, 正好是上图里的 ×. `cut=None` 时不禁止任何额外的格子, 结果应与上一节一致.

### 读数

```text
Michael Jordan plays the sport of  切断=None  basketball 0.237  swimming 0.024  第一名 ' basketball'
Michael Jordan plays the sport of  切断=0     basketball 0.109  swimming 0.014  第一名 ' basketball'
Michael Jordan plays the sport of  切断=1     basketball 0.033  swimming 0.019  第一名 ' his'
Michael Jordan plays the sport of  切断=2     basketball 0.009  swimming 0.008  第一名 ' the'
Michael Jordan plays the sport of  切断=3     basketball 0.029  swimming 0.004  第一名 ' which'
Michael Jordan plays the sport of  切断=4     basketball 0.004  swimming 0.000  第一名 ' the'
Michael Phelps plays the sport of  切断=None  basketball 0.016  swimming 0.287  第一名 ' swimming'
Michael Phelps plays the sport of  切断=0     basketball 0.064  swimming 0.019  第一名 ' baseball'
Michael Phelps plays the sport of  切断=1     basketball 0.033  swimming 0.019  第一名 ' his'
Michael Phelps plays the sport of  切断=2     basketball 0.001  swimming 0.207  第一名 ' swimming'
Michael Phelps plays the sport of  切断=3     basketball 0.005  swimming 0.185  第一名 ' choice'
Michael Phelps plays the sport of  切断=4     basketball 0.002  swimming 0.001  第一名 ' the'
```

### 读法

切断姓氏的位置 (切断=1), 两句的第一名都变成 ' his'. 姓氏位置的 28 层 MLP 都照常运行, 算出的东西却到不了 of. 只看这一行, 很容易得出「名字的内容经 attention 搬到 of」的结论. 对照组否定了这个读法: Jordan 句切断 plays (切断=2) 或 sport (切断=4), basketball 同样只剩 0.009 和 0.004. 切断一个不带名字的位置也让预测失效, 所以这组读数分辨不出被切断的是不是名字的内容; 它只说明 of 的预测离不开经 attention 读取别的位置.

两句对同一刀的反应也不同. Phelps 句切断 plays 后 swimming 仍有 0.207, 切断 the 仍有 0.185; Jordan 句切断这两处, basketball 都在 0.03 以下. 原因没有查.

切断 Michael (切断=0) 还混着另一个因素. 下面这段代码重算每一层 attention 的权重, 看 of 这个位置的注意力有多少落在位置 0 上:

```python
def attn_weights(layer, x):
    """重算一层 attention 的权重, (1, 头数, L, L). 和 mlx_lm 的 Attention.__call__ 同一套步骤."""
    a = layer.self_attn
    _, L, _ = x.shape
    q = a.rope(a.q_norm(a.q_proj(x).reshape(1, L, a.n_heads, -1)).transpose(0, 2, 1, 3))
    k = a.rope(a.k_norm(a.k_proj(x).reshape(1, L, a.n_kv_heads, -1)).transpose(0, 2, 1, 3))
    k = mx.repeat(k, a.n_heads // a.n_kv_heads, axis=1)  # 16 个 q 头共用 8 个 kv 头
    scores = (q.astype(mx.float32) @ k.astype(mx.float32).swapaxes(-1, -2)) * a.scale
    scores = scores + mx.triu(mx.full((L, L), -mx.inf), k=1)
    return mx.softmax(scores, axis=-1)


resid = inner.embed_tokens(ids_of("Michael Phelps plays the sport of"))
to_pos0 = []
for layer in layers:
    x = layer.input_layernorm(resid)
    to_pos0.append(round(attn_weights(layer, x)[0, :, -1, 0].mean().item(), 2))  # of 读位置 0, 16 头平均
    resid = resid + layer.self_attn(x, mask="causal")
    resid = resid + layer.mlp(layer.post_attention_layernorm(resid))
print(to_pos0)
```

```text
[0.02, 0.08, 0.09, 0.85, 0.7, 0.72, 0.7, 0.57, 0.83, 0.69, 0.53, 0.57, 0.62, 0.52, 0.46, 0.65, 0.65, 0.55, 0.7, 0.71, 0.7, 0.82, 0.81, 0.84, 0.84, 0.84, 0.68, 0.77]
```

列表的第 i 个数是第 i 层里, of 分给位置 0 的注意力权重 (16 个头取平均). 从第 3 层起, 这个权重在 0.46 到 0.85 之间. 句子的第一个位置吸走大量注意力, 是语言模型里常见的现象, 文献里叫 attention sink (Xiao, Tian, Chen, Han, Lewis 2024, ICLR, "Efficient Streaming Language Models with Attention Sinks", arXiv 2309.17453). 所以切掉位置 0, 拿掉的除了 Michael 的内容, 还有这个承接大半注意力的位置本身, 这组实验分不开两者. 由切断=0 的读数推断「Phelps 要和 Michael 合在一起才引出游泳」, 证据不够.

「内容在位置之间流动只能经过 attention」由架构决定: MLP 的公式里只有当前位置 t 的下标, attention 的公式里有 t 和被读的位置 j. 切掩码是对这一点的演示, 算不上证明. 它也看不出名字带来的差别在哪里写进网络, 这要靠下一组实验.

## 激活替换: 把一格 MLP 输出换成另一句的

### 问题怎样落到代码上

上一节确定了两件事: 答案来自参数, 内容只能经 attention 在位置之间流动. 参数一共只有两组, attention 的 q_proj/k_proj/v_proj/o_proj 和 MLP 的 gate_proj/up_proj/down_proj. 下一个问题是: 名字带来的差别, 有没有经过 MLP 的输出写进残差流; 如果经过, 是哪一层、哪个位置.

回到 forward 的循环. 每一层里, MLP 先算出 `mlp_out`, 形状 (1, L, 1024), 每个位置一行 1024 个数; 下一行 `resid = resid + mlp_out` 把它加进残差流. 名字带来的差别如果经过 MLP, 就一定体现在某些行上: Michael 句和 Maria 句在这些行上取值不同, 而且正是这个不同让答案变了. 句子有 6 个位置, 模型有 28 层, 所以候选一共是 28 × 6 = 168 行.

每一行用一个地址 (层号, 位置) 来指认. 以对照句 Maria Phelps 为例:

```text
            位置0     位置1      位置2   ...  位置5
            Maria    Phelps    plays        of
  层 0      [1024]   [1024]    [1024]       [1024]
  层 1      [1024]   [1024]    [1024]       [1024]
  层 2      [1024]   [1024] ← 地址 (2, 1): 第 2 层 mlp_out[0, 1, :]
  ...
  层 27     [1024]   ...
```

两个整数只是地址, 实验要动的是地址上的 1024 个浮点数.

### 怎样检查一行: 替换优于清零

1024 个数直接看不出含义, 只能改动它, 再看答案怎样变. 改动有两种做法. 一种是清零, 把这一行全设成 0. 可这一行除了名字带来的差别, 还承担着别的加工, 清零之后答案变了, 分不清是拿掉了知识, 还是把模型弄坏了. 另一种是替换: 跑 Maria 句, 只把这一行换成 Michael 句同一地址上的那一行, 其余 167 行和所有 attention 的计算都照 Maria 句进行. 换进去的仍是一份正常的 MLP 输出, 它与原来那一行的区别只来自开头的名字. 本文采用替换, 文献里叫 activation patching.

替换后答案向 Michael 句靠拢多少, 用一个统一的读数衡量, 称为补回比例:

```text
补回比例 = (替换后的目标概率 − 对照句概率) ÷ (干净句概率 − 对照句概率)
```

干净句指 Michael 句, 对照句指 Maria 句. 0% 表示这一行与名字带来的差别无关, 100% 表示只换这一行就回到干净句的水平. 读数可以是负数, 也可以超过 100%, 下文会遇到.

前面的 `forward` 已经留好了两个口子. 传入空列表 `record`, 它会依次存下 28 层的 `mlp_out`. 传入 `plan`, 它会在指定的层把指定位置的那一行换掉. `plan` 是一个字典, 键是地址, 值是要写进去的 1024 个数, 例如 `{(2, 1): Michael 句第 2 层位置 1 的那一行}`. 替换用 `mx.where` 生成新数组, 不在原数组上赋值: 这一层的 `mlp_out` 可能已经存进了 `record`, 原地赋值会把存下的那一份也改掉.

### 六步检查

下面的 `patching` 依次做六步, 每一步回答上一步留下的一个疑问: 自检确认替换代码本身不改变结果; 上限问 168 行一起换能补回多少; 单格逐行定位; 累加问几行够用; 必要性反过来问缺了这几行行不行; 最后一步留到后面的小节, 看换进去的差值本身指向什么.

```python
def plan_from(cells, source):
    """一组地址 (层号, 位置) → 替换计划: 每个地址填 source 句同一地址的 1024 个数."""
    return {(l, p): source[l][0, p] for (l, p) in cells}


def patching(clean, corrupt, target):
    tid = token_id(target)
    ids_clean, ids_corrupt = ids_of(clean), ids_of(corrupt)
    assert ids_clean.shape == ids_corrupt.shape, "两句 token 数要相同, 位置才能一一对应"
    L = ids_clean.shape[1]
    tokens = [tokenizer.decode([t]) for t in ids_clean[0].tolist()]
    all_cells = [(l, p) for l in range(len(layers)) for p in range(L)]

    out_clean, out_corrupt = [], []  # 两句各 28 层的 MLP 输出, 每层 (1, L, 1024)
    p_clean = prob(forward(ids_clean, record=out_clean), tid)
    p_corrupt = prob(forward(ids_corrupt, record=out_corrupt), tid)
    gap = p_clean - p_corrupt

    def recovery(cells, run_on=ids_corrupt, source=out_clean):
        """换完之后目标词补回了差距的几成. 0 = 没变, 1 = 回到干净句的水平."""
        return (prob(forward(run_on, plan=plan_from(cells, source)), tid) - p_corrupt) / gap

    print(f"干净句 {p_clean:.3f}  对照句 {p_corrupt:.3f}  差距 {gap:.3f}")
    print(f"[自检] 全部地址填回对照句自己的值: {recovery(all_cells, source=out_corrupt):.1%}")
    ceiling = recovery(all_cells)
    print(f"[上限] 全部地址填干净句的值: {ceiling:.1%}")
    per_cell = {c: recovery([c]) for c in all_cells}
    print("[单格] 行 = 位置, 列 = 层, 格内 = 补回百分比")
    print(" " * 10 + "".join(f"{l:>4}" for l in range(len(layers))))
    for p in range(L):
        print(f"{tokens[p]!r:>10}" + "".join(f"{round(100 * per_cell[(l, p)]):>4}" for l in range(len(layers))))
    order = sorted(all_cells, key=lambda c: per_cell[c], reverse=True)
    print("  最大六格:", [(c, f"{per_cell[c]:.1%}") for c in order[:6]], f" 单格之和 {sum(per_cell.values()):.1%}")
    print("[累加] 按单格从高到低逐格加入:")
    for k in (1, 2, 3, 4, 5):
        r = recovery(order[:k])
        print(f"  前 {k} 格 → 补回 {r:6.1%}  P = {p_corrupt + r * gap:.3f}")
    print("[必要性] 跑干净句, 把这几格换成对照句的值:")
    for k in (1, 3, 10):
        r = recovery(order[:k], run_on=ids_clean, source=out_corrupt)
        print(f"  前 {k:>2} 格 → 剩余 {r:6.1%}  P = {p_corrupt + r * gap:.3f}")
    return dict(tid=tid, out_clean=out_clean, out_corrupt=out_corrupt, per_cell=per_cell,
                order=order, recovery=recovery, gap=gap, ids_corrupt=ids_corrupt)


phelps = patching("Michael Phelps plays the sport of", "Maria Phelps plays the sport of", " swimming")
jordan = patching("Michael Jordan plays the sport of", "Maria Jordan plays the sport of", " basketball")
```

`recovery` 是全文的尺子. 默认调用时跑的是对照句, 被换的地址从干净句取值. 必要性一步把两个参数对调: 跑干净句, 被换的地址填对照句的值, 读数改称「剩余比例」, 含义是拿掉这几行之后干净句还保留差距的几成. `patching` 返回一个字典, 后面几个小节要用到其中存下的 MLP 输出和单格读数.

### Phelps 句的读数

```text
干净句 0.287  对照句 0.073  差距 0.214
[自检] 全部地址填回对照句自己的值: 0.0%
[上限] 全部地址填干净句的值: 96.0%
[单格] 行 = 位置, 列 = 层, 格内 = 补回百分比
             0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27
 'Michael' -14  -8  -1   2   0  -1   2   2   0  -1  -1   0  -1   0   0  -1   0   1   1   1   1   0   1   1   0   0   0   0
 ' Phelps'  19 -17  91   6  13   9  38  29  35   1   1   5   4   1  -2  -1  14   7   1   9   2   2   2   0   0   1   0   0
  ' plays'  -3  -4  -4  -1  -9  -3   0   1  -1   6   1  -1   2   0   1  -1   2   2  -1   1   1   2   1   1   0   2   0   0
    ' the'  -2  -1   9  -1   1  -3  -2  -5   0   2  -2  -1  -3   0   2   1   0  -2   1   0   0   1   0   0  -1   2  -1   0
  ' sport'  -1  -2  -9   1 -12  -8 -13  -8   0  -1 -13   5   9  -1   2   2   4   0   0   1   0   0   0   1   0   2   0   0
     ' of'   1   2  -2 -15   5  -3  -4  -8   1  -4  -3  -3  -9   5  -6  -1   2  -5  -4   2   7   2  25   4   2   4  -7 -14
  最大六格: [((2, 1), '90.6%'), ((6, 1), '38.1%'), ((8, 1), '35.4%'), ((7, 1), '28.9%'), ((22, 5), '24.9%'), ((0, 1), '18.7%')]  单格之和 173.8%
[累加] 按单格从高到低逐格加入:
  前 1 格 → 补回  90.6%  P = 0.267
  前 2 格 → 补回 117.9%  P = 0.325
  前 3 格 → 补回 152.3%  P = 0.399
  前 4 格 → 补回 120.2%  P = 0.330
  前 5 格 → 补回 141.6%  P = 0.376
[必要性] 跑干净句, 把这几格换成对照句的值:
  前  1 格 → 剩余 -10.0%  P = 0.052
  前  3 格 → 剩余 -18.5%  P = 0.034
  前 10 格 → 剩余  -9.7%  P = 0.053
```

自检是 0.0%: 把 168 行全部填回对照句自己的值, 答案没有任何变化, 说明替换的代码只改了它该改的地方.

上限是 96.0%: 168 行一起换成干净句的值, 差距补回了 96%. 名字带来的差别几乎全部能经由 MLP 的输出带进来. 剩下的 4% 在 attention 一侧: 这个实验只换 MLP 的输出, 每一层 attention 仍按对照句的残差流计算. 如果上限只有百分之几十, 后面逐行找就没有意义, 因为差别主要不走这条路.

单格表里, 亮的格子集中在 Phelps 那一行的前几层: 第 2 层 91%, 第 6、7、8 层在 29% 到 38% 之间, 第 0 层 19%. plays、the、sport 三行几乎都在 ±10% 以内, 同样换一行, 换在这些位置上不起作用. of 那一行第 22 层有一个孤立的 25%. 也有负数格, 例如 Michael 那一行第 0 层 −14%, 说明换进去的值把答案推向了反方向; 原因在后面「几行一起换」一节讨论. 168 个单格读数加起来是 173.8%, 超过 100%: 好几行各自都携带了一部分相同的差别, 单独换哪一行都能补回一段.

累加一步按单格读数从高到低逐行加入. 只换第 2 层 Phelps 一行就补回 90.6%, 已经接近 96% 的上限. 再往上加, 曲线冲过 100% 并来回摆动: 2 行 117.9%, 3 行 152.3%, 4 行又降到 120.2%. 补回超过 100% 时 P 也确实高于干净句本身, 例如 3 行时 P = 0.399, 而干净句是 0.287. 几行一起换的效果不等于各自效果之和, 所以这条曲线只能读出第一行就接近上限, 后面的走势不代表「加得越多越好」.

必要性一步反过来做: 跑干净句, 只把第 2 层 Phelps 那一行换成对照句的值, swimming 从 0.287 掉到 0.052, 比对照句本身的 0.073 还低, 剩余比例是 −10.0%. 合起来看, 这一行既够用 (单独换进去补回 90.6%), 也缺不了 (单独拿掉, 答案掉到对照句以下). 掉到对照句以下的原因没有验证; 一个可能的解释是, 拿掉的是干净句依赖的写入, 换进来的 Maria 句的值又和干净句其余部分的状态不匹配, 两种损失叠在一起 (这是推断).

### Jordan 句: 对照

```text
干净句 0.237  对照句 0.035  差距 0.202
[自检] 全部地址填回对照句自己的值: 0.0%
[上限] 全部地址填干净句的值: 93.1%
[单格] 行 = 位置, 列 = 层, 格内 = 补回百分比
             0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27
 'Michael'  14  43   1   0   1   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
 ' Jordan'  20  12  13  39  18   4  -1  -1  -3   4   1   1   2   1   0  -1   5   2   1   1  -1   0   0   0   0   0   0   0
  ' plays'  -3  -4   4   2  -2  -7   3  10   1  -1  -2  -2  -1   0   0  -1   0   0   1  -1   0   1   0   1   0   0   0   0
    ' the'   0   2  -2   4   1   4   3  -1   1   0   1   0   1   2  -2   2   2   0   1   1   1   0   0   0   0   1   0   0
  ' sport'   1  -3  -1   1   3  -2   0  -1   1   3  -2   0   1   3   0   0   1   1   1   1   0   1   0   1   0   0   0   0
     ' of'  -2   8   2   1   1  -7  -4  -5   8  -3   0   7   2   2  -5   4   8  -3  -4   3   4   5   8   8  -1   4 -10   8
  最大六格: [((1, 0), '42.9%'), ((3, 1), '39.4%'), ((0, 1), '20.1%'), ((4, 1), '18.3%'), ((0, 0), '13.7%'), ((2, 1), '12.6%')]  单格之和 249.4%
[累加] 按单格从高到低逐格加入:
  前 1 格 → 补回  42.9%  P = 0.122
  前 2 格 → 补回  73.0%  P = 0.183
  前 3 格 → 补回 107.1%  P = 0.252
  前 4 格 → 补回 112.6%  P = 0.263
  前 5 格 → 补回  99.7%  P = 0.237
[必要性] 跑干净句, 把这几格换成对照句的值:
  前  1 格 → 剩余  44.4%  P = 0.125
  前  3 格 → 剩余  45.4%  P = 0.127
  前 10 格 → 剩余  -5.7%  P = 0.024
```

Jordan 句的上限同样很高 (93.1%), 但差别的分布与 Phelps 句不同. 最亮的一格只有 42.9%, 在 Michael 自己的位置 (第 1 层); 其次是 Jordan 位置的第 3 层 39.4%, 再往下是 Jordan 位置第 0、4 层各约 20%. 要累加 3 行才补回到 100% 左右. 必要性方向更能看出区别: 干净句里拿掉最亮的一行, basketball 还保留差距的 44.4%; 拿掉前 3 行也还有 45.4%; 拿掉前 10 行才降到 −5.7%. 单独拿掉任何一处, 别的行都能补回大半.

两句的对照说明, Phelps 句里「一行就够、缺了不行」是这一例的结果, 换一个例子就不成立. 这与「出发点」一节的读数相符: 只给 Jordan 一词时 basketball 已有 0.250, 名字 Michael 只贡献差别中的一部分, 而这部分又由好几行一起携带.

## 几行一起换: 效果不可加

累加曲线已经显示, 几行一起换的效果不等于各自效果之和. 下面固定第 2 层 Phelps 那一行, 分别再搭一行, 比较「一起换」的实测值和「两者单独读数之和」:

```python
r, pc = phelps["recovery"], phelps["per_cell"]
for extra in [(6, 1), (8, 1), (6, 4), (4, 4), (0, 0)]:
    both = r([(2, 1), extra])
    print(f"(2,1) {pc[(2, 1)]:6.1%} + {extra} {pc[extra]:6.1%} → 一起换 {both:6.1%}   两者之和 {pc[(2, 1)] + pc[extra]:6.1%}")
```

```text
(2,1)  90.6% + (6, 1)  38.1% → 一起换 117.9%   两者之和 128.7%
(2,1)  90.6% + (8, 1)  35.4% → 一起换 176.4%   两者之和 126.1%
(2,1)  90.6% + (6, 4) -13.2% → 一起换  56.5%   两者之和  77.4%
(2,1)  90.6% + (4, 4) -11.7% → 一起换  48.7%   两者之和  78.9%
(2,1)  90.6% + (0, 0) -13.7% → 一起换  24.7%   两者之和  77.0%
```

同样是 Phelps 位置上单独读数相近的两行, 第 6 层与第 2 层一起换得到 117.9%, 低于两者之和; 第 8 层与第 2 层一起换得到 176.4%, 比两者之和高出 50 个百分点. 三个负数格单独换时只让答案下降 12% 到 14%, 和第 2 层一起换时却分别把结果拖低 21、30、52 个百分点; 其中第 0 层 Michael 位置那一行影响最大.

不可加的原因在于, 每一行 MLP 输出都是 MLP 对这个位置当时残差流的响应. 把 Michael 句的一行放进 Maria 句, 等于把一个在 Michael 句状态下算出的响应放进另一个状态. 它与周围状态是否相容, 决定了它起正作用还是反作用; 第 0 层的值进入得最早, 后面 27 层的计算都以它为输入, 不相容造成的影响因此可能最大 (这是推断, 没有单独验证).

两行加起来超过 100% 时, 那个概率是实际的概率. 换第 2、8 两层的 Phelps 行之后, 词表上的概率加起来仍是 1.0, swimming 的概率是 0.451, 仍排第一; swimming 的 logit 是 16.75, 干净句是 15.81. 补回比例里的 100% 只是「回到干净句水平」这个标定点, 替换后的前向可以越过它. 这样的前向在正常输入下不会出现: 残差流的底子是 Maria 句, 第 2、8 两层却写进了 Michael 句的值. 读数描述的是被干预之后的模型行为, 不能直接当成模型正常工作时的分工.

## 差值本身指向什么: 投影到词表

### 投影是一次点积

替换说明了第 2 层 Phelps 那一行带着名字带来的差别, 却没有说明这 1024 个数写的是什么. 一个直接的检查是, 把两句在这一行上的差值 (Michael 句的值减 Maria 句的值) 放到出口的打分流程里, 看它会抬高哪些词.

出口的打分流程是: 最后一个位置的残差流向量 h 先过 RMSNorm, 即除以自身的均方根, 再逐分量乘一个缩放向量 γ; 然后与嵌入表的每一行做点积, 得到每个词的分数. 词 v 的分数就是 `(h / rms(h)) · γ · e_v`, 其中 e_v 是词 v 在嵌入表里的那一行. 投影检查把差值 delta 当作 h 的一部分走一遍这个流程: 乘上 γ, 再与每一行做点积. 除以均方根这一步略去, 因为它只把整个向量乘上同一个正数, 不改变各个词分数的排名.

```python
def project(delta):
    """把 1024 维的差值当成出口向量的一部分, 乘 final norm 的缩放向量, 与词表每一行做点积."""
    return inner.embed_tokens.as_linear((delta * inner.norm.weight).astype(mx.float32)[None, None, :])[0, 0]


oc, ox = phelps["out_clean"], phelps["out_corrupt"]
for (l, p) in [(2, 1), (6, 1), (8, 1), (22, 5)]:
    scores = project(oc[l][0, p] - ox[l][0, p])
    rank = int((scores > scores[S]).sum()) + 1
    top3 = [tokenizer.decode([t]) for t in mx.argsort(-scores)[:3].tolist()]
    print(f"({l:>2},{p}) 补回 {pc[(l, p)]:6.1%}  swimming 排第 {rank:>6}  前三 {top3}")
```

`inner.norm.weight` 就是 γ; `as_linear` 用嵌入表做 `x @ W.T`, 一次算出与全部 151936 行的点积. `rank` 数的是有多少个词的分数高于 swimming.

```text
( 2,1) 补回  90.6%  swimming 排第  14727  前三 [' itself', ' save', ' themselves']
( 6,1) 补回  38.1%  swimming 排第   5344  前三 ['olini', 'reator', '_trap']
( 8,1) 补回  35.4%  swimming 排第  97161  前三 ['oundary', 'صدر', 'utron']
(22,5) 补回  24.9%  swimming 排第     75  前三 ['ό', '是一位', '有一位']
```

为了确认 `as_linear` 算的确实就是与嵌入表那一行的点积, 下面从 4 bit 的嵌入表里把 swimming 那一行解量化出来, 手工点积一次, 再看几个对照词的分数:

```python
scores = project(oc[2][0, 1] - ox[2][0, 1])
emb = inner.embed_tokens
row = mx.dequantize(emb.weight[S:S + 1], emb.scales[S:S + 1], emb.biases[S:S + 1],
                    group_size=emb.group_size, bits=emb.bits)[0]  # swimming 在嵌入表里的那一行, (1024,)
v = (oc[2][0, 1] - ox[2][0, 1]) * inner.norm.weight
print(f"as_linear 给 swimming 的分数 {scores[S].item():.6f}   手工点积 {(v.astype(mx.float32) * row.astype(mx.float32)).sum().item():.6f}")
for w in [" swimming", " basketball", " tennis", " banana", " water", " the", " itself"]:
    t = token_id(w)
    print(f"{w!r:14} 分数 {scores[t].item():7.3f}  排第 {int((scores > scores[t]).sum()) + 1:>6}")
```

```text
as_linear 给 swimming 的分数 0.713841   手工点积 0.714920
' swimming'    分数   0.714  排第  14727
' basketball'  分数   0.416  排第  26864
' tennis'      分数   0.378  排第  28754
' banana'      分数   0.604  排第  18616
' water'       分数  -0.392  排第  82123
' the'         分数  -0.159  排第  63858
' itself'      分数   3.476  排第      1
```

两个数在第三位小数上差 0.001, 来自计算精度: `as_linear` 走 4 bit 的量化矩阵乘, 手工点积在 float32 下逐项相乘再求和 (这是推断, 没有逐步核对 mlx 的内部实现). 投影里没有别的步骤.

### 读法

最关键的第 2 层 Phelps 那一行, 差值投影后 swimming 排在第 14727 名, 前三名是 itself、save、themselves 这类与游泳无关的词. 分数表里有一点倾向: 三个运动词 swimming、basketball、tennis 的分数都是正的, water 和 the 是负的. 但 banana (0.604) 与 swimming (0.714) 相差不多, itself 以 3.476 遥遥领先. 从词表上读不出一个清楚的「游泳」方向.

这一读数有两种解释, 现有数据分不开. 一种是这一行写的东西本来就不是某个词的方向, 要经后面的层加工, 到出口才变成 swimming 的分数. 另一种是这种直接投到词表的读法在早期层上不可靠: 出口的 norm 和嵌入表是按最后一层的残差流训练出来的, 拿它们去读第 2 层的向量, 本身就可能读不出东西. 文献里也报告过后一种情况 (见 [activation-patching-lineage-notes.md](activation-patching-lineage-notes.md) 的 Patchscopes 一条). 所以排名靠后, 只说明这一行的差值不直接指向 swimming, 不能证明它不带游泳的信息.

另外三行的读数也说明, 补回比例和投影排名是两种不同的量. 第 8 层 Phelps 行补回 35.4%, swimming 却排在第 97161 名; 第 22 层 of 行补回 24.9%, swimming 排第 75 名, 前三名里出现了「是一位」「有一位」这类中文片段. 补回比例反映的是这一行对最终答案的影响, 中间要经过后面所有层; 投影排名反映的是这一行的差值本身与词表各行的夹角. 这四行里, 最靠后的第 22 层投影排名最靠前, 符合「越接近出口, 差值越能直接读成答案词」的预期; 但第 8 层比第 2、6 层排得更后, 只凭四行不能说排名随层号单调变化.

## 用一次反向传播估计全部格

单格表要做 168 次前向, 每次只换一行. 如果句子更长、模型更深, 格数随之增长. 一种省事的近似是 attribution patching: 只做一次反向传播, 用梯度估计每一格替换后的效果.

道理是一阶近似. 把目标词的概率 P 看作每一行 MLP 输出的函数. 在对照句上, 某一行从对照句的值 x 换成干净句的值 x', 变化量 Δ = x' − x. 如果 Δ 不大, P 的变化约等于梯度与 Δ 的点积: ΔP ≈ ∂P/∂x · Δ. 一次反向传播就能同时得到全部 168 行的梯度 ∂P/∂x, 再与各自的 Δ 做点积, 就得到全部格的估计.

代码在 forward 的每一层 MLP 输出上加一个偏移 `extra[i]`, 偏移全为零时就是对照句的原样前向; 对这些偏移求梯度, 得到的就是 ∂P/∂x:

```python
def attribution(run):
    ids, oc, ox, tid, gap = run["ids_corrupt"], run["out_clean"], run["out_corrupt"], run["tid"], run["gap"]

    def p_target(extra):  # extra: 每层加在 MLP 输出上的偏移, 全零时就是对照句的原样前向
        resid = inner.embed_tokens(ids)
        for i, layer in enumerate(layers):
            resid = resid + layer.self_attn(layer.input_layernorm(resid), mask="causal")
            resid = resid + layer.mlp(layer.post_attention_layernorm(resid)) + extra[i]
        logits = inner.embed_tokens.as_linear(inner.norm(resid))[0, -1].astype(mx.float32)
        return mx.softmax(logits)[tid]

    grads = mx.grad(p_target)([mx.zeros_like(m) for m in ox])  # 一次反向, 得到每层每个位置的梯度
    return {(l, p): ((oc[l] - ox[l]) * grads[l])[0, p].astype(mx.float32).sum().item() / gap
            for (l, p) in run["per_cell"]}


for name, run in [("Phelps", phelps), ("Jordan", jordan)]:
    est, act = attribution(run), run["per_cell"]
    cells = list(act)
    a, e = mx.array([act[c] for c in cells]), mx.array([est[c] for c in cells])
    corr = (((a - a.mean()) * (e - e.mean())).mean() / (a.std() * e.std())).item()
    best = run["order"][0]
    print(f"{name}: 相关系数 {corr:.3f}; 估计最大三格 {[(c, f'{est[c]:.1%}') for c in sorted(cells, key=lambda c: est[c], reverse=True)[:3]]};"
          f" 实测最大格 {best} 实测 {act[best]:.1%} 估计 {est[best]:.1%}")
```

`mx.grad(p_target)` 返回一个函数, 它对 `p_target` 的第一个参数求梯度; 这里第一个参数是 28 个偏移组成的列表, 所以得到 28 个梯度, 每个形状 (1, L, 1024). 估计值也除以差距, 与补回比例同一口径.

```text
Phelps: 相关系数 0.543; 估计最大三格 [((8, 1), '22.5%'), ((22, 5), '21.8%'), ((5, 1), '10.9%')]; 实测最大格 (2, 1) 实测 90.6% 估计 6.5%
Jordan: 相关系数 0.698; 估计最大三格 [((3, 1), '22.7%'), ((1, 0), '16.9%'), ((7, 2), '12.3%')]; 实测最大格 (1, 0) 实测 42.9% 估计 16.9%
```

估计与 168 格实测的相关系数, Phelps 句是 0.543, Jordan 句是 0.698. 它能把亮格大致筛出来: Jordan 句估计最大的两格, 正是实测最大的两格. 但 Phelps 句里实测补回 90.6% 的那一行, 线性估计只有 6.5%, 连前三都没有进.

估计失准的原因与「几行一起换」一节的不可加是同一件事. 一阶近似假设 P 随每一行的变化是线性的, 而第 2 层 Phelps 行的替换后面还要经过 26 层非线性计算; Δ 也不小, 它是两个不同名字造成的完整差别. 线性近似在这里只适合粗筛, 定位仍要靠逐格替换.

## 收拢: 这些实验确立了什么

| 实验 | 改动了什么 | 读数 | 能说明什么 |
|---|---|---|---|
| 换名字 | 输入 | swimming 0.287 → 0.073; 只输入 "of" 时为 0.000 | 答案来自参数, 由输入内容触发 |
| 切掩码 | attention 能读的位置 | 切断姓氏位置, 两句都退成 ' his'; 切断 plays、sport 也会失效 | of 的预测依赖经 attention 读取别的位置; 分辨不出读的是什么 |
| 激活替换 | 单行 MLP 输出 | Phelps 句: 第 2 层 Phelps 行补回 90.6%, 拿掉后 P = 0.052 | 名字带来的差别主要经名字位置前几层的 MLP 写入 |
| 对照例 | 换成 Jordan 句 | 最亮一行 42.9%, 拿掉后仍剩 44.4% | 集中与否随例子而变 |
| 增量投影 | 不改动, 只读 | 第 2 层 Phelps 行的差值, swimming 排第 14727 | 这一行的差值不直接指向答案词 |
| 梯度估计 | 不改动, 一次反向 | 与实测相关 0.543 / 0.698; 关键行估计 6.5% | 线性近似只能粗筛 |

把能确定的部分连起来, 「Michael Phelps → swimming」这条事实在这个模型里的路径是这样的. 第 2 层前后, Phelps 位置上的 MLP 写下了与名字有关的差别; 这份差别不是 swimming 这个词的方向, 要由后面的层继续加工. 加工后的信息经 attention 从 Phelps 位置到达 of 位置, 这一段由架构决定, 切掩码只是演示了它. 最后 of 位置上的向量经出口打分, swimming 排到第一.

这条路径里还有两段没有测. 一段是第 2 层之后, 是哪些层、哪些部件把这份差别变成了 swimming 的分数. 另一段是它在哪些层、经哪些注意力头从 Phelps 位置到达 of 位置. 两段都可以用本文的替换方法回答, 只要把被替换的对象从 MLP 输出换成 attention 的输出, 或者单个注意力头的输出.

## 边界与下一步

- 本文只有两组句子, 每组一个干净句、一个对照句、一个目标词. Phelps 句「一行够用、缺了不行」的结论, 换成 Jordan 句就不成立, 不能推广成「事实存在某一行里」.
- 读数对数值精度敏感. 同样的实验在另一份手写实现上跑过: 那份实现把同一个模型的权重解量化成 bf16, attention 在 float32 下计算. 两者的数字略有不同: Phelps 句干净句 0.272、对照句 0.070, 第 2 层 Phelps 行补回 86.6%, 上限 99.0%. Jordan 句里, 补回最大的一行在那份实现上是第 3 层 Jordan 行 (44.2%), 在本文的实现上是第 1 层 Michael 行 (42.9%). 两行的读数本来就接近, 排名第一的位置随精度改变. 读单格表时, 应当看亮格集中在哪一片, 不要只看第一名是哪一格.
- 替换之后的前向在正常输入下不会出现, 读数描述的是被干预之后的模型行为. 补回超过 100% 和负数格都是这一点的表现.
- 读数用的是目标词的概率. 文献建议改用 logit difference, 即目标词与对照答案的 logit 之差, 因为概率可能掩盖起反作用的部件; 本文没有按这种指标重扫.
- 这个模型的出口复用嵌入表, 所以把差值投到词表, 同时也是在比较它与各个词嵌入的相似度; 早期层上这种读法可能读不出东西.
- 下一步可以做三件事: 把替换对象换成 attention 输出或单个注意力头, 测前面没有测的两段路径; 用 logit difference 重扫单格表, 看亮格是否不变; 多找几组「名字 → 属性」的句子, 看 Phelps 和 Jordan 两种分布哪一种更常见.
