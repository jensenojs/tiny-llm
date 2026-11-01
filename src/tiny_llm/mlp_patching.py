"""名字带来的差别, 有没有经过 MLP 的输出写进残差流? 经过哪些层、哪些位置? 至少要几处?

跑法 (仓库根目录):
    PYTHONPATH=$PWD:$PWD/src .venv/bin/python -m tiny_llm.mlp_patching
    PYTHONPATH=$PWD:$PWD/src .venv/bin/python -m tiny_llm.mlp_patching \
        "Michael Jordan plays the sport of" "Maria Jordan plays the sport of" " basketball"
在 debug_repl 里:
    from tiny_llm.mlp_patching import run_patching
    run_patching(model, tokenizer, "Michael Phelps plays the sport of", "Maria Phelps plays the sport of", " swimming")

前提链:
  1. 换掉开头的名字, 句末答案跟着变, 句子里却没有答案这个词. 所以答案来自参数.
  2. 参数只有两组: attention 的 W_Q/W_K/W_V/W_O, MLP 的 w_gate/w_up/w_down.
  3. 内容换位置只能经 attention (架构如此, 切列实验是演示). 这没有说明答案存在哪组参数里.
  4. 所以要问: 名字带来的差别, 有没有经过 MLP 的输出?

做法 (activation patching):
  干净句与对照句只差开头的名字, token 数相同, 位置一一对应.
  每层 MLP 对每个位置写一行增量进残差流, 28 层 × L 个位置 = 28L 行.
  跑对照句的前向, 只把选中的几行换成干净句同层同位置的那行, 其余全部照旧.
  读数: 补回比例 = (换完的目标概率 - 对照句概率) / (干净句概率 - 对照句概率).

本文件只扫描和打表, 不预设哪一层、哪一处; 打印里的摘要都由数据算出.
"""

import sys

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from .qwen3_week1 import Qwen3ModelWeek1

# 一格 = 一个地址 (层号, 位置), 指向某一层 MLP 输出 m (1, L, E) 里的一行.
# 两个整数只是地址, 要换的是地址上的 1024 个浮点数. 以对照句 Maria Phelps 为例:
#
#             位置0     位置1      位置2   ...  位置5
#             Maria    Phelps    plays        of
#   层 0      [1024]   [1024]    [1024]       [1024]
#   层 1      [1024]   [1024]    [1024]       [1024]
#   层 2      [1024]   [1024] ← (2, 1) 指的是这一行, 即第 2 层的 m[0, 1, :]
#   ...
#   层 27     [1024]   ...
#
Cell = tuple[int, int]

# 替换计划: 地址 → 要写进这个地址的 1024 个数.
# 例: 把 (2, 1)、(8, 1) 两个地址换成干净句的值:
#     {(2, 1): mlp_out_clean[2][0, 1],     形状 (1024,)
#      (8, 1): mlp_out_clean[8][0, 1]}     形状 (1024,)
# forward 跑到第 2 层时, 把对照句第 2 层 MLP 输出的位置 1 换成第一个值;
# 不在计划里的地址保留对照句自己算出的值.
Plan = dict[Cell, mx.array]


def plan_from(cells: list[Cell], source: list[mx.array]) -> Plan:
    """给出一组地址, 从另一句话每层的 MLP 输出 source 里取出这些地址上的值, 组成计划.

    source[layer]          那句话第 layer 层的 MLP 输出, (1, L, E)
    source[layer][0, pos]  其中位置 pos 那一行, (E,), 即 1024 个数
    """
    plan: Plan = {}
    for layer, pos in cells:
        plan[(layer, pos)] = source[layer][0, pos]
    return plan


# (干净句, 对照句, 目标词). 两句只差开头的名字.
CASES = [
    # 主例: Michael 换成 Maria, swimming 从 0.272 掉到 0.070.
    (
        "Michael Phelps plays the sport of",
        "Maria Phelps plays the sport of",
        " swimming",
    ),
    # 对照例: 只给 "Jordan plays the sport of" 时 basketball 已有 0.250
    # (见 qwen3_week1.py「答案随人名变」), Michael 这个名字只贡献一部分.
    # 用它检验主例的结果形状是否通用: 主例集中在一格, 这一例分散在多格, 拿掉一格别处能补上.
    (
        "Michael Jordan plays the sport of",
        "Maria Jordan plays the sport of",
        " basketball",
    ),
]


def forward(
    model: Qwen3ModelWeek1,
    ids: mx.array,
    record: list[mx.array] | None = None,
    plan: Plan | None = None,
) -> mx.array:
    """手写一遍 Qwen3ModelWeek1.__call__, 计算完全相同, 只多出两个口子:
    record 把每层 MLP 输出存下来, plan 把指定地址上的 MLP 输出换掉.

    ids:    (1, L) 的 token id.
    record: 传入空列表时, 依次存下每层 MLP 的输出 (1, L, E), 共 28 个.
    plan:   替换计划, 见 Plan. 不传就是原样前向.
    返回最后一个位置的 logits (V,).
    """
    resid = model.embed_tokens(ids)  # 残差流, (1, L, E)
    plan = plan or {}
    for layer_idx, layer in enumerate(model.layers):
        resid = resid + layer.multiHeadAttention(
            layer.input_layernorm(resid), mask="causal"
        )
        mlp_out = layer.mlp(layer.post_attention_layernorm(resid))  # (1, L, E)
        if record is not None:
            record.append(mlp_out)
        for (at_layer, pos), value in plan.items():
            if at_layer == layer_idx:
                mlp_out = replace_row(mlp_out, pos, value)
        resid = resid + mlp_out  # 换过的 MLP 输出照常加回残差流, 后面的层照常计算
    return model.lm_head.as_linear(model.norm(resid))[0, -1]


def replace_row(mlp_out: mx.array, pos: int, value: mx.array) -> mx.array:
    """返回一份新数组: mlp_out 里位置 pos 那一行换成 value, 其余行不变.

    效果等于 mlp_out[0, pos, :] = value. 不原地赋值, 因为这个数组可能已经存进 record,
    原地改会把存下的那份也改掉. 以 L=6, pos=1 为例:
      mx.arange(L) == pos     [F, T, F, F, F, F]   (L,)
      [None, :, None]         补成 (1, L, 1), 与 mlp_out 的 (1, L, E) 对齐
      mx.where(条件, A, B)    逐元素挑: 条件为真取 A, 为假取 B.
                              位置 1 那一行取 value 的 1024 个数, 其余 5 行取原值.
    """
    L = mlp_out.shape[1]
    is_pos = (mx.arange(L) == pos)[None, :, None]
    return mx.where(is_pos, value, mlp_out)


def _prob(logits: mx.array, token_id: int) -> float:
    """logits (V,) → 这个 token 的概率."""
    return mx.softmax(logits.astype(mx.float32), axis=-1)[token_id].item()


def run_patching(
    model: Qwen3ModelWeek1,
    tokenizer: TokenizerWrapper,
    clean: str,
    corrupt: str,
    target: str,
) -> None:
    """跑一组 (干净句, 对照句, 目标词), 分六步打印. 每一步都是用不同参数调 recovery."""

    def encode(text: str) -> mx.array:
        return mx.array(tokenizer.encode(text, add_special_tokens=False))[None, :]

    # ── 准备: 两句分词, 检查两句的位置能一一对应, 目标词是单个 token ──
    ids_clean, ids_corrupt = encode(clean), encode(corrupt)
    tokens = [tokenizer.decode([t]) for t in ids_clean[0].tolist()]
    tokens_corrupt = [tokenizer.decode([t]) for t in ids_corrupt[0].tolist()]
    assert ids_clean.shape == ids_corrupt.shape, (
        f"两句 token 数不同: {tokens} vs {tokens_corrupt}"
    )
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    assert len(target_ids) == 1, f"目标词不是单个 token: {target!r} -> {target_ids}"
    target_id = target_ids[0]
    n_layers = len(model.layers)
    L = ids_clean.shape[1]
    all_cells: list[Cell] = [
        (layer, pos) for layer in range(n_layers) for pos in range(L)
    ]

    # ── 两次不做替换的前向: 存下两句每层的 MLP 输出, 以及目标词的概率 ──
    mlp_out_clean: list[mx.array] = []  # 干净句每层的 MLP 输出, 28 个 (1, L, E)
    mlp_out_corrupt: list[mx.array] = []  # 对照句每层的 MLP 输出, 28 个 (1, L, E)
    p_clean = _prob(forward(model, ids_clean, record=mlp_out_clean), target_id)
    p_corrupt = _prob(forward(model, ids_corrupt, record=mlp_out_corrupt), target_id)
    gap = p_clean - p_corrupt

    # ── 一把尺子: 换完之后, 目标词补回了两句差距的百分之几 ──
    # 默认: 跑对照句, 被换的地址填干净句的值. 0% = 没变, 100% = 回到干净句的水平.
    # 必要性那一步反过来: 跑干净句, 被换的地址填对照句的值.
    def recovery(
        cells: list[Cell],
        run_on: mx.array = ids_corrupt,
        source: list[mx.array] = mlp_out_clean,
    ) -> float:
        p = _prob(forward(model, run_on, plan=plan_from(cells, source)), target_id)
        return (p - p_corrupt) / gap

    print("=" * 100)
    print(f"干净句 : {clean!r}  ->  P({target!r}) = {p_clean:.3f}")
    print(f"对照句 : {corrupt!r}  ->  P({target!r}) = {p_corrupt:.3f}")
    print(f"分词   : {tokens}  vs  {tokens_corrupt}")
    print(f"差距   : {gap:.3f}    (补回比例以它为 100%)")

    print("\n[自检] 问: 手写的 forward 与模型本身一致吗? 替换代码有没有改到别处?")
    ref = model(ids_corrupt)[0, -1]
    print(
        f"  forward 与 model() 的 logits 最大差: {mx.abs(forward(model, ids_corrupt) - ref).max().item()}"
    )
    self_fill = recovery(all_cells, source=mlp_out_corrupt)
    print(
        f"  把对照句 {len(all_cells)} 个地址填回它自己的值, 补回: {self_fill:.1%}   (应为 0%)"
    )

    print("\n[上限] 问: 名字带来的差别, 有多少能经由 MLP 的输出带进来?")
    print(
        f"  做法: 对照句的全部 {len(all_cells)} 个地址一次填成干净句的值. attention 仍按对照句的残差流计算."
    )
    ceiling = recovery(all_cells)
    print(f"  补回: {ceiling:.1%}")

    print("\n[单格] 问: 差别经过了哪一层、哪个位置的 MLP 输出?")
    print(
        f"  做法: 每次只换一个地址, 共 {n_layers}×{L} = {len(all_cells)} 次."
        "  读法: 0% = 这一格与名字无关, 100% = 只换这一格就补回全部差距."
    )
    per_cell = {cell: recovery([cell]) for cell in all_cells}
    print(
        "  "
        + " " * 10
        + "".join(f"{layer:>4}" for layer in range(n_layers))
        + "   (层号; 格内是补回百分比)"
    )
    for pos in range(L):
        print(
            f"  {tokens[pos]!r:>10}"
            + "".join(
                f"{round(100 * per_cell[(layer, pos)]):>4}" for layer in range(n_layers)
            )
        )
    by_recovery = sorted(
        all_cells, key=lambda cell: per_cell[cell], reverse=True
    )  # 补回从高到低
    best_layer, best_pos = by_recovery[0]
    print(
        f"  摘要 (由数据算出): 最大一格 层{best_layer} 位置{best_pos} {tokens[best_pos]!r} {per_cell[by_recovery[0]]:.1%};"
        f" 超过 20% 的格子 {sum(r > 0.2 for r in per_cell.values())} 个, 超过 50% 的 {sum(r > 0.5 for r in per_cell.values())} 个;"
        f" 全部单格之和 {sum(per_cell.values()):.1%}"
    )

    print("\n[最少几处] 问: 至少要换几格, 才能接近上限?")
    print(
        "  做法: 按单格补回从高到低排序, 逐格累加着换. 排序是启发式, 曲线给的是上界估计."
    )
    print("  补回可以超过 100%: 换进去的几格合起来, 可能让目标概率高过干净句本身.")
    curve = []
    for k in range(1, 11):
        curve.append(recovery(by_recovery[:k]))  # 简单贪心
        layer, pos = by_recovery[k - 1]
        print(
            f"  换 {k:>2} 格 (新加 层{layer:>2} 位置{pos} {tokens[pos]!r:>10}) -> 补回 {curve[-1]:6.1%}"
            f"   P = {p_corrupt + curve[-1] * gap:.3f}"
        )
    needed = next((k + 1 for k, r in enumerate(curve) if r >= 0.9 * ceiling), None)
    print(
        f"  摘要 (由数据算出): 上限 {ceiling:.1%}; 达到上限 90% 需要 {needed if needed else '超过 10'} 格"
    )

    print("\n[必要性] 问: 反过来, 干净句里把这些格换成对照句的值, 答案掉多少?")
    print(
        "  读法: 剩余比例 = (换完的概率 - 对照句概率) / 差距. 100% = 没影响, 0% = 掉到对照句的水平."
    )
    for k in (1, 3, 10):
        remaining = recovery(by_recovery[:k], run_on=ids_clean, source=mlp_out_corrupt)
        print(
            f"  干净句换掉前 {k:>2} 格 -> 剩余 {remaining:6.1%}   P = {p_corrupt + remaining * gap:.3f}"
        )

    print(
        "\n[增量本身] 问: 最大那一格的增量 (干净 - 对照), 不经过后面的层直接投到词表, 偏向谁?"
    )
    print("  做法: 只乘 final norm 的缩放向量与词表 (略去 RMS 归一, 不影响排名).")
    print(
        "  读法: 目标词排名靠前 = 增量本身已指向答案词; 靠后 = 增量不直接指向答案词,"
        " 要经后面的层才变成答案, 或者这种投影读不出早期层的内容."
    )
    delta = (
        mlp_out_clean[best_layer][0, best_pos]
        - mlp_out_corrupt[best_layer][0, best_pos]
    )  # (E,)
    scores = model.lm_head.as_linear(
        (delta * model.norm.weight).astype(mx.float32)[None, None, :]
    )[0, 0]  # (V,)
    rank = (scores > scores[target_id]).sum().item()
    top5_ids = mx.argsort(-scores)[:5].tolist()
    print(
        f"  {target!r} 在 {scores.shape[0]} 个词里排第 {rank + 1}; 前 5: {[tokenizer.decode([t]) for t in top5_ids]}"
    )


if __name__ == "__main__":
    from mlx_lm import load

    from model_names import shortcut_name_to_full_name
    from tiny_llm import models

    name = shortcut_name_to_full_name("qwen3-0.6b")
    mlx_model, tokenizer = load(name)
    with mx.stream(mx.gpu):
        model = models.dispatch_model(name, mlx_model, week=1)
    cases = [tuple(sys.argv[1:4])] if len(sys.argv) == 4 else CASES
    for case in cases:
        run_patching(model, tokenizer, *case)
