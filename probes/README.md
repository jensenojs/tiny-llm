# probes

观测脚本，按 `week<N>/day<M>/` 放。每个脚本只回答一个问题，打印可复核的数值，不做断言、不改被观测的对象。

跑法（在仓库根目录）：

```bash
PYTHONPATH=$PWD:$PWD/src .venv/bin/python probes/week1/day6/zero_mask_two_hop.py
```

`$PWD` 要进 `PYTHONPATH`，因为脚本从根目录的 `model_names.py` 取模型短名；`$PWD/src` 供 `tiny_llm`。脚本放在子目录里，Python 只把脚本自己所在的目录加进 `sys.path`，所以这两段都得给。

命名按"它回答什么"，不按发现顺序。新脚本进对应的 `week/day` 目录，并在下面的表里加一行。

## week1/day6

流式解码器：

| 脚本 | 回答的问题 |
| :--- | :--- |
| `detokenizer_minimal_model.py` | 纯 Python 复刻的 `last_segment`：连读两次，第二次拿到什么 |
| `detokenizer_consuming_read.py` | 真货 `tokenizer.detokenizer` 上同一个问题 |
| `prompt_and_decode_snapshot.py` | prompt 渲染结果、token 数、top-8、8 步贪心、解码器逐步输出 |

## 因果掩码的 dtype

| 脚本 | 回答的问题 |
| :--- | :--- |
| `mask_dtype_values.py` | `causal_mask` 在 int32/int64/uint32 下的实际内容，以及 `scores + mask` 的 dtype |
| `input_dtype_propagation.py` | 输入 dtype 换三种时的 top5；`argmax` 的返回类型；`concatenate` 的提升规则 |
| `mask_criterion_vs_behaviour.py` | `zeros/row` 判据与屏蔽位实际权重在 float32/int64/bool 下各是多少（不加载模型，数可手算） |
| `mask_forced_float32.py` | 把 `causal_mask` 猴补成恒用 float32，输入 dtype 还影响结果吗 |
| `activation_delta_by_stage.py` | int32 与 int64 输入逐级比对激活值 max\|Δ\|，定位差异出现在哪一级 |
| `zero_mask_two_hop.py` | 两边都 float32、只换掩码内容，逐层比对末尾位置与前序位置 |
