# Day 6 真机验收排障实录: 单元测试全绿, 真机一个字都不吐

> 记录 2026-10-15 晚的排查全过程。起点是一个所有测试都通过、但 `pdm run main` 无声退出的状态;终点是两个 bug 和一条排查路数。按时间线写,每个判断都带上当时的证据,方便日后回看"当时是怎么想到的"。

## 现场还原

Day 6 的交付物是 `simple_generate` (src/tiny_llm/generate.py): prefill 产第一个 token, 循环里判 EOS、喂流式 detokenizer、产下一个。`pdm run test --week 1 --day 6` 7 passed。然后跑真验收:

```bash
pdm run main --solution tiny_llm --loader week1 --model qwen3-0.6b \
  --prompt "Give me a short introduction to large language model"
```

进程正常退出 (exit 0), 输出停在 `Using week1 loader for ...`, 之后什么都没有。

**第一个结论从症状直接来**: 测试绿、真机哑 → 代码在测试覆盖的路径上是对的, 问题藏在只有真机才走的东西里。真机和 FakeTokenizer 的差别就是那个真的流式解码器 `BPEStreamingDetokenizer`。嫌疑先落在它的用法上。

## Bug 一: `last_segment` 读取即消耗

### 取证

给 `TokenizerWrapper.detokenizer` 这个 property 打猴子补丁, 把每次 `add_token` 之后的 `last_segment` 记进日志:

```python
orig_prop = TokenizerWrapper.detokenizer
def patched(self):
    d = orig_prop.__get__(self)
    orig_add = d.add_token
    def add(t):
        orig_add(t)
        log.append(("add", t, repr(d.last_segment)))
    d.add_token = add
    return d
TokenizerWrapper.detokenizer = property(patched)
```

再在 `contextlib.redirect_stdout` 里跑 simple_generate, 拿到决定性证据:

```
('add', 151667, "'<think>'")   ← token 加了, 段也有货
('add', 198, "'\\n'")
...
captured: ''                    ← 但 print 出来的东西是空的
```

段在 add 之后确实存在, 到 print 就没了。中间只隔一句 `if detokenize.last_segment:`。

### 机制

tokenizer_utils.py:53 的源码:

```python
def last_segment(self):
    text = self.text
    segment = text[self.offset:]
    self.offset = len(text)   # ← 每次访问都推进书签
    return segment
```

`last_segment` 的语义是"自上次访问以来新长出的文本", 实现方式就是**每次访问都把 offset 推进到当前末尾**——它是个有副作用的 getter。于是原代码:

```python
if detokenize.last_segment:                 # 第 1 次读: '<think>', offset 推进
    print(detokenize.last_segment, end="")  # 第 2 次读: ''
```

if 吃掉了内容, print 拿到空串。模型一直在好好说话, 每个字都被第二次读吞掉。

### 修复

读一次存变量, if 和 print 共用:

```python
segment = detokenize.last_segment
if segment:
    print(segment, end="")
```

**通用教训**: 有副作用的 getter 不能按无副作用属性来用。判断标准不是名字, 是实现里有没有状态推进。mlx_lm 自己的 generate 也是先 `segment = detokenizer.last_segment` 存下来再用。

## Bug 二: 掩码 dtype 跟错了侧 (乱码之谜)

第一个修完后输出变成 `Aşurançaurança凝凝凝`。比无声更糟, 但信息量更大: 模型在算, 算出来是垃圾。

### 变量分离

写对照脚本, 同一个模型、同一段 prompt, 分别跑两条路径:

- 手工逐步循环 (`mx.concatenate([tokens, mx.array([id])])` 拼) → **正经英文** `<think>\nOkay, the`
- simple_generate (`mx.concatenate([tokens, nxt])` 拼) → **乱码**

两条路径几乎逐行相同, 唯一可见差异在拼接方式。打印 dtype:

```
argmax dtype: mlx.core.uint32      ← argmax 的输出类型
A (concat argmax): final dtype int64,  乱码
B (concat int32):  final dtype int32,  正常
```

真相: `nxt` 是 argmax 的 uint32 输出, int32 tokens 拼上它升成 int64。**同一批 token id, dtype 不同, 一个对一个错**——问题不在值, 在值的类型。

### 逐层二分: 零件全对, 接缝有毒

光知道 int64 会坏不够, 要知道坏在哪一层。把前向拆开, 每段同时跑 int32/int64 输入、比最大绝对差:

```
embed_tokens   diff = 0.0
layer 0..27    diff = 0.0   ← 28 个 block 全部无害
final norm     diff = 0.0
lm_head        diff = 0.0
整机 model()   diff = 22.4  ← 唯独整机有毒
```

零件全对、整机坏, 矛盾只能出在零件之间的**接缝**。重读 `Qwen3ModelWeek1.__call__`:

```python
h = self.embed_tokens(inputs)
mask = causal_mask(h.shape[1], h.shape[1], inputs.dtype)   # ← 接缝在这
```

我的二分给每个 block 传的是 `mask="causal"` 字符串 (每层自己造掩码, dtype 跟分数走), `__call__` 造的是数组掩码, **dtype 跟 token 索引走**。

### 完整因果链

1. `mx.argmax` 产出 uint32
2. tokens 拼接后升成 int64
3. `inputs.dtype` → 掩码被铸成 int64
4. `causal_mask` 尾部 `mx.where(allowed, 0.0, float("-inf")).astype(dtype)`——整数类型里没有 -inf, 负无穷被压成 INT64_MIN (约 -9.2e18)
5. 这个值加进 bf16 的 scores, softmax 出垃圾分布

**为什么测试没抓到**: 测试喂的掩码要么是 `"causal"` 字符串、要么是 float 数组。int64 掩码这条路径只有"真模型 + 真生成循环"的组合才踩得到——单元测试给不了这个组合。

### 修复与原则

掩码是要**加进分数的浮点量** (0 和 -inf), dtype 应跟计算侧走:

```python
mask = causal_mask(h.shape[1], h.shape[1], h.dtype)   # h 是嵌入输出, bf16
```

归属规则一句话: **一个量跟谁参与运算, dtype 就跟谁走**。token 索引是查表用的整数, 掩码是打分用的浮点, 两条线不该串。(生成循环里 tokens 被 uint32 污染升 int64 本身也是脆弱点, 但掩码修复后 int64 输入已能正确计算——这是两个独立的脆弱点, 先修了要命的那个。)

## 排查路数 (可复用)

**套路一: 测试绿但真机坏 → 差异即线索。** 先问"哪条路径只有真机才走", 答案直接指向嫌疑模块 (真 detokenizer、真 dtype 链)。单元测试的绿是"覆盖到的路径绿", 覆盖不到的路径它一无所知。

**套路二: 零件全对整机坏 → 查接缝。** 把调用链拆段, 每段双跑 (好输入 vs 坏输入) 比差值, 全零的段排除, 有毒的段缩到接缝。读代码只能证明"看起来对", 跑差值能直接指认"哪里错"。

**取证工具箱** (这轮实际用过的):
- 猴子补丁 property 记录真实调用时的值 ( Bug 一的关键证据)
- `contextlib.redirect_stdout` 捕获函数真实产出
- `mx.eval(...)` 后比 `max(abs(a - b))` 做逐段等价性检查
- 对照实验固定其余变量, 只留一个嫌疑 (dtype / stream / prompt)

## 时间线备注

排查在深夜完成, 修复以 commit cfadc51 落盘。学习者 (仓 owner) 未参与过程——本文档即补课材料: 每个症状、每层证据、每步判断都在上面, 建议从"现场还原"开始自己重走一遍 (把两处修复手工改回原样, 从测试全绿但真机无声的起点出发)。
