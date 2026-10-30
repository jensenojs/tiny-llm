import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .trace import rule, stage
from .qwen3_week1 import Qwen3ModelWeek1
from .qwen3_week2 import Qwen3ModelWeek2
from typing import Callable


def _release_kv_cache(kv_cache):
    if kv_cache is None:
        return
    for layer in kv_cache:
        layer.release()


def simple_generate(
    model: Qwen3ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
    max_tokens: int = 256,
) -> None:
    # ── Day 6 全景: 生成 = 一个圈 ─────────────────────────────────────
    #
    #   prompt 字符串 (main.py 已用 chat template 穿好格式)
    #     → tokenizer.encode                    ← 只做一次, 文本 → id 数组
    #     → ┌─ model(整句) → (1,S,vocab) logits
    #       │    取最后位置 [:, -1, :]          ← 只有最后一个位置在"预测下一个"
    #       │    log-sum-exp 归一化 → log 概率
    #       │    sampler(log_probs) → 新 id     ← sampler=None 时退化为 argmax(贪心)
    #       └─ 把新 id 拼回句尾                  ← 圈: 新 id 是下一步输入的一部分
    #          每个新 id → detokenizer → 屏幕
    #
    #   走出圈的两个条件: sampler 挑中的新 id == EOS (词表里的特殊 id,
    #   Qwen3 是 <|im_end|>=151645, 和正经 token 平起平坐, 模型训练时学过
    #   "话说完就输出它"), 或新生成 token 数达到 max_tokens (保险丝, 防复读)。
    #   prompt 里本来就有的 EOS 是上下文, 不触发退出。
    #
    # ── 到手的三个对象各管一段 ────────────────────────────────────────
    #
    #   model      你 Week 1 写的: id 数组 → 每个 token 对词表的打分 (logits)
    #   sampler    决策策略: 从打分挑出一个 id。模型给分布, sampler 做决定
    #              (main.py 默认 temp=0 → 里面就是 argmax; Day 7 自己实现花样版)
    #   tokenizer  TokenizerWrapper, 双面壳:
    #              .encode/.eos_token_id 经 __getattr__ 转发给 HF tokenizer (文本 ↔ id)
    #              .detokenizer 是 id → 文本的流式解码器 (见下)
    #
    # ── 流式 detokenizer: 为什么要"流" ────────────────────────────────
    #
    #   词表 = 固定表: 第 i 行是一段字节序列, id 就是行号, 模型输出的每个 id
    #   都是指向某行的指针, id ↔ token 严格一对一。暧昧不在对应关系上, 在
    #   行的内容: 一行可能是 E4 BD 这种不构成完整字符的字节片段。
    #
    #   一个 token 不一定是完整的字 (例: "你" 的 UTF-8 是 3 字节 E4 BD A0,
    #   可能被切成 E4 BD + A0 两个 token)。逐 id decode 直接打, 屏幕上全是碎片。
    #   Qwen3 实际用的是 BPEStreamingDetokenizer (load 时按 tokenizer.json 的
    #   decoder 类型选定): 每个 id 查 tokenmap 得到字符串, 攒进 _unflushed,
    #   还原成字节后能解出完整字符才并入 text; 只补了半个字时 text 不变,
    #   last_segment 为空, 继续攒。它不调 HF 的 decode, 也不重解码。
    #   "每次对攒过的全部 token 整体 decode" 是 NaiveStreamingDetokenizer 的做法,
    #   词表不是 BPE/SPM 时才落到它。
    #   它有状态, 由此三个纪律:
    #     1. tokenizer.detokenizer 这个 @property 每次访问都造全新对象
    #        (tokenizer_utils.py:450) → 循环外 bind 一次, 全程复用同一个
    #     2. 每步 add_token(id) 后 last_segment 有货就 print
    #     3. 退出前 finalize(), 把最后攒着没凑齐的尾巴冲出来, 不冲就丢
    #
    # ── Week 1 版的浪费 (Week 2 用 KV cache 收账) ─────────────────────
    #
    #   每一步 _step 都拿"prompt + 已生成的全部 id"整句重跑模型:
    #     _step([1..6])   prefill, L=S=6
    #     _step([1..6,7]) decode,  L=1, S=7   ← 只新增 1 个 token, 却重付整句的钱
    #     _step([1..6,7,8]) ...
    #   Day 3 的 L/S 对齐规则 (j <= i + (S-L)) 保证 decode 步掩码形同虚设。

    # ── 站 2: id 数组 → logits ──────────────────────────────────────
    # 入口是 id 数组, 出口是 (1, vocab) 的打分. 中间是整台模型.
    #
    # 两个量在入口处盯死:
    #   dtype 必须和入口一致 —— id 是行号, 全程都该是同一个整数类型.
    #         具体几位由入口决定, 不在这里规定.
    #   shape 的最后一维必须等于当前序列长度 (第一步 33, 之后 34, 35 …).
    #   (加 batch 维后是 (1, S); 模型只认带 batch 维的输入)
    _base = [0]
    _calls = [0]
    _expect = {}
    _screen = [0]

    def _step(model, y):
        _calls[0] += 1
        y = y[None, :]  # (S, ) -> (1, S), 模型约定吃 batch 维
        _want_len = _base[0] + _calls[0] - 1
        _what = "prefill" if _calls[0] == 1 else "decode"
        rule(f"forward {_calls[0]}  {_what}")
        # 判据是"不变": 入口是什么类型, 后面每一步都该还是它.
        # 不写死 int32 —— 模型并不声明输入类型, int32 只是 mx.array(列表) 的默认落点,
        # 行号数组真正要守的是"全程同一个整数类型".
        _dtype = "✓" if y.dtype == _expect["dtype"] else f"✗ want={_expect['dtype']}"
        _len = "✓" if y.shape[-1] == _want_len else f"✗ want={_want_len}"
        stage(
            3,
            "model",
            f"{_what:<8} in={tuple(y.shape)}  {y.dtype}  dtype {_dtype}  len {_len}",
        )

        logits = model(y)  # (1, S) -> (1, S, vocab) # 每个位置为词表打分
        logits = logits[:, -1, :]  # only keep the last one, wastes!

        # 出口是 logits: 形状 (1, vocab), 每个词表位置一个原始打分.
        #   它是分数不是概率: 有正有负, 总和不为 1, 大小只用于比较先后.
        # 三个量一起读, 比单看 top1 有用:
        #   top3  该接 'A' 时应该是 32; 语义判据, 会骗人, 只作旁证.
        #   gap   top1 减 top2 的原始分差. 正常是 2 以上; 塌到 0.1 量级,
        #         说明分布被抹平, 后面必然出怪 token.
        #   H     softmax 之后的熵 (nats). 正常个位数以下; 接近词表上限
        #         (ln 151936 ≈ 11.9) 就是均匀分布, 等于模型什么都没在选.
        #   gap 与 H 是形状判据, 不需要任何参照就能判"这东西坏了".
        _idxs = mx.argsort(logits[0])[-3:][::-1]
        _vals = logits[0][_idxs]
        _p = mx.softmax(logits[0], -1)
        _H = -(_p * mx.log(_p + 1e-9)).sum().item()
        _gap = float(_vals[0] - _vals[1])
        stage(4, "logits", f"top3={_idxs.tolist()}  gap={_gap:.2f}")
        stage(
            5,
            "softmax",
            f"p={[round(float(v), 3) for v in _p[_idxs]]}  sum={float(_p.sum()):.3f}  H={_H:.2f}",
        )
        # ── log-sum-exp 归一化: 把"打分"变成"对数概率" ────────────────
        #
        #   logits 只是分数, 不是概率 (可为任意实数, 不和为 1)。
        #   目标: log(p_i) = logits_i - log(Σ exp(logits_j))  —— exp 后总和恰为 1,
        #   这就是 softmax 取对数。Day 7 的 sampler 要求输入是 log 概率, 现在把账做正规。
        #
        #   直接按公式算会在两个方向爆炸: logits 很大 → exp 溢出成 inf;
        #   logits 很小(负得多) → exp 全变 0, log(0) = -inf。
        #   技巧: 先减去最大值 m —— 被减过的分数里最大的那个 exp 后恰为 1,
        #   不会再溢出; 而"每个分数都减同一个常数"不改变谁的 logit 高谁低,
        #   归一化结果在数学上完全等价 (softmax 的平移不变性)。
        #
        #   逐步拆解 (v = logits - m, 最大的 v 是 0):
        #     mx.exp(logits - m)            → 每个 v 取 exp, 全部落在 (0, 1]
        #     mx.sum(..., axis=-1)          → 词表上求和 = 配分函数 Z
        #     logits - m - log(Z)           → log(v_i / Z) = log softmax_i
        #   keepdims=True 保住 (1,1) 形状, 广播时逐词表位置对齐相减, 不塌维。
        m = mx.max(logits, axis=-1, keepdims=True)
        logprobs = (
            logits - m - mx.log(mx.sum(mx.exp(logits - m), axis=-1, keepdims=True))
        )
        """
logits → logprobs → 新 id: 减 m 不改排序, exp 后和为 1; temp=0 的 sampler 就是 argmax.

.venv/bin/python debug_repl.py
ids = mx.array(tokenizer.encode(prompt, add_special_tokens=False))[None, :]
lg = model(ids)[:, -1, :]
m = mx.max(lg, axis=-1, keepdims=True)
lp = lg - m - mx.log(mx.sum(mx.exp(lg - m), axis=-1, keepdims=True))
print(f"logits 范围        : [{lg.min().item():.2f}, {lg.max().item():.2f}]  (分数, 可正可负)")
print(f"exp(logprobs) 之和 : {mx.exp(lp).sum().item():.4f}  (概率, 和为 1)")
print(f"argmax(logits)     : {mx.argmax(lg, -1).item()}")
print(f"argmax(logprobs)   : {mx.argmax(lp, -1).item()}  (同上: 平移不改排序)")
print(f"sampler(logprobs)  : {sampler(lp).item()}  (temp=0 → argmax)")
print(f"新 id 的 dtype     : {mx.argmax(lp, -1).dtype}")
"""
        nxt = mx.argmax(logprobs, axis=-1) if sampler is None else sampler(logprobs)
        stage(6, "sampler", f"{'argmax' if sampler is None else 'fn'} → {nxt.item()}")
        return nxt

    # add_special_tokens=False: main.py 已经用 chat template 把特殊标记
    # (<|im_start|> 等) 穿进 prompt 字符串了, encode 再加一遍就是双层壳,
    # 模型会看到重复的格式标记。裸文本入口才需要 True 让分词器补壳。
    # (测试 FakeTokenizer 也验证: True 会多塞一个 6 进来)
    # ── 站 1: 收到的 prompt 字符串 ────────────────────────────────────
    # 这个字符串不在这里产生: main.py:184 调 apply_chat_template 渲染得到.
    # 模板是 tokenizer_config.json 里的 "chat_template" 字段 (一段 Jinja 源码),
    # 渲染在 transformers/utils/chat_template_utils.py:581; 详情见那边的调用链.
    # 这里只负责收下它, 并判"收对了没有".
    #
    # 进来的是 main.py 用 chat template 拼好的整串文本, 长这样:
    #   <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
    #   <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n\n<｜end▁of▁thinking｜>\n\n
    #
    # 这一步打一行:
    #   tail     解回来的最后三个 token. 尾是空 think 块的收尾
    #           '\n\n</think>\n\n' (enable_thinking=False 时模板留的空块),
    #           模型从这里往下接. 这一行是给人看的旁证, 不是判据 ——
    #           模板一变它就变; 硬判据是站 2 的 roundtrip.
    # 打印走 stderr: stdout 是被测对象 (生成的文本), 观测不能混进被测的通道.
    stage(1, "prompt", f"chars={len(prompt)}  tail={prompt[-28:]!r}")
    """
── 站 2: 字符串 → id 数组 ────────────────────────────────────────
encode 做的事是查表: 词表每一行是一段字节序列 (tokenizer.json 里用 GPT-2 的字节→字符映射写成字符串), id 就是行号.
所以 id 数组 = 这串文本在词表里的行号序列, 每个整数没有任何数值含义,
它只是"第几行". 行号相邻不代表内容相近.

判据:
  n         长度. 必须非零, 且和模板拼出来的长度对得上 (这里 33).
  roundtrip decode(encode(prompt)) == prompt. 站 2 唯一的结构判据:
            文本 → id → 文本能原样回来, 说明查表没丢字、没多壳.
            对不上时, 先查 add_special_tokens 和模板, 后面全部不用看.

调用这一行时:
mlx_lm/tokenizer_utils.py:457  TokenizerWrapper.__getattr__
  (TokenizerWrapper 自己没有 encode; :465 getattr(self._tokenizer, "encode"), self._tokenizer 是 Qwen2Tokenizer)
  └─ transformers/tokenization_utils_base.py:2254  PreTrainedTokenizerBase.encode
     └─ transformers/tokenization_utils_base.py:2291  self._encode_plus(...)
        └─ transformers/tokenization_utils_tokenizers.py:857  TokenizersBackend._encode_plus
           ├─ transformers/tokenization_utils_tokenizers.py:959  self._tokenizer.encode_batch(...)
           │  这里的 self._tokenizer 是 tokenizers.Tokenizer (Rust), encode_batch 是 method_descriptor, 没有 Python 源码可跳
           └─ transformers/tokenization_utils_tokenizers.py:966  _convert_encoding, 把 Rust 的 Encoding 转回 Python 的 dict

Rust 那一侧的行为在 load 时组装好:
transformers/models/qwen2/tokenization_qwen2.py:61  self._tokenizer = Tokenizer(BPE(...))
  (:73 ByteLevel decoder; :74 NFC normalizer; :75 Regex + ByteLevel pre_tokenizer)
  词表与合并规则随后从 tokenizer.json 载入.

两次 _tokenizer 不是同一个对象: 上面那层是 Qwen2Tokenizer (Python),
最下面那层是 tokenizers.Tokenizer (Rust).

.venv/bin/python debug_repl.py
ids = tokenizer.encode(prompt, add_special_tokens=False)
print(f"id 个数        : {len(ids)}")
print(f"前 6 个 id     : {ids[:6]}")
print(f"roundtrip      : {tokenizer.decode(ids) == prompt}")
print(f"上层 _tokenizer: {type(tokenizer._tokenizer).__name__}")
print(f"底层 _tokenizer: {type(tokenizer._tokenizer._tokenizer)}")
print(f"encode_batch   : {type(type(tokenizer._tokenizer._tokenizer).encode_batch).__name__}  (Rust, 无 Python 源码)")
"""
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    _base[0] = len(tokens)
    # 观测不能要求被测对象多实现一个方法: 假 tokenizer 只有 encode.
    _decode = getattr(tokenizer, "decode", None)
    _rt = (_decode(tokens) == prompt) if _decode else None
    _tail = repr(_decode(tokens[-3:])) if _decode else None
    stage(2, "encode", f"n={len(tokens)}  roundtrip={_rt}  tail={_tail}")
    """
load 时决定用哪个解码器:
mlx_lm/tokenizer_utils.py:605  _is_bpe_decoder(tokenizer.json 的 decoder) → BPEStreamingDetokenizer
mlx_lm/tokenizer_utils.py:451  TokenizerWrapper.detokenizer  每次访问都 new 一个
  └─ mlx_lm/tokenizer_utils.py:165  BPEStreamingDetokenizer.__init__
     (:169-171 vocab 反过来存成 tokenmap[id] = 字符串; :177 make_byte_decoder)
     ├─ transformers/tokenization_utils_tokenizers.py:609  vocab → :606 Rust get_vocab, 151669 条
     │  数据本体: <tok.name_or_path>/tokenizer.json 的 model.vocab (151643) + added_tokens (26)
     └─ mlx_lm/tokenizer_utils.py:229  make_byte_decoder  GPT-2 的 替代字符 → 字节 表

tokenmap 每一行是字符串, 每个字符代表一个字节: 可打印 ASCII 原样, 其余换成替代字符.
  tokenmap[32]   = 'A'
  tokenmap[8162] = 'ĠApple'    Ġ 是空格字节 (_byte_decoder['Ġ'] == 32)

词表的前 256 行是单字节 ('!'=0, ..., 'A'=32), 后面 15 万行是 BPE 训练时把语料里
最常相邻的两段反复合并出来的, 编号按合并先后. 常见词因此整词成行, 而且大小写、
前面有没有空格都算不同的行; 训练后出现的新词不加行, 用已有片段拼:
  'Apple'     [26567]              ['Apple']
  ' Apple'    [8162]               ['ĠApple']
  'apple'     [22377]              ['apple']
  ' ChatGPT'  [12853, 38, 2828]    ['ĠChat', 'G', 'PT']
  '你好'      [108386]             ['ä½łå¥½']    6 个 UTF-8 字节合成 1 行
词表大小训练前就定死, 因为 embed_tokens.weight 的行数和它绑在一起; 最坏退回单字节,
所以任何文本都能编码, 代价是生僻词占的 token 多.

.venv/bin/python debug_repl.py
print(f"{'文本':10}  {'ids':20}  tokenmap 里的字符串")
for w in ['Apple', ' Apple', 'apple', ' ChatGPT', '你好']:
    ids = tokenizer.encode(w, add_special_tokens=False)
    print(f"{w!r:12}  {str(ids):20}  {tokenizer.convert_ids_to_tokens(ids)}")

"""
    detokenize = tokenizer.detokenizer
    detokenize.reset()  # 解码器带状态, 复用前先归零
    tokens = mx.array(tokens)
    # 数组化之后的 dtype 由 MLX 决定 (Python int 列表的默认落点, 现在实测 int32).
    # 记下来当基准: 后面每一步都该还是它. 要守的是"不变", 不是"等于 32 位".
    _expect["dtype"] = tokens.dtype
    stage(2, "encode", f"dtype={tokens.dtype}")

    # prefill: 第一个 token
    nxt = _step(model, tokens)
    tokens = mx.concatenate([tokens, nxt])
    """
拼回句尾: argmax 产出 uint32, 和 int32 的 ids 拼接后类型提升成 int64 (Bug2 的入口).
句尾 dtype 漂移本身不报错, 掩码若借 inputs.dtype 造就会把 -inf 铸成整数.

.venv/bin/python debug_repl.py
ids = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
nxt = mx.argmax(model(ids[None, :])[:, -1, :], axis=-1)
cat = mx.concatenate([ids, nxt])
print(f"ids          : {ids.dtype}  长度 {ids.shape[0]}")
print(f"nxt (argmax) : {nxt.dtype}  值 {nxt.item()}")
print(f"拼接后       : {cat.dtype}  长度 {cat.shape[0]}  (int32 + uint32 → int64)")
print(f"-inf 铸 int64: {mx.array([float('-inf')]).astype(mx.int64).item()}  (屏蔽值变成 0)")
"""

    # decode: prefill 的产物也是新 token, 它的 add_token/判 EOS 在循环第一圈做
    # (每圈处理一个已生成 token 再产下一个, 故圈数 = max_tokens)
    for _ in range(max_tokens):
        if nxt.item() == tokenizer.eos_token_id:  # .item(): (1,)数组拆封成 int
            stage(7, "EOS", f"{nxt.item()} 命中, 退出")
            break
        _ch = repr(_decode([nxt.item()])) if _decode else None
        stage(7, "new id", f"{nxt.item()} = {_ch}")
        """
每生成一个 id:
mlx_lm/tokenizer_utils.py:206  BPEStreamingDetokenizer.add_token
  (:208 tokenmap[token] 查表得字符串; :209 追加进 _unflushed)
  ├─ mlx_lm/tokenizer_utils.py:185  _decode_bytes  逐字符查 _byte_decoder 还原字节, 再按 utf-8 解码
  └─ mlx_lm/tokenizer_utils.py:195  _maybe_trim_space
  (:214 结尾是 \\ufffd 或单个空格时先不定稿; 否则 :217 并入 self.text)
mlx_lm/tokenizer_utils.py:52  StreamingDetokenizer.last_segment  text[self.offset:], 然后水位抬到 len(text)

模型输出的单位是 token, 屏幕显示的单位是字符, 两者边界不一定对齐.
例: 🦙 的 UTF-8 是 4 字节 F0 9F A6 99, 分词器把它拆进了 3 个 token:
  id      tokenmap   代表的字节     _unflushed     last_segment
  11162   'ĠðŁ'      20 F0 9F       'ĠðŁ'          ''        字节不全, 不定稿
  99      '¦'        A6             'ĠðŁ¦'         ''
  247     'Ļ'        99             ''             '🦙'      凑齐, 并入 text
  5394    'Ġok'      20 6F 6B       ''             ' ok'     本身完整, 直接放出
每来一个 id 就单独解码打印的话, 前三步屏幕上是三个 '�', 🦙 永远不出现.
add_token 把字节攒在 _unflushed, 解码结尾仍是 \\ufffd 就不写进 text; text 因此只增不改,
last_segment 才能放心按水位往后切. 开头空格消失是另一件事: 这是整段输出的第一个
字符, :200-201 在 text 为空时去掉前导空格.

.venv/bin/python debug_repl.py
d = tokenizer.detokenizer
print(f"{'id':>6}  {'tokenmap':10}  {'_unflushed':12}  last_segment")
for i in tokenizer.encode(" 🦙 ok", add_special_tokens=False):
    d.add_token(i)
    print(f"{i:>6}  {d.tokenmap[i]!r:10}  {d._unflushed!r:12}  {d.last_segment!r}")

"""
        detokenize.add_token(nxt.item())
        segment = detokenize.last_segment  # 读一次即消费一次, 存下来复用
        stage(
            8, "detokenizer", repr(segment) + ("" if segment else "  (半个字, 继续攒)")
        )
        if segment:
            print(segment, end="", flush=True)
            _screen[0] += len(segment)
        nxt = _step(model, tokens)
        tokens = mx.concatenate([tokens, nxt])

    detokenize.finalize()
    segment = detokenize.last_segment
    stage(8, "detok.final", repr(segment))
    if segment:
        print(segment, end="", flush=True)
        _screen[0] += len(segment)
    stage(9, "screen", f"{_screen[0]} chars")


def simple_generate_with_kv_cache(
    model: Qwen3ModelWeek2,
    tokenizer: TokenizerWrapper,
    prompt: str,
    max_tokens: int = 256,
    use_bounded_kv_capacity: bool | None = None,
) -> str:
    def _step(model, y, offset, kv_cache):
        pass


def speculative_generate(
    draft_model: Qwen3ModelWeek2,
    model: Qwen3ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
    proposal_length: int = 4,
    max_tokens: int = 256,
) -> str:
    pass
