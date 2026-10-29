import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
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
    #   流式解码器不逐 token 解码, 而是"重解码 diff": 每次对攒过的全部 token
    #   整体 decode, 新长出来的文本段 (last_segment) 才放出来; 只补了半个字
    #   时 last_segment 为空, 继续攒。这手法不依赖"前缀稳定"性质
    #   (字节级 BPE 有此性质; SentencePiece 类吞空格词表没有, 手法照样工作)。
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

    def _step(model, y):
        y = y[None, :]  # (S, ) -> (1, S), 模型约定吃 batch 维
        logits = model(y)  # (1, S) -> (1, S, vocab) # 每个位置为词表打分
        logits = logits[:, -1, :]  # only keep the last one
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
        if sampler is None:
            return mx.argmax(logprobs, axis=-1)
        return sampler(logprobs)

    # add_special_tokens=False: main.py 已经用 chat template 把特殊标记
    # (<|im_start|> 等) 穿进 prompt 字符串了, encode 再加一遍就是双层壳,
    # 模型会看到重复的格式标记。裸文本入口才需要 True 让分词器补壳。
    # (测试 FakeTokenizer 也验证: True 会多塞一个 6 进来)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    detokenize = tokenizer.detokenizer
    detokenize.reset()  # why reset?

    tokens = mx.array(tokens)
    # prefill: 第一个 token
    nxt = _step(model, tokens)
    tokens = mx.concatenate([tokens, nxt])

    # decode: prefill 的产物也是新 token, 它的 add_token/判 EOS 在循环第一圈做
    # (每圈处理一个已生成 token 再产下一个, 故圈数 = max_tokens)
    for _ in range(max_tokens):
        if nxt.item() == tokenizer.eos_token_id:  # .item(): (1,)数组拆封成 int
            break
        detokenize.add_token(nxt.item())
        # 探针: 先读一次看拿到什么, 再走业务分支
        import sys

        print(
            f"<nxt={nxt.item():<7} last_segment={detokenize.last_segment!r}>",
            file=sys.stderr,
        )
        if detokenize.last_segment:
            print(detokenize.last_segment, end="")
            print("  [进了分支]", file=sys.stderr)
        nxt = _step(model, tokens)
        tokens = mx.concatenate([tokens, nxt])

    detokenize.finalize()
    if detokenize.last_segment:
        print(detokenize.last_segment, end="")


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
