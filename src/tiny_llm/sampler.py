import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float | None, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)

        """
输入 logprobs 形状 (1, vocab), 第 0 维是 batch, 最后一维是词表; 值是对数概率.
下面所有 axis=-1 都指词表这一维: 不管前面有几维 batch, -1 总指向最后一维.
例子统一用这个玩具输入, 四个词条概率 0.1 / 0.4 / 0.2 / 0.3:

.venv/bin/python debug_repl.py
import copy
x = mx.log(mx.array([[0.1, 0.4, 0.2, 0.3]]))
print(f"x       : {x.tolist()}  shape={x.shape}")

拷贝: 下面的 cp[:, idx] = -inf 是原地修改. b = x 只是起别名, 改 b 等于改 x;
copy.copy 得到独立对象. mlx 没有 mx.copy (0.32.0 实测 hasattr 为 False).

b = x
c = copy.copy(x)
c[:, 0] = -mx.inf
print(f"c is x  : {c is x}   b is x: {b is x}")
print(f"x 未被改: {x.tolist()}")

"""
        cp = copy.copy(logprobs)

        if top_k is not None and top_k > 0:
            """
top-k: 只留概率最大的 k 个.
argpartition 返回下标, 只保证第 kth 小的元素落在位置 kth, 左边都比它小, 右边都比它大,
两边内部不排序. 传 -cp 是因为它从小往大分, 而要的是 cp 最大的那几个.
kth 从 0 数. kth=k-1 时位置 k-1 放第 k 小的, 连它在内左边正好 k 个. kth=k 时左边 k 个
同样是最小的 k 个, 也分得对, 只是 top_k 等于词表长度时 kth 越界, argpartition 报 ValueError
(测试 full_vocabulary_boundaries 与 seeded_sampling 都给 4 个词条传 top_k=4), 所以写 k-1.
结果是 [[左 k 个 | 其余]], 外面还有 batch 一层,
所以切片写在最后一维 [:, k:], 写成 [k:] 切的是只有 1 行的 batch 维.
argsort 会把整行排好; argpartition 只分两组, 词表 15 万条时省掉不必要的排序.

idx = mx.argpartition(-x, kth=1, axis=-1)
print(f"argpartition(kth=1): {idx.tolist()}   左 [1, 3] 是最大的两个 (0.4, 0.3)")
print(f"argpartition(kth=2): {mx.argpartition(-x, kth=2, axis=-1).tolist()}   左两个同样是 id 1 和 3")
try:
    mx.argpartition(-x, kth=4, axis=-1)
except ValueError as e:
    print(f"argpartition(kth=4): {e}")

y = copy.copy(x)
y[:, idx[:, 2:]] = -mx.inf
print(f"top_k=2 之后       : {y.tolist()}")

"""
            index = mx.argpartition(-cp, kth=top_k - 1, axis=-1)[:, top_k:]
            cp[:, index] = -mx.inf

        if top_p is not None and top_p > 0:
            """
top-p: 从大到小累加概率, 刚好越过 top_p 的那一个也留下, 其余丢掉.
一, 排序: argsort(-cp) 得到 order, order[0, j] 是排第 j 的那个 id.
二, 取值: take_along_axis(a, indices) 做 out[行, j] = a[行, indices[行, j]],
   只有一行时等价于 [cp_row[i] for i in order_row]. indices 必须是 mx.array
   (list 报 TypeError), 维数与 a 相同 ((1,) 对 (1, 4) 报 ValueError).
三, 判据: cs - origin 是排在我前面的概率之和 (不含我). 前面还不够 top_p 就留我.
   用 cs < top_p 会丢掉越过阈值的那个; 用 <= 在前面之和恰好等于 top_p 时多留一个.
   位置 0 前面之和恒为 0, 所以最大的那个总被保留, top_p 再小也不会全丢.
四, 对齐: 布尔数组按排序后的顺序排, cp 按原 id 排. 拿它对 cp 做 where 会错位
   (保留 id 0/1, 丢掉 id 3). 所以 where 作用在 sorted_lp 上, 两者同序.
   这个错位能通过 Day7 的全部 10 个测试. top-p 真正起过滤作用、又检查分布的两个测试, 输入概率
   按 id 已从大到小排好 ([0.5, 0.3, 0.15, 0.05] 与 [0.55, 0.25, 0.15, 0.05]), 排序后的顺序就是
   原 id 顺序. seeded_sampling 的输入是升序 [0.05, 0.15, 0.3, 0.5], 错位在那里确实发生,
   丢掉的是概率最大的 id 3, 但这个测试只检查同一种子能否复现, 不看分布.
exp 只用于累加, 结果存进 origin; sorted_lp 本身仍是对数概率.

p = 0.5
order = mx.argsort(-x, axis=-1)
sorted_lp = mx.take_along_axis(x, order, axis=-1)
origin = mx.exp(sorted_lp)
cs = mx.cumsum(origin, axis=-1)
print(f"order        : {order.tolist()}")
print(f"概率 (排序后): {origin.tolist()}")
print(f"累计和       : {cs.tolist()}")
print(f"前面之和     : {(cs - origin).tolist()}")
print(f"cs - o < p   : {(cs - origin < p).tolist()}   要的结果")
print(f"cs < p       : {(cs < p).tolist()}  丢掉了越过阈值的 0.3")
print(f"错位的 where : {mx.where(cs - origin < p, x, -mx.inf).tolist()}  id 3 被丢")

"""
            order = mx.argsort(-cp, axis=-1)
            sorted_lp = mx.take_along_axis(cp, order, axis=-1)

            origin = mx.exp(sorted_lp)
            cs = mx.cumsum(origin, axis=-1)
            sorted_lp = mx.where(cs - origin < top_p, sorted_lp, -mx.inf)

            """
温度与抽样: sorted_lp 仍是对数概率. 除以 T 把词条间的对数差值按同一倍数缩放,
概率比 r 变成 r^(1/T): 0.4 : 0.3 约 1.33 倍, T=0.5 时 1.78 倍, T=2 时 1.15 倍.
直接除概率没用, 归一化时 T 被约掉.
categorical 接收对数值, 按 softmax(输入) 抽一个位置, 返回 (1,) uint32; 不必先归一化.
抽到的是排序后的位置, 还要到 order 里查 id: pos 是 (1,), 比 order 少一维,
[:, None] 补成 (1, 1) 才满足 take_along_axis; 查出来的 (1, 1) 再用 [:, 0] 变回 (1,),
与贪心分支 argmax 的形状一致, generate 里 concatenate 才接得上.

T = 1.0
masked = mx.where(cs - origin < p, sorted_lp, -mx.inf)
pos = mx.random.categorical(masked / T, axis=-1)
tok = mx.take_along_axis(order, pos[:, None], axis=-1)[:, 0]
print(f"pos={pos.tolist()} shape={pos.shape}  ->  id={tok.tolist()} dtype={tok.dtype}")
ids = [mx.take_along_axis(order, mx.random.categorical(masked, axis=-1)[:, None], axis=-1)[:, 0].item() for _ in range(1000)]
print(f"1000 次频率 : {[ids.count(i) / 1000 for i in range(4)]}   约 0 / 0.57 / 0 / 0.43")

"""
            pos = mx.random.categorical(sorted_lp / temp, axis=-1)
            return mx.take_along_axis(order, pos[:, None], axis=-1)[:, 0]

        # 没有 top_p 时 cp 仍按原 id 排, 抽出来的位置就是 id, 不需要查 order
        return mx.random.categorical(cp / temp, axis=-1)

    return sample
