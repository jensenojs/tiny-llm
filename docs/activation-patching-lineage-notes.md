# 激活替换的来历与边界: 从因果中介分析到机制可解释性

> 姊妹篇 [fact-retrieval-interventions-notes.md](fact-retrieval-interventions-notes.md) 在 Qwen3 0.6B 上做了两组实验: 切掩码, 以及激活替换 (activation patching). 那组实验起于一个疑问: attention 和 MLP 的计算步骤几乎相同, 都是点积打分、非线性、加权求和, 两者的分工从哪里来. 本篇回答另外两个问题: 激活替换这套方法从哪里来; 姊妹篇的做法与文献里的常见做法差在哪里, 已知的边界在哪里. 研究这类问题的方向叫机制可解释性 (mechanistic interpretability), 它要把网络的某个行为归到具体的部件和通路上, 而不只看输出的指标. 下文转述文献时只写定性的主张, 不搬运其中的具体数字; 引用的条目已对照 arXiv 题录核对过作者、会议与摘要 (2026-11-03), 细节以原文为准. 文中「实验」均指姊妹篇的实验, 读数取自那里 (mlx_lm 加载的 `Qwen/Qwen3-0.6B-MLX-4bit`), 逐条对照放在「对照表」一节.

## 为什么需要干预

最直接的内部观测是 probing: 在中间层向量上训练一个线性分类器, 看能否读出某个属性, 例如「这里是一位运动员」. probing 读出某个属性, 只能证明这份信息存在于这个向量里; 它是谁写进来的、后面有没有被用到, probing 回答不了. 残差流是同宽的向量, 每个子层的输出都加在同一条向量上, 所以同一份信息可能是第 2 层写的, 也可能是第 15 层加工出来的. 要区分来源, 需要干预: 让一个候选来源的值换掉或消失, 看输出是否随之改变.

## 中介分析: 干预的统计学来源

这种干预做法来自因果推断. Pearl 的框架提供了两样工具. do 算子把一个变量强行设为某个值, 并切断它对上游变量的依赖. 中介分析处理「处理 → 中介 → 结果」这样的链路: 固定中介变量的取值, 就能把「经过中介」与「绕过中介」两部分效应分开.

把 transformer 放进这个框架, 输入句子是处理, 某层某位置的激活是中介, 下一个词的概率是结果. Vig et al. 2020 (NeurIPS) 较早把中介分析用到 transformer 上, 研究 GPT-2 里性别偏见经过哪些神经元和注意力头. 它的操作方式就是后来 patching 的雏形: 跑两个输入, 把其中一个输入的某个中间激活换成另一个输入的, 看输出随哪一个变化. 这里被干预的是激活, 权重不改. 姊妹篇的实验也是这样做的.

## causal tracing 与 ROME

Meng, Bau, Andonian, Belinkov 2022 (NeurIPS, "Locating and Editing Factual Associations in GPT") 把这套做法用来定位事实. 他们的方法叫 causal tracing: 先给事实句里主语 (subject) 的词嵌入加上高斯噪声, 让模型答不出正确答案; 再逐个 (层, 位置) 把干净输入的激活放回这次带噪声的前向, 看哪一处放回后答案恢复得最多. 他们报告, 起决定作用的是主语 token 上中间层的 MLP, 并据此在这些层的 MLP 权重上做秩一修改, 改写事实 (ROME). 后来的文献把这种「在被破坏的前向里放回干净激活」的方向叫 denoising.

姊妹篇的主扫描与它方向相同: 跑对照句 (Maria), 把干净句 (Michael) 的 MLP 输出放回去, 也是 denoising. 「必要性」一步方向相反: 跑干净句, 放进对照句的值, 叫 noising. 两者真正的差别在另外三处:

- 破坏方式. ROME 给主语嵌入加高斯噪声; 姊妹篇把名字换成另一个真实的名字. 后者叫 symmetric token replacement (STR), 见「批评与边界」一节.
- 放回的粒度. 姊妹篇每次只换一层一个位置的 MLP 输出; ROME 的 MLP 实验一次放回连续一段层 (sliding window).
- 结论的位置. Phelps 句补回最多的一格在主语最后一个 token (Phelps) 上, 这一点与 ROME 一致; 层号是 28 层里的第 2 层, 比 ROME 报告的中间层靠前. 模型不同、粒度不同、例句只有两组, 这个差别由哪一项造成, 现有数据分不出来.

## FFN 作为记忆: Geva 等人的两篇

「知识存在 MLP 里」有直接对应的文献. Geva, Schuster, Berant, Levy 2021 (EMNLP, "Transformer Feed-Forward Layers Are Key-Value Memories") 的经验发现是: FFN 第一层的每一行像一个 key, 与训练语料里的特定文本模式相关; 第二层的每一列像一个 value, 它在词表上诱导出一个分布, 在对应的 key 被激活时写入输出. 写成 Qwen3 的 SwiGLU 形式, key 大致对应 up_proj 与 gate_proj 的行, value 对应 down_proj 的列; 这个对应是本文的类推, 原文研究的是两层的传统 FFN.

Geva, Caciularu, Wang, Goldberg 2022 (EMNLP, "Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space") 更进一步: 把每层 FFN 的输出看作对「词表上的分布」的一次加性更新, 再把这次更新拆成单个参数向量的贡献, 投到词表空间上看; 许多这样的子更新会推高一组人能看懂的相关词.

姊妹篇的「投影到词表」借用的是同一种读法: 把一格 MLP 输出的差值乘上 final norm 的缩放向量, 与嵌入表的每一行做点积. 两者有一处不同: Geva 2022 投的是一层 FFN 的更新本身, 姊妹篇投的是两句话在同一格上 MLP 输出的差. 读数是: 第 2 层 Phelps 格的差值投影后 swimming 排第 14727, 第 6 层 Phelps 格排第 5344, 第 8 层 Phelps 格排第 97161, 第 22 层 of 格排第 75. 最靠后的第 22 层排名最靠前, 这与「逐层把预测推向词表空间」的图景相符; 第 8 层比第 2、6 层都排得更后, 所以只凭这几格不能说排名随层号单调变化. 还要注意, Patchscopes (Ghandeharioun et al. 2024, ICML) 指出, 直接投到词表的读法在早期层上常常读不出东西. 所以第 2 层投影排名靠后, 不能当作「这一格里没有游泳信息」的证据.

## 残差流与叠加

两组实验能做成, 依赖一种看待网络的方式. Elhage et al. 2021 (Anthropic, "A Mathematical Framework for Transformer Circuits") 把残差流看作所有层共用的一条通道: 每个子层从中读取、再把输出加回去; attention 读的是别的位置, 写回当前位置; MLP 读写都在当前位置. 每个子层的输入和输出都是形状确定的张量, 所以才能在 attention 上改掩码, 在 MLP 的输出加回残差流之前把它换掉.

子层内部怎样存放信息, 由 Elhage et al. 2022 (Anthropic, "Toy Models of Superposition") 补充: 要表示的特征多于维度时, 模型用彼此近似正交的方向叠加存放这些特征, 单个神经元会对多个特征都有响应 (多义性). 实验里有三个读数与叠加的图景相容: Phelps 句 168 个单格的补回之和是 173.8%, Jordan 句是 249.4%, 都超过 100%, 说明同一份差别分散在多格上、彼此有重叠; 没有哪一格的差值投影指向 swimming; 几格一起替换的效果不等于各自效果之和. 这些读数与叠加相容, 却不能反过来证明叠加, 因为它们也可以由后续层的非线性单独造成.

## 批评与边界

文献里已知的问题有三类, 每一类在实验读数里都有对应.

### 结论随实验设置变化

Zhang & Nanda 2024 (ICLR, "Towards Best Practices of Activation Patching in Language Models") 系统比较了 patching 的几个设置: 用什么方式破坏输入、用什么指标读数、一次放回一层还是一段层. 他们发现, 这些设置一变, 定位结论可能明显不同. 他们给出两条建议. 破坏方式优先用 STR, 即把关键 token 换成同类的另一个 token, 例如把 "The Eiffel Tower is in" 换成 "The Colosseum is in"; 这样破坏后的输入仍在训练分布之内, 高斯噪声做不到这一点. 读数指标不要用概率, 改用 logit difference, 因为概率可能检测不出起反作用的组件.

实验里的 Michael → Maria 正是 STR. 读数用的是目标词的概率, 属于他们建议避免的那一种; 单格表里仍然出现了负数格, 但概率指标漏掉了多少起反作用的部件, 没有用 logit difference 重扫来核对.

Hase, Bansal, Kim, Ghandeharioun 2023 (NeurIPS, "Does Localization Inform Editing?") 质疑的是定位结论的用途. 他们发现, causal tracing 认定的「事实所在层」并不能预测改哪一层的权重最容易改写这个事实; 预测编辑效果最好的是层号本身. 所以「补回最多的一格」只说明这一格在这次前向里承载了差别, 不说明知识只存在这里, 也不说明改这里的权重最有效.

### 替换产生的是训练分布之外的前向

替换之后的前向, 底子是对照句的残差流, 某几格却是干净句的 MLP 输出, 这种组合在正常输入下不会出现. 所以读数描述的是被干预之后的模型行为. 实验里的两个现象都来自这一点: Phelps 句只换第 2、8 两层的 Phelps 格, swimming 的概率是 0.451, 比干净句本身的 0.287 还高; 第 2 层 Phelps 格与几个负数格一起换, 效果比两者之和低 21 到 52 个百分点. 一种缓解办法是只沿真实的计算路径做替换, 即 path patching: Wang et al. 2022 在间接宾语识别 (IOI) 电路的研究里使用了它, Goldowsky-Dill et al. 2023 ("Localizing Model Behavior with Path Patching") 把它整理成可以定量检验的形式. 实验没有做 path patching.

### 逐格全扫的成本与顺序依赖

姊妹篇的「累加」一步按单格补回从高到低逐格加入. 这种贪心做法的结果依赖加入的顺序: Phelps 句的曲线在 117.9%、152.3%、120.2%、141.6% 之间来回摆动. 全扫 168 格要做 168 次前向, 换长句或大模型时成本随格数线性增长.

系统化的替代是自动电路发现: Conmy et al. 2023 (NeurIPS, ACDC) 用搜索代替单一顺序的贪心, 逐个判断部件是否必要. 更便宜的近似是 attribution patching: 用一次反向传播得到的梯度, 线性估计每一格替换后的效果, 只需两次前向和一次反向; Syed, Rager, Conmy 2023 (NeurIPS ATTRIB Workshop) 报告它在电路恢复任务上优于此前的自动方法.

姊妹篇对两句都跑了 attribution patching: 在对照句的前向上对每层 MLP 输出求目标词概率的梯度, 乘以「干净句 MLP 输出 − 对照句 MLP 输出」, 在 1024 维上求和, 得到每一格的线性估计. 估计与 168 格实测的相关系数, Phelps 句是 0.543, Jordan 句是 0.698. Jordan 句估计最大的两格正是实测最大的两格; Phelps 句实测补回 90.6% 的第 2 层 Phelps 格, 线性估计只有 6.5%, 连前三都没有进. 所以在这两例上, attribution patching 能把亮格大致筛出来, 却可能漏掉最关键的一格. 代码与读数见姊妹篇「用一次反向传播估计全部格」一节.

## 对照表

| 文献 | 主张 (定性) | 实验读数 | 判定 |
|---|---|---|---|
| Vig et al. 2020 | 中介分析可以落实为激活替换 | 实验装置就是激活替换 | 方法同源 |
| Geva et al. 2021 | FFN 像一张 key-value 表, 由输入模式触发 | 只换名字、不改权重, swimming 0.287 → 0.073; 第 2 层 Phelps 格单独补回 90.6% | 相容 |
| Geva et al. 2022 | FFN 的输出把一组相关词推向词表空间 | 第 22 层 of 格投影排 75, 第 2 层 Phelps 格排 14727, 第 8 层 Phelps 格排 97161 | 后层更接近词表, 但不单调 |
| Elhage et al. 2021 | 残差流是各层共用的通道, 子层读写它 | 改掩码、换 MLP 输出都能按形状接上; 168 格全换补回 96.0% | 相容 |
| Elhage et al. 2022 (叠加) | 特征叠加存放, 神经元多义 | 单格之和 173.8% / 249.4%; 无一格投影指向 swimming; 组合效果不可加 | 相容, 但不能反证叠加 |
| Meng et al. 2022 (ROME) | 事实由主语 token 上中间层的 MLP 决定 | Phelps 句补回最多的一格在主语最后一个 token 上, 但在第 2 层; Jordan 句最亮的一格在 Michael 位置第 1 层 | 部分一致 |
| Zhang & Nanda 2024 | 结论随破坏方式与指标变化; 建议 STR 与 logit difference | 实验用了 STR, 指标是概率; 同一实验换一份实现 (bf16) 后 Jordan 句最亮的格子换了位置 | 未按 logit difference 重扫 |
| Hase et al. 2023 | 定位结果不能预测最佳编辑层 | 实验没有做权重编辑 | 未测 |
| attribution patching | 一次反向线性估计全部格 | 与实测相关 0.543 / 0.698; Phelps 关键格估计 6.5% vs 实测 90.6% | 只能粗筛 |

## 与常见做法的差别

- 粒度: 只替换了 MLP 输出 (28 × 6 格); attention 的输出、单个注意力头都没有替换.
- 方向: 主扫描是 denoising (跑对照句、放回干净值), 全部 168 格都扫了; noising 方向 (跑干净句、放进对照值) 只对补回最高的 1、3、10 格做了.
- 读数: 用目标词概率, 没有用 logit difference.
- 规模: 0.6B 参数、28 层、6 个 token 的短句, 两组句子的全部实验在 Apple Silicon 上约 7 秒跑完. 换长句或大模型时, attribution patching 这类近似会变得必要.
- 词表投影: 这个模型的出口复用嵌入表 (tie_word_embeddings), 所以把差值投到词表, 同时也是在比较它与各个词嵌入的相似度, 解释投影结果时要记住这一点.
- 所有实验都在一次 prefill 里完成, 没有经过 KV cache 的逐步解码.

## 参考文献

以下条目的作者、题目与会议已对照 arXiv 题录核对 (2026-11-03).

- Vig, Gehrmann, Belinkov, Qian, Nevo, Sakenis, Huang, Singer, Shieber, 2020. Investigating Gender Bias in Language Models Using Causal Mediation Analysis. NeurIPS. arXiv 2004.12265 (arXiv 版题为 "Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias").
- Meng, Bau, Andonian, Belinkov, 2022. Locating and Editing Factual Associations in GPT. NeurIPS. arXiv 2202.05262.
- Geva, Schuster, Berant, Levy, 2021. Transformer Feed-Forward Layers Are Key-Value Memories. EMNLP. arXiv 2012.14913.
- Geva, Caciularu, Wang, Goldberg, 2022. Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space. EMNLP. arXiv 2203.14680.
- Elhage et al., 2021. A Mathematical Framework for Transformer Circuits. Anthropic, 发布于 transformer-circuits.pub, 不在 arXiv 上 (题录未核对).
- Elhage, Hume, Olsson, Schiefer, Henighan et al., 2022. Toy Models of Superposition. arXiv 2209.10652.
- Wang, Variengien, Conmy, Shlegeris, Steinhardt, 2022. Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small. arXiv 2211.00593.
- Goldowsky-Dill, MacLeod, Sato, Arora, 2023. Localizing Model Behavior with Path Patching. arXiv 2304.05969.
- Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso, 2023. Towards Automated Circuit Discovery for Mechanistic Interpretability. NeurIPS. arXiv 2304.14997.
- Syed, Rager, Conmy, 2023. Attribution Patching Outperforms Automated Circuit Discovery. NeurIPS ATTRIB Workshop. arXiv 2310.10348.
- Zhang, Nanda, 2024. Towards Best Practices of Activation Patching in Language Models: Metrics and Methods. ICLR. arXiv 2309.16042.
- Hase, Bansal, Kim, Ghandeharioun, 2023. Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models. NeurIPS. arXiv 2301.04213.
- Ghandeharioun, Caciularu, Pearce, Dixon, Geva, 2024. Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models. ICML. arXiv 2401.06102.
