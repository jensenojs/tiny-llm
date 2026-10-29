# 测试绿与真机静默: 两条验收路径

`pdm run test --week 1 --day 6` 七条通过，不等于 `simple_generate` 可用。两条路径看到的不是同一件事。

测试路径用替身：`FakeTokenizer`、`FakeDetokenizer`、`ScriptedModel` 把环境换成可控对象，断言的是函数与替身之间的交互契约。真机路径用真货：transformers tokenizer、`TokenizerWrapper`、流式 detokenizer、4bit 权重。只在真货一侧成立的契约，测试没有能力覆盖。

## 三条对照命令

同一份权重、同一个 prompt、同一组参数，只换 `--solution`：

```bash
pdm run main --solution tiny_llm     --loader week1 --model qwen3-0.6b --max-tokens 8
pdm run main --solution tiny_llm_ref --loader week1 --model qwen3-0.6b --max-tokens 8
pdm run main --solution mlx          --loader week1 --model qwen3-0.6b --max-tokens 8
```

实测 stdout 字节数与最后一行：`tiny_llm` 77 字节，最后一行是 `Using week1 loader for ...`，生成文本 0 字；`tiny_llm_ref` 104 字节、`mlx` 57 字节，两者最后一行都是 `A large language model (LLM)`。

## head -5 造成的两次误判

`| head -5` 数行不数字节。banner、进度条、loader 提示占满前五行，生成文本落在第五行之后被切掉，于是 ref 看起来“也没有输出”。判“有没有输出”要取字节数，或把 stderr 分离：

```bash
pdm run main --solution tiny_llm_ref ... 2>/dev/null | wc -c
```

## ref 分支要先编扩展

`--solution tiny_llm_ref` 直接报：

```
ImportError: cannot import name '_ext' from 'extensions_ref.tiny_llm_ext_ref'
```

`tiny_llm_ref/__init__.py` 导入时就带 week2 的编译内核，扩展没编，import 阶段就死。先跑 `pdm run build-ext-ref`（`pyproject.toml:31-32`，`working_dir src/extensions_ref`）。

## 对照的产出只有一位

三种实现的对照结果只有两种取值：有输出 / 没输出。它回答的是上游（权重、tokenizer、模板、sampler）健康与否，不指向函数内部任何一行。这条信息与“测试全绿”重复，两者都不含方向。

修好之后对照才有第二种用途：靶子。同参数下输出应与 mlx 逐 token 一致，差异出现的位置就是两份实现的分歧点。
