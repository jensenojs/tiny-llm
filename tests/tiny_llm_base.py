# from tiny_llm import * — 测试经星号导入拿到实现符号。
# 这是 API 契约: 测试里直接调用的名字 (softmax, linear, SimpleMultiHeadAttention,
# ...) 必须从 src/tiny_llm/__init__.py 导出, 签名必须与测试期望一致。
# Pyright 会对星号导入报 "may be undefined", 这是测试风格的固有代价, 不是 bug。
from tiny_llm import *
