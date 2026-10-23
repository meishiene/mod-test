"""千问（Qwen 3.6 / 3.7 / 3.8）专项测试套件。

权威实现：``cases.py``（卡片元数据 + 型号矩阵）、``core.py``（用例与判定）、
``runner.py``（装配与收尾）。入口脚本 ``scripts/run_qwen.py`` 只做参数转发。
"""
from .cases import CASES, QWEN_MODELS  # noqa: F401
