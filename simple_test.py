#!/usr/bin/env python3
import sys
import os

print("=== 基础测试开始 ===")
print("这行应该出现在stdout中")
print("这行显式输出到stdout", file=sys.stdout)
print("这行输出到stderr", file=sys.stderr)

# 强制刷新缓冲区
sys.stdout.flush()
sys.stderr.flush()

print("当前工作目录:", os.getcwd())
print("Python路径:", sys.executable)
print("=== 基础测试结束 ===")