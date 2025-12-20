#!/bin/bash
#SBATCH --job-name=simple_test
#SBATCH --output=stdout_%j.log
#SBATCH --error=stderr_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:02:00

echo "=== Bash输出开始 ==="
echo "这个bash输出应该可见"
python simple_test.py
echo "=== Bash输出结束 ==="
echo "检查文件生成情况:"
ls -la *.log 2>/dev/null || echo "没有找到log文件"