#!/bin/bash

# data/preprocess.sh - 数据预处理脚本（需在data目录下直接运行）
# 使用方法：
# 1. cd data/
# 2. ./preprocess.sh

current_dir=$(pwd)
dataset="musical"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate llmesr

echo "=============================================="
echo "数据预处理启动 - 当前目录: $current_dir"
echo "正在处理数据集: $dataset"
echo "=============================================="

# 验证当前目录
if [[ ! -f "preprocess.sh" ]]; then
    echo "错误：必须在data目录下运行本脚本"
    echo "请执行: cd data/ && ./preprocess.sh"
    exit 1
fi

# 步骤1: 初始数据处理
echo "[1/9] 执行data_process.py"
python data_process.py --dataset "$dataset" || exit 1

# 步骤2: 转换交互数据
echo "[2/9] 转换inter.txt"
python convert_inter.py --dataset "$dataset" || exit 1

# 步骤3: 计算交互频率
echo "[3/9] 计算inter_freq"
python inter_freq.py --dataset "$dataset" || exit 1

# 步骤4: 生成物品嵌入
echo "[4/9] 生成物品嵌入"
(cd "$dataset" && python get_item_embedding_bert.py) || exit 1

# 步骤5: PCA降维
echo "[5/9] 执行PCA降维"
python pca.py --dataset "$dataset" || exit 1

# 步骤6: 用户嵌入(基础版)
echo "[6/9] 生成基础用户嵌入"
(cd "$dataset" && python get_user_embedding_bert.py) || exit 1

# 步骤7: 用户嵌入(带ID版)
echo "[7/9] 生成带ID用户嵌入"
(cd "$dataset" && python get_user_embedding_bert.py --enable_id) || exit 1

# 步骤8: 检索用户(基础版)
echo "[8/9] 检索基础相似用户"
python retrieval_users.py --dataset "$dataset" || exit 1

# 步骤9: 检索用户(带ID版)
echo "[9/9] 检索带ID相似用户"
python retrieval_users.py --dataset "$dataset" --enable_id || exit 1

echo "=============================================="
echo "预处理完成！"
echo "生成文件保存在: $current_dir/$dataset/handled/"
echo "=============================================="