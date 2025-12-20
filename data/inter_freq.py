import os
from collections import defaultdict
from tqdm import tqdm

def calculate_item_frequency(dataset):
    """计算物品共现频率并保存结果"""
    
    # 文件路径
    inter_file = f"./{dataset}/handled/inter.txt"
    output_file = f"./{dataset}/handled/frequency.txt"
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 读取交互数据并构建用户-物品字典
    user_items = defaultdict(list)
    with open(inter_file, 'r') as f:
        for line in tqdm(f, desc="Reading interaction data"):
            user, item = line.strip().split()
            user_items[user].append(item)
    
    # 计算物品共现频率
    cooccurrence = defaultdict(int)
    for items in tqdm(user_items.values(), desc="Calculating co-occurrence"):
        # 获取去重后的物品列表
        unique_items = list(set(items))
        # 计算两两组合
        for i in range(len(unique_items)):
            for j in range(i+1, len(unique_items)):
                item1, item2 = sorted((unique_items[i], unique_items[j]))
                cooccurrence[(item1, item2)] += 1
    
    # 写入结果文件
    with open(output_file, 'w') as f:
        for (item1, item2), count in tqdm(sorted(cooccurrence.items()), desc="Writing results"):
            f.write(f"{item1}\t{item2}\t{count}\n")
    
    print(f"Frequency calculation completed. Results saved to {output_file}")

if __name__ == "__main__":
    calculate_item_frequency(dataset="beauty")