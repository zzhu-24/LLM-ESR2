import os
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import copy
from datetime import datetime
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.utils import unzip_data, random_neq, set_seed
from utils.earlystop import EarlyStopping


def setup_logger(log_dir, experiment):
    """创建同时输出到控制台和文件的 logger"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment}_{timestamp}.log")

    logger = logging.getLogger("problem")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 文件 handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")
    return logger

# ====================== 1. 配置与全局参数 ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 默认数据参数（可通过 --dataset 等参数覆盖）
DEFAULT_DATASET = "beauty"
DEFAULT_INTER_FILE = "inter"
EMBED_DIM = 64  # ID Embedding维度
MODALITY_EMBED_DIM = 768  # 模态Embedding（如LLM）维度
# 保存路径（Step1的Item Embedding、模型权重）
SAVE_DIR = "./Intro_problem"
os.makedirs(SAVE_DIR, exist_ok=True)


def load_real_dataset(dataset, inter_file, aug=True, aug_seq_len=0):
    """
    加载真实数据集，与仓库 generators/generator.py 中的 _load_dataset 保持一致。
    数据格式：./data/{dataset}/handled/{inter_file}.txt，每行 "user_id item_id"
    用户/物品ID从1开始，按时间顺序排列。
    划分策略：train=除最后2条外全部，valid=倒数第2条，test=最后1条
    """
    data_path = os.path.join("./data", dataset, "handled", f"{inter_file}.txt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"数据文件不存在: {data_path}\n"
            "请先运行 data/data_process.py 进行数据预处理，"
            "并将处理后的 inter.txt 放在对应目录。"
        )

    usernum, itemnum = 0, 0
    User = defaultdict(list)
    user_train, user_valid, user_test = {}, {}, {}

    with open(data_path, "r") as f:
        for line in f:
            u, i = line.rstrip().split()
            u, i = int(u), int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

    for user in tqdm(User, desc="划分训练/验证/测试集"):
        nfeedback = len(User[user]) - aug_seq_len
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = [User[user][-2]]
            user_test[user] = [User[user][-1]]

    return user_train, user_valid, user_test, usernum, itemnum


# ====================== 2. 真实数据集（与仓库 SeqDataset 格式一致） ======================
class SeqRecDataset(Dataset):
    """
    序列推荐数据集（真实数据）：与 generators/data.py 中 SeqDataset 逻辑一致。
    训练模式：返回 (seq, pos)，负样本由 in-batch negatives 提供。
    评估模式：返回 (seq, pos, negs)，negs 含 test_neg 个随机负样本用于排名评估。
    """
    def __init__(self, train_data, valid_data, item_num, max_seq_len,
                 for_eval=False, aug_seq_len=0, test_neg=100):
        """
        train_data: {user_id: [item1, item2, ...]} 训练集
        valid_data: {user_id: [item]} 验证集或测试集（用于构造 eval 序列）
        for_eval: 若 True，则构造 eval 数据（序列=train+valid[:-1]，目标=valid[-1]）
        aug_seq_len: 序列展开时的增强长度，与仓库 generator 一致
        test_neg: eval 时每个样本的随机负样本数量
        """
        super().__init__()
        self.item_num = item_num
        self.max_seq_len = max_seq_len
        self.for_eval = for_eval
        self.test_neg = test_neg

        if for_eval:
            self.data = []
            for user in train_data:
                if len(valid_data[user]) > 0:
                    seq = train_data[user] + valid_data[user]
                    self.data.append(seq)
        else:
            self.data = unzip_data(train_data, aug=True, aug_num=aug_seq_len)

    def __len__(self):
        return len(self.data)

    def _make_seq(self, inter):
        """将交互序列（除最后一个item外）左填充到 max_seq_len"""
        seq = np.zeros(self.max_seq_len, dtype=np.int64)
        idx = self.max_seq_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx < 0:
                break
        return seq

    def __getitem__(self, index):
        inter = self.data[index]
        pos = inter[-1]
        seq = self._make_seq(inter)

        if self.for_eval:
            # 评估时：采样 test_neg 个负样本用于排名
            non_neg = set(inter)
            negs = []
            for _ in range(self.test_neg):
                neg = random_neq(1, self.item_num + 1, non_neg)
                negs.append(neg)
            return (
                torch.LongTensor(seq),
                torch.LongTensor([pos]),
                torch.LongTensor(negs),
            )
        else:
            # 训练时：只返回 seq 和 pos，负样本由 in-batch negatives 提供
            return (
                torch.LongTensor(seq),
                torch.LongTensor([pos]),
            )

# ====================== 3. 核心模块：Adapter（支持MLP/归一化） ======================
class Adapter(nn.Module):
    """
    Adapter模块：
    - MLP模式（Step3）：Linear -> ReLU -> Linear
    - Normalize模式（Step4）：Linear -> LayerNorm -> Linear（替换激活为归一化）
    """
    def __init__(self, in_dim, out_dim, adapter_type="mlp"):
        super().__init__()
        self.adapter_type = adapter_type
        # 隐藏层维度（经验值：out_dim*2）
        hidden_dim = out_dim * 2

        # 基础线性层
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        
        # 激活/归一化分支
        if adapter_type == "mlp":
            self.act = nn.ReLU()  # Step3：ReLU激活
        elif adapter_type == "normalize":
            self.norm = nn.LayerNorm(hidden_dim)  # Step4：LayerNorm替代激活
        else:
            raise ValueError(f"Adapter type {adapter_type} not supported!")

    def forward(self, x):
        x = self.fc1(x)
        if self.adapter_type == "mlp":
            x = self.act(x)
        elif self.adapter_type == "normalize":
            x = self.norm(x)
        x = self.fc2(x)
        return x

# ====================== 4. 核心模型：SASRec（兼容ID/模态Embedding） ======================
class SASRec(nn.Module):
    """
    SASRec模型：
    - Step1：仅ID Embedding，无Adapter，仅BCE损失
    - Step2：仅模态Embedding，无Adapter，仅BCE损失
    - Step3/4：模态Embedding + Adapter，BCE + MSE对齐损失
    """
    def __init__(self, item_num, max_seq_len, embed_dim, 
                 id_item_emb=None, modality_emb=None, adapter=None, use_adapter=False):
        super().__init__()
        self.item_num = item_num
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.use_adapter = use_adapter  # 是否启用Adapter（Step3/4=True）

        # 1. ID Embedding（Step1使用）：优先用 pca64 预训练 Embedding 初始化
        if id_item_emb is not None:
            self.item_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(id_item_emb), padding_idx=0)
            self.item_embedding.weight.requires_grad = True  # from_pretrained 默认 freeze，需显式开启
        else:
            self.item_embedding = nn.Embedding(item_num + 2, embed_dim, padding_idx=0)
            nn.init.normal_(self.item_embedding.weight, 0, 0.01)

        # 2. 模态Embedding（Step2/3/4使用，预加载，可微调）
        if modality_emb is not None:
            self.modality_item_emb = nn.Embedding.from_pretrained(torch.FloatTensor(modality_emb))
            self.modality_item_emb.weight.requires_grad = True  # 显式开启梯度，与 DualLLMSRS.py 一致
        else:
            self.modality_item_emb = None

        # 3. Adapter模块（Step3/4使用）
        self.adapter = adapter

        # 4. SASRec Transformer层（固定结构）
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.fc = nn.Linear(embed_dim, embed_dim)

        # 5. 位置编码（序列推荐必备）
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        nn.init.normal_(self.pos_embedding.weight, 0, 0.01)

    def get_item_emb(self, item_ids):
        """获取物品Embedding（分支控制：ID/模态+Adapter）"""
        if self.use_adapter:
            # Step3/4：模态Embedding + Adapter
            emb = self.modality_item_emb(item_ids)
            emb = self.adapter(emb)
        elif self.modality_item_emb is not None:
            # Step2：仅模态Embedding
            emb = self.modality_item_emb(item_ids)
        else:
            # Step1：仅ID Embedding
            emb = self.item_embedding(item_ids)
        return emb

    def forward(self, item_seq, return_attn_weights=False):
        """SASRec前向传播：序列编码得到用户兴趣表征"""
        # 1. 位置编码
        pos_ids = torch.arange(self.max_seq_len, device=DEVICE).unsqueeze(0).repeat(item_seq.shape[0], 1)
        pos_emb = self.pos_embedding(pos_ids)

        # 2. 物品Embedding + 位置编码
        item_emb = self.get_item_emb(item_seq)
        seq_emb = item_emb + pos_emb

        # 3. Self-Attention（掩码：仅看历史序列）
        key_padding_mask = (item_seq == 0)  # [B, L]，PAD位置为True
        seq_emb = self.layer_norm(seq_emb)
        attn_output, attn_weights = self.attention(
            seq_emb, seq_emb, seq_emb,
            key_padding_mask=key_padding_mask,
            need_weights=True, average_attn_weights=True  # [B, L, L]
        )
        seq_emb = seq_emb + attn_output  # 残差

        # 4. FFN + 残差
        seq_emb = self.layer_norm(seq_emb)
        ffn_output = self.fc(seq_emb)
        seq_emb = seq_emb + ffn_output

        # 5. 取最后一个有效位置的表征（用户兴趣）
        seq_len = (item_seq != 0).sum(dim=1) - 1  # 最后一个非PAD位置
        seq_len = torch.clamp(seq_len, min=0)  # 避免0长度
        user_interest = seq_emb[torch.arange(item_seq.shape[0]), seq_len]

        if return_attn_weights:
            return user_interest, attn_weights  # attn_weights: [B, L, L]
        return user_interest

    def calculate_score(self, item_seq, item_ids):
        """计算用户对物品的预测得分：user_interest · item_emb"""
        user_interest = self.forward(item_seq)
        item_emb = self.get_item_emb(item_ids.squeeze(1))  # [B, D]
        score = torch.mul(user_interest, item_emb).sum(dim=1)  # [B]
        return score

# ====================== 5. 损失函数 ======================
bce_loss_func = nn.BCEWithLogitsLoss()

def mse_align_loss(pred_emb, target_emb):
    """MSE对齐损失：Adapter输出的Embedding对齐Step1的ID Embedding"""
    return nn.MSELoss()(pred_emb, target_emb)

# ====================== 6. 训练与评估函数 ======================
def train(model, dataloader, optimizer, args, target_item_emb=None):
    """
    通用训练函数（in-batch negatives）：
    - 每个 batch 的正样本互为负样本
    - target_item_emb：Step1保存的Item Embedding（仅Step3/4需要）
    """
    model.train()
    total_loss = 0.0
    for batch_idx, (item_seq, pos_item) in enumerate(dataloader):
        item_seq = item_seq.to(DEVICE)         # [B, L]
        pos_item = pos_item.to(DEVICE).squeeze(1)  # [B]

        optimizer.zero_grad()

        # 1. In-batch negatives BCE 损失
        user_interest = model.forward(item_seq)         # [B, D]
        pos_emb = model.get_item_emb(pos_item)          # [B, D]
        # 计算 batch 内所有 user-item 对的得分矩阵
        all_logits = torch.matmul(user_interest, pos_emb.T)  # [B, B]
        # 标签：对角线为1（正样本），其余为0（in-batch负样本）
        labels = torch.eye(all_logits.shape[0], device=DEVICE)
        loss_rec = bce_loss_func(all_logits, labels)

        # 2. 对齐损失（仅Step3/4）
        loss_align = torch.tensor(0.0, device=DEVICE)
        if args.experiment in ["step3", "step4"] and target_item_emb is not None:
            pred_pos_emb = model.get_item_emb(pos_item)
            target_pos_emb = target_item_emb[pos_item]
            loss_align = mse_align_loss(pred_pos_emb, target_pos_emb)

        # 3. 总损失
        loss_total = loss_rec + args.align_lambda * loss_align

        loss_total.backward()
        optimizer.step()

        total_loss += loss_total.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, test_neg=100):
    """
    排名评估：NDCG@10 和 HR@10
    与 trainers/sequence_trainer.py 的 eval 逻辑一致：
    将 pos + test_neg 个 neg 拼成候选集，计算得分并排名。
    """
    model.eval()
    pred_ranks = []

    with torch.no_grad():
        for item_seq, pos_item, neg_items in dataloader:
            item_seq = item_seq.to(DEVICE)       # [B, L]
            pos_item = pos_item.to(DEVICE)       # [B, 1]
            neg_items = neg_items.to(DEVICE)     # [B, test_neg]

            # 拼接候选：[pos, neg1, neg2, ...] -> [B, 1+test_neg]
            candidates = torch.cat([pos_item, neg_items], dim=1)

            # 用户兴趣表征
            user_interest = model.forward(item_seq)  # [B, D]

            # 候选物品 Embedding
            cand_emb = model.get_item_emb(candidates)  # [B, 1+test_neg, D]

            # 计算得分：[B, 1+test_neg]
            scores = torch.matmul(cand_emb, user_interest.unsqueeze(-1)).squeeze(-1)

            # 取反后排序（得分越高排名越靠前），pos 在 index=0
            ranks = torch.argsort(torch.argsort(-scores, dim=1), dim=1)[:, 0]  # pos 的排名
            pred_ranks.append(ranks.cpu())

    pred_ranks = torch.cat(pred_ranks).numpy()

    # 计算 NDCG@10 和 HR@10
    ndcg_10, hr_10 = 0.0, 0.0
    for rank in pred_ranks:
        if rank < 10:
            ndcg_10 += 1.0 / np.log2(rank + 2)
            hr_10 += 1.0
    n = len(pred_ranks)
    ndcg_10 /= n
    hr_10 /= n

    return {"NDCG@10": ndcg_10, "HR@10": hr_10}

# ====================== 7. Attention 可视化 ======================
def collect_avg_attention(model, dataloader, max_display_len=20):
    """
    收集评估集上的平均 attention weights，并截取有效序列部分。
    只保留序列长度 >= max_display_len 的样本，截取最后 max_display_len 个位置。
    返回平均 attention 矩阵 [max_display_len, max_display_len]。
    """
    import matplotlib
    matplotlib.use("Agg")

    model.eval()
    attn_sum = None
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            item_seq = batch[0].to(DEVICE)  # [B, L]
            _, attn_weights = model.forward(item_seq, return_attn_weights=True)  # [B, L, L]

            # 逐样本处理：只取有效长度 >= max_display_len 的样本
            seq_lens = (item_seq != 0).sum(dim=1)  # [B]
            for i in range(item_seq.shape[0]):
                slen = seq_lens[i].item()
                if slen < max_display_len:
                    continue
                # 左 padding，有效部分在右侧，取最后 max_display_len 个位置
                start = item_seq.shape[1] - max_display_len
                attn_crop = attn_weights[i, start:, start:]  # [D, D]
                attn_np = attn_crop.cpu().numpy()
                if attn_sum is None:
                    attn_sum = np.zeros_like(attn_np)
                attn_sum += attn_np
                count += 1

    if count == 0:
        # 如果没有足够长的序列，降低要求取所有样本
        with torch.no_grad():
            for batch in dataloader:
                item_seq = batch[0].to(DEVICE)
                _, attn_weights = model.forward(item_seq, return_attn_weights=True)
                seq_lens = (item_seq != 0).sum(dim=1)
                for i in range(item_seq.shape[0]):
                    slen = seq_lens[i].item()
                    if slen < 2:
                        continue
                    actual_len = min(int(slen), max_display_len)
                    start = item_seq.shape[1] - actual_len
                    attn_crop = attn_weights[i, start:, start:].cpu().numpy()
                    # 填充到 max_display_len
                    padded = np.zeros((max_display_len, max_display_len))
                    padded[-actual_len:, -actual_len:] = attn_crop
                    if attn_sum is None:
                        attn_sum = np.zeros((max_display_len, max_display_len))
                    attn_sum += padded
                    count += 1

    if count > 0:
        attn_sum /= count

    return attn_sum, count


def plot_attention_heatmap(attn_matrix, save_path, title="Average Attention Weights", display_len=20):
    """绘制 attention 热力图并保存"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(attn_matrix, cmap="YlOrRd", aspect="equal")

    ax.set_xlabel("Key Position", fontsize=12)
    ax.set_ylabel("Query Position", fontsize=12)
    ax.set_title(title, fontsize=14)

    # 设置刻度标签（显示相对位置）
    tick_positions = list(range(0, display_len, max(1, display_len // 10)))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels([str(p + 1) for p in tick_positions])
    ax.set_yticklabels([str(p + 1) for p in tick_positions])

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def collect_modality_cosine(modality_emb, dataloader, max_display_len=20):
    """收集 modality embedding 之间的平均余弦相似度矩阵。"""
    emb_mat = torch.tensor(modality_emb, device=DEVICE, dtype=torch.float32)
    sim_sum = None
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            item_seq = batch[0].to(DEVICE)  # [B, L]
            seq_lens = (item_seq != 0).sum(dim=1)  # [B]
            for i in range(item_seq.shape[0]):
                slen = seq_lens[i].item()
                if slen < max_display_len:
                    continue
                start = item_seq.shape[1] - max_display_len
                ids = item_seq[i, start:]  # [D]
                # 取对应的 modality embedding
                emb = emb_mat[ids]  # [D, dim]
                # L2 归一化
                emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
                sim = torch.matmul(emb, emb.T).cpu().numpy()  # [D, D]
                if sim_sum is None:
                    sim_sum = np.zeros_like(sim)
                sim_sum += sim
                count += 1

    if count == 0:
        # 若没有足够长的序列，则对所有样本按实际长度截取并 padding 到固定尺寸
        with torch.no_grad():
            for batch in dataloader:
                item_seq = batch[0].to(DEVICE)
                seq_lens = (item_seq != 0).sum(dim=1)
                for i in range(item_seq.shape[0]):
                    slen = seq_lens[i].item()
                    if slen < 2:
                        continue
                    actual_len = min(int(slen), max_display_len)
                    start = item_seq.shape[1] - actual_len
                    ids = item_seq[i, start:]  # [actual_len]
                    emb = emb_mat[ids]
                    emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
                    sim = torch.matmul(emb, emb.T).cpu().numpy()  # [actual_len, actual_len]
                    padded = np.zeros((max_display_len, max_display_len))
                    padded[-actual_len:, -actual_len:] = sim
                    if sim_sum is None:
                        sim_sum = np.zeros((max_display_len, max_display_len))
                    sim_sum += padded
                    count += 1

    if count > 0:
        sim_sum /= count

    return sim_sum, count


def visualize_modality_cosine(modality_emb, dataloader, save_dir, experiment, logger, max_display_len=20):
    """训练完成后基于 modality embedding 余弦相似度画热力图。"""
    if modality_emb is None:
        logger.info("No modality embedding provided, skip modality cosine heatmap.")
        return

    logger.info(f"Collecting modality cosine similarity (display_len={max_display_len})...")
    sim_matrix, count = collect_modality_cosine(modality_emb, dataloader, max_display_len=max_display_len)

    if sim_matrix is None or count == 0:
        logger.info("No valid samples for modality cosine visualization.")
        return

    logger.info(f"Modality cosine averaged over {count} samples")
    save_path = os.path.join(save_dir, f"modality_sim_heatmap_{experiment}.png")
    title = f"Modality Cosine Sim ({experiment}, n={count})"
    plot_attention_heatmap(sim_matrix, save_path, title=title, display_len=max_display_len)
    logger.info(f"Modality cosine heatmap saved to {save_path}")


def visualize_attention(model, dataloader, save_dir, experiment, logger, max_display_len=20):
    """训练完成后可视化 attention weights"""
    logger.info(f"Collecting attention weights (display_len={max_display_len})...")
    attn_matrix, count = collect_avg_attention(model, dataloader, max_display_len=max_display_len)

    if attn_matrix is None or count == 0:
        logger.info("No valid samples for attention visualization.")
        return

    logger.info(f"Averaged over {count} samples")
    save_path = os.path.join(save_dir, f"attn_heatmap_{experiment}.png")
    title = f"Avg Attention Weights ({experiment}, n={count})"
    plot_attention_heatmap(attn_matrix, save_path, title=title, display_len=max_display_len)
    logger.info(f"Attention heatmap saved to {save_path}")


def collect_avg_modality_cosine(modality_emb, dataloader, max_display_len=20):
    """
    收集评估集上模态Embedding之间的平均余弦相似度矩阵。
    与 attention 可视化保持相同的截取逻辑与尺寸：[max_display_len, max_display_len]。
    """
    import matplotlib
    matplotlib.use("Agg")

    # 转为张量并做L2归一化，方便用矩阵乘得到余弦相似度
    emb_tensor = torch.FloatTensor(modality_emb).to(DEVICE)  # [N_items+2, D]
    emb_norm = emb_tensor / (emb_tensor.norm(dim=1, keepdim=True) + 1e-8)

    cos_sum = None
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            item_seq = batch[0].to(DEVICE)  # [B, L]
            seq_lens = (item_seq != 0).sum(dim=1)  # [B]

            for i in range(item_seq.shape[0]):
                slen = seq_lens[i].item()
                if slen < max_display_len:
                    continue

                # 与 attention 相同：左 padding，有效在右侧，取最后 max_display_len 个位置
                start = item_seq.shape[1] - max_display_len
                item_ids = item_seq[i, start:]  # [max_display_len]

                emb = emb_norm[item_ids]  # [max_display_len, D]
                cos_mat = torch.matmul(emb, emb.T).cpu().numpy()  # [max_display_len, max_display_len]

                if cos_sum is None:
                    cos_sum = np.zeros_like(cos_mat)
                cos_sum += cos_mat
                count += 1

    # 若没有足够长的序列，则允许短序列并做 padding，保持尺寸不变
    if count == 0:
        with torch.no_grad():
            for batch in dataloader:
                item_seq = batch[0].to(DEVICE)
                seq_lens = (item_seq != 0).sum(dim=1)

                for i in range(item_seq.shape[0]):
                    slen = seq_lens[i].item()
                    if slen < 2:
                        continue

                    actual_len = min(int(slen), max_display_len)
                    start = item_seq.shape[1] - actual_len
                    item_ids = item_seq[i, start:start + actual_len]  # [actual_len]

                    emb = emb_norm[item_ids]  # [actual_len, D]
                    cos_mat = torch.matmul(emb, emb.T).cpu().numpy()  # [actual_len, actual_len]

                    padded = np.zeros((max_display_len, max_display_len))
                    padded[-actual_len:, -actual_len:] = cos_mat

                    if cos_sum is None:
                        cos_sum = np.zeros((max_display_len, max_display_len))
                    cos_sum += padded
                    count += 1

    if count > 0:
        cos_sum /= count

    return cos_sum, count


def visualize_modality_cosine(modality_emb, dataloader, save_dir, experiment, logger, max_display_len=20):
    """可视化模态Embedding之间的平均余弦相似度热力图，尺寸与attention热力图一致。"""
    logger.info(f"Collecting modality cosine similarity (display_len={max_display_len})...")
    cos_matrix, count = collect_avg_modality_cosine(modality_emb, dataloader, max_display_len=max_display_len)

    if cos_matrix is None or count == 0:
        logger.info("No valid samples for modality cosine visualization.")
        return

    logger.info(f"Averaged over {count} samples for modality cosine")
    save_path = os.path.join(save_dir, f"modality_cosine_heatmap_{experiment}.png")
    title = f"Avg Modality Cosine ({experiment}, n={count})"
    plot_attention_heatmap(cos_matrix, save_path, title=title, display_len=max_display_len)
    logger.info(f"Modality cosine heatmap saved to {save_path}")


# ====================== 8. 主函数（参数控制实验分支） ======================
def main(args):
    # 0. 初始化 logger
    log_dir = os.path.join(SAVE_DIR, args.dataset, "logs")
    logger = setup_logger(log_dir, args.experiment)

    logger.info("=" * 60)
    logger.info("Experiment Configuration")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    # 1. 加载真实数据
    user_train, user_valid, user_test, user_num, item_num = load_real_dataset(
        args.dataset, args.inter_file, aug=True, aug_seq_len=args.aug_seq_len
    )
    max_seq_len = args.max_len
    step1_item_emb_path = os.path.join(SAVE_DIR, args.dataset, "step1_item_emb.pkl")
    ckpt_dir = os.path.join(SAVE_DIR, args.dataset, args.experiment)
    os.makedirs(ckpt_dir, exist_ok=True)
    logger.info(f"Dataset: {args.dataset} | Users: {user_num} | Items: {item_num}")

    train_dataset = SeqRecDataset(
        user_train, user_valid, item_num, max_seq_len,
        for_eval=False, aug_seq_len=args.aug_seq_len
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, drop_last=True)

    # 验证集：序列 = train + valid，预测 valid（含 test_neg 个负样本用于排名评估）
    eval_dataset = SeqRecDataset(
        user_train, user_valid, item_num, max_seq_len,
        for_eval=True, test_neg=args.test_neg
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)
    logger.info(f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")

    # 2. 加载 Embedding（与 DualLLMSRS.py 一致的预处理：在 index=0 插入零向量作为 PAD，末尾追加零向量）
    handled_dir = os.path.join("./data", args.dataset, "handled")

    # 2a. 加载 pca64_itm_emb_np.pkl 作为 ID Embedding 初始化
    id_emb_path = os.path.join(handled_dir, "pca64_itm_emb_np.pkl")
    if os.path.exists(id_emb_path):
        id_item_emb = pickle.load(open(id_emb_path, "rb"))
        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
        logger.info(f"Loaded ID Embedding from {id_emb_path}, shape: {id_item_emb.shape}")
    else:
        id_item_emb = None
        logger.info("pca64_itm_emb_np.pkl not found, using random init for ID Embedding")

    # 2b. 加载 itm_emb_np.pkl 作为 Modality Embedding
    modality_emb_path = os.path.join(handled_dir, "itm_emb_np.pkl")
    if os.path.exists(modality_emb_path):
        modality_emb = pickle.load(open(modality_emb_path, "rb"))
        modality_emb = np.insert(modality_emb, 0, values=np.zeros((1, modality_emb.shape[1])), axis=0)
        modality_emb = np.concatenate([modality_emb, np.zeros((1, modality_emb.shape[1]))], axis=0)
        logger.info(f"Loaded Modality Embedding from {modality_emb_path}, shape: {modality_emb.shape}")
    else:
        modality_emb = None
        logger.info("itm_emb_np.pkl not found, modality embedding disabled")

    modality_emb_dim = modality_emb.shape[1] if modality_emb is not None else MODALITY_EMBED_DIM

    # 3. 实验分支逻辑
    if args.experiment == "step1":
        # Step1：ID Embedding + SASRec + 仅BCE损失，保存Item Embedding
        logger.info("===== Running Step1: ID Embedding + SASRec + BCE =====")
        model = SASRec(
            item_num=item_num,
            max_seq_len=max_seq_len,
            embed_dim=EMBED_DIM,
            id_item_emb=id_item_emb,
            modality_emb=None,
            adapter=None,
            use_adapter=False
        ).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        stopper = EarlyStopping(patience=args.patience, verbose=True, path=ckpt_dir)

        for epoch in range(args.num_train_epochs):
            train_loss = train(model, train_dataloader, optimizer, args)
            eval_metrics = evaluate(model, eval_dataloader, test_neg=args.test_neg)
            logger.info(f"Epoch {epoch+1}/{args.num_train_epochs} | Train Loss: {train_loss:.4f} | NDCG@10: {eval_metrics['NDCG@10']:.4f} | HR@10: {eval_metrics['HR@10']:.4f}")
            stopper(eval_metrics['NDCG@10'], epoch, model)
            if stopper.early_stop:
                logger.info(f"Early stopping at epoch {epoch+1}, best epoch: {stopper.best_epoch+1}, best NDCG@10: {stopper.best_score:.4f}")
                break

        logger.info(f"Training finished. Best epoch: {stopper.best_epoch+1}, Best NDCG@10: {stopper.best_score:.4f}")

        # 加载最优模型后保存 Item Embedding
        model.load_state_dict(torch.load(stopper.path))
        item_emb = model.item_embedding.weight.data.cpu().numpy()
        with open(step1_item_emb_path, "wb") as f:
            pickle.dump(item_emb, f)
        logger.info(f"Step1 Item Embedding saved to {step1_item_emb_path}")

        # Attention 可视化
        visualize_attention(model, eval_dataloader, ckpt_dir, args.experiment, logger, max_display_len=args.attn_display_len)

    elif args.experiment == "step2":
        # Step2：模态Embedding + SASRec + 仅BCE损失
        logger.info("===== Running Step2: Modality Embedding + SASRec + BCE =====")
        model = SASRec(
            item_num=item_num,
            max_seq_len=max_seq_len,
            embed_dim=EMBED_DIM,
            id_item_emb=id_item_emb,
            modality_emb=modality_emb,
            adapter=None,
            use_adapter=False
        ).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        stopper = EarlyStopping(patience=args.patience, verbose=True, path=ckpt_dir)

        for epoch in range(args.num_train_epochs):
            train_loss = train(model, train_dataloader, optimizer, args)
            eval_metrics = evaluate(model, eval_dataloader, test_neg=args.test_neg)
            logger.info(f"Epoch {epoch+1}/{args.num_train_epochs} | Train Loss: {train_loss:.4f} | NDCG@10: {eval_metrics['NDCG@10']:.4f} | HR@10: {eval_metrics['HR@10']:.4f}")
            stopper(eval_metrics['NDCG@10'], epoch, model)
            if stopper.early_stop:
                logger.info(f"Early stopping at epoch {epoch+1}, best epoch: {stopper.best_epoch+1}, best NDCG@10: {stopper.best_score:.4f}")
                break

        logger.info(f"Training finished. Best epoch: {stopper.best_epoch+1}, Best NDCG@10: {stopper.best_score:.4f}")

        # Attention 可视化
        model.load_state_dict(torch.load(stopper.path))
        visualize_attention(model, eval_dataloader, ckpt_dir, args.experiment, logger, max_display_len=args.attn_display_len)
        if modality_emb is not None:
            visualize_modality_cosine(modality_emb, eval_dataloader, ckpt_dir, args.experiment, logger, max_display_len=args.attn_display_len)

    elif args.experiment == "step3":
        # Step3：模态Embedding + MLP Adapter + BCE + MSE对齐
        logger.info("===== Running Step3: Modality + MLP Adapter + BCE + MSE Align =====")
        # 加载Step1的Item Embedding
        if not os.path.exists(step1_item_emb_path):
            raise ValueError("Step1 Item Embedding not found! Run Step1 first.")
        with open(step1_item_emb_path, "rb") as f:
            target_item_emb = torch.FloatTensor(pickle.load(f)).to(DEVICE)

        # 初始化MLP Adapter
        adapter = Adapter(
            in_dim=modality_emb_dim,
            out_dim=EMBED_DIM,
            adapter_type="mlp"
        ).to(DEVICE)

        # 初始化模型
        model = SASRec(
            item_num=item_num,
            max_seq_len=max_seq_len,
            embed_dim=EMBED_DIM,
            id_item_emb=id_item_emb,
            modality_emb=modality_emb,
            adapter=adapter,
            use_adapter=True
        ).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        stopper = EarlyStopping(patience=args.patience, verbose=True, path=ckpt_dir)

        # 训练（传入目标Embedding用于对齐）
        for epoch in range(args.num_train_epochs):
            train_loss = train(model, train_dataloader, optimizer, args, target_item_emb)
            eval_metrics = evaluate(model, eval_dataloader, test_neg=args.test_neg)
            logger.info(f"Epoch {epoch+1}/{args.num_train_epochs} | Train Loss: {train_loss:.4f} | NDCG@10: {eval_metrics['NDCG@10']:.4f} | HR@10: {eval_metrics['HR@10']:.4f}")
            stopper(eval_metrics['NDCG@10'], epoch, model)
            if stopper.early_stop:
                logger.info(f"Early stopping at epoch {epoch+1}, best epoch: {stopper.best_epoch+1}, best NDCG@10: {stopper.best_score:.4f}")
                break

        logger.info(f"Training finished. Best epoch: {stopper.best_epoch+1}, Best NDCG@10: {stopper.best_score:.4f}")

        # Attention 可视化
        model.load_state_dict(torch.load(stopper.path))
        visualize_attention(model, eval_dataloader, ckpt_dir, args.experiment, logger, max_display_len=args.attn_display_len)
        if modality_emb is not None:
            visualize_modality_cosine(modality_emb, eval_dataloader, ckpt_dir, args.experiment, logger, max_display_len=args.attn_display_len)

    elif args.experiment == "step4":
        # Step4：模态Embedding + Normalize Adapter + BCE + MSE对齐
        logger.info("===== Running Step4: Modality + Normalize Adapter + BCE + MSE Align =====")
        # 加载Step1的Item Embedding
        if not os.path.exists(step1_item_emb_path):
            raise ValueError("Step1 Item Embedding not found! Run Step1 first.")
        with open(step1_item_emb_path, "rb") as f:
            target_item_emb = torch.FloatTensor(pickle.load(f)).to(DEVICE)

        # 初始化Normalize Adapter
        adapter = Adapter(
            in_dim=modality_emb_dim,
            out_dim=EMBED_DIM,
            adapter_type="normalize"
        ).to(DEVICE)

        # 初始化模型
        model = SASRec(
            item_num=item_num,
            max_seq_len=max_seq_len,
            embed_dim=EMBED_DIM,
            id_item_emb=id_item_emb,
            modality_emb=modality_emb,
            adapter=adapter,
            use_adapter=True
        ).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        stopper = EarlyStopping(patience=args.patience, verbose=True, path=ckpt_dir)

        # 训练（传入目标Embedding用于对齐）
        for epoch in range(args.num_train_epochs):
            train_loss = train(model, train_dataloader, optimizer, args, target_item_emb)
            eval_metrics = evaluate(model, eval_dataloader, test_neg=args.test_neg)
            logger.info(f"Epoch {epoch+1}/{args.num_train_epochs} | Train Loss: {train_loss:.4f} | NDCG@10: {eval_metrics['NDCG@10']:.4f} | HR@10: {eval_metrics['HR@10']:.4f}")
            stopper(eval_metrics['NDCG@10'], epoch, model)
            if stopper.early_stop:
                logger.info(f"Early stopping at epoch {epoch+1}, best epoch: {stopper.best_epoch+1}, best NDCG@10: {stopper.best_score:.4f}")
                break

        logger.info(f"Training finished. Best epoch: {stopper.best_epoch+1}, Best NDCG@10: {stopper.best_score:.4f}")

        # Attention 可视化
        model.load_state_dict(torch.load(stopper.path))
        visualize_attention(model, eval_dataloader, ckpt_dir, args.experiment, logger, max_display_len=args.attn_display_len)
        if modality_emb is not None:
            visualize_modality_cosine(modality_emb, eval_dataloader, ckpt_dir, args.experiment, logger, max_display_len=args.attn_display_len)

    else:
        raise ValueError(f"Experiment {args.experiment} not supported! Choose step1/step2/step3/step4.")

if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser(description="SASRec Experiment (Step1-Step4)")
    parser.add_argument("--experiment", type=str, required=True, 
                        choices=["step1", "step2", "step3", "step4"],
                        help="Specify experiment step: step1/step2/step3/step4")
    # 数据参数（与 train_baseline / generators 一致）
    parser.add_argument("--dataset", type=str, default="beauty",
                        help="数据集名称（需在 ./data/<dataset>/handled/ 下有对应数据）")
    parser.add_argument("--inter_file", type=str, default="inter", help="交互文件名（不含.txt）")
    parser.add_argument("--max_len", type=int, default=200, help="序列最大长度")
    parser.add_argument("--aug_seq_len", type=int, default=0, help="序列展开增强长度")
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--test_neg", type=int, default=100, help="评估时每个样本的随机负样本数")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=20, help="早停耐心值（连续多少轮不提升则停止）")
    parser.add_argument("--align_lambda", type=float, default=1.0, 
                        help="Weight of MSE align loss (only for step3/step4)")
    parser.add_argument("--attn_display_len", type=int, default=20, help="Attention热力图显示的序列长度（10或20）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    set_seed(args.seed)

    # 运行实验
    main(args)