# here put the import lib
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GRU4Rec import GRU4Rec
from models.SASRec import SASRec_seq
from models.Bert4Rec import Bert4Rec
from models.utils import Multi_CrossAttention, MLPAdapter

def hsic(x, y, sigma_x=1.0, sigma_y=1.0):
    """
    Compute the HSIC value between two sets of embeddings
    """
    if x.shape[0] <= 1 or y.shape[0] <= 1:
        return torch.tensor(0.0, dtype=x.dtype, device=x.device)
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    if y.dtype != torch.float32:
        y = y.to(torch.float32)
    
    # Ensure all tensors are the same type as input
    m = x.shape[0]
    dtype = x.dtype
    
    K = torch.exp(-torch.cdist(x, x) / (2 * sigma_x**2))
    L = torch.exp(-torch.cdist(y, y) / (2 * sigma_y**2))
    H = torch.eye(m, dtype=dtype, device=x.device) - (1.0/m) * torch.ones((m,m), dtype=dtype, device=x.device)
    
    KH = torch.mm(K, H)
    LH = torch.mm(L, H)
    hsic_value = torch.trace(torch.mm(KH, LH)) / ((m-1)**2)
    return hsic_value


def manhattan_similarity(A, B, method='global'):

    """
    Similarity based on ManhattanDistance between matrices A and B.
    """
    assert method in ['global', 'rowwise', 'columnwise', 'elementwise'], "Unsupported parameter."

    if A.shape != B.shape:
        raise ValueError("A and B must have same dimension.")
    
    diff_matrix = np.abs(A - B)
    
    if method == 'global':
        total_diff = np.sum(diff_matrix)
    elif method == 'rowwise':
        row_sums = np.sum(diff_matrix, axis=1)
        total_diff = np.mean(row_sums)
    elif method == 'columnwise':
        col_sums = np.sum(diff_matrix, axis=0)
        total_diff = np.mean(col_sums)
    elif method == 'elementwise':
        weights = (np.abs(A) + np.abs(B)) / 2
        weighted_diff = diff_matrix * weights
        total_diff = np.sum(weighted_diff) / (np.sum(weights) + 1e-10)
    else:
        raise ValueError("Unsupported parameter.")
    
    max_possible_dist = np.sum(np.abs(A)) + np.sum(np.abs(B))
    similarity = 1 - (total_diff / (max_possible_dist + 1e-10))
    return max(0, min(1, similarity))


class DualLLMGRU4Rec(GRU4Rec):

    def __init__(self, user_num, item_num, device, args):
        
        super().__init__(user_num, item_num, device, args)

        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.use_cross_att = args.use_cross_att
        self.adapter_type = getattr(args, 'adapter_type', 'cross_att')

        # load llm embedding as item embedding
        llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "itm_emb_np.pkl"), "rb"))
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
        self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))    
        self.llm_item_emb.weight.requires_grad = True   # the grad is false in default
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        id_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
        self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))    
        self.id_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.id_item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)
        
        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        if self.use_cross_att:
            if self.adapter_type == 'mlp':
                self.llm2id = MLPAdapter(args.hidden_size, args.dropout_rate)
                self.id2llm = MLPAdapter(args.hidden_size, args.dropout_rate)
            else:  # default to cross_att
                self.llm2id = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)
                self.id2llm = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)

        if args.freeze: # freeze the llm embedding
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
        self._init_weights()

    
    def _get_embedding(self, log_seqs):

        id_seq_emb = self.id_item_emb(log_seqs)
        llm_seq_emb = self.llm_item_emb(log_seqs)
        llm_seq_emb = self.adapter(llm_seq_emb)

        item_seq_emb = torch.cat([id_seq_emb, llm_seq_emb], dim=-1)

        return item_seq_emb


    def log2feats(self, log_seqs):

        id_seqs = self.id_item_emb(log_seqs)
        llm_seqs = self.llm_item_emb(log_seqs)
        llm_seqs = self.adapter(llm_seqs)

        if self.use_cross_att:
            cross_id_seqs = self.llm2id(llm_seqs, id_seqs, log_seqs)
            cross_llm_seqs = self.id2llm(id_seqs, llm_seqs, log_seqs)
        else:
            cross_id_seqs = id_seqs
            cross_llm_seqs = llm_seqs

        id_log_feats = self.backbone(cross_id_seqs, log_seqs)
        llm_log_feats = self.backbone(cross_llm_seqs, log_seqs)

        log_feats = torch.cat([id_log_feats, llm_log_feats], dim=-1)

        return log_feats
    


class DualLLMSASRec(SASRec_seq):

    def __init__(self, user_num, item_num, device, args):
        
        super().__init__(user_num, item_num, device, args)

        # self.user_num = user_num
        # self.item_num = item_num
        # self.dev = device
        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.use_cross_att = args.use_cross_att
        self.adapter_type = getattr(args, 'adapter_type', 'cross_att')

        # load llm embedding as item embedding
        # llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset, "pca_itm_emb_np.pkl"), "rb"))
        llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "itm_emb_np.pkl"), "rb"))
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
        self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))    
        self.llm_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.adapter = nn.Linear(llm_item_emb.shape[1], args.hidden_size)
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        id_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
        self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))    
        self.id_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.id_item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)
        
        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        if self.use_cross_att:
            if self.adapter_type == 'mlp':
                self.llm2id = MLPAdapter(args.hidden_size, args.dropout_rate)
                self.id2llm = MLPAdapter(args.hidden_size, args.dropout_rate)
            else:  # default to cross_att
                self.llm2id = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)
                self.id2llm = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)

        if args.freeze: # freeze the llm embedding
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
        self._init_weights()

    
    def _get_embedding(self, log_seqs):

        id_seq_emb = self.id_item_emb(log_seqs)
        llm_seq_emb = self.llm_item_emb(log_seqs)
        llm_seq_emb = self.adapter(llm_seq_emb)

        item_seq_emb = torch.cat([id_seq_emb, llm_seq_emb], dim=-1)
        return item_seq_emb              


    def log2feats(self, log_seqs, positions):

        id_seqs = self.id_item_emb(log_seqs)
        id_seqs *= self.id_item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        id_seqs += self.pos_emb(positions.long())
        id_seqs = self.emb_dropout(id_seqs)

        llm_seqs = self.llm_item_emb(log_seqs)
        llm_seqs = self.adapter(llm_seqs)
        llm_seqs *= self.id_item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        llm_seqs += self.pos_emb(positions.long())
        llm_seqs = self.emb_dropout(llm_seqs)

        if self.use_cross_att:
            cross_id_seqs = self.llm2id(llm_seqs, id_seqs, log_seqs)
            cross_llm_seqs = self.id2llm(id_seqs, llm_seqs, log_seqs)
            cross_id_seqs = 1 * cross_id_seqs + 0 * id_seqs
            cross_llm_seqs = 1 * cross_llm_seqs + 0 * llm_seqs
        else:
            cross_id_seqs = id_seqs
            cross_llm_seqs = llm_seqs

        id_log_feats = self.backbone(cross_id_seqs, log_seqs)
        llm_log_feats = self.backbone(cross_llm_seqs, log_seqs)

        log_feats = torch.cat([id_log_feats, llm_log_feats], dim=-1)

        return log_feats
    


class DualLLMBert4Rec(Bert4Rec):

    def __init__(self, user_num, item_num, device, args):
        
        super().__init__(user_num, item_num, device, args)

        # self.user_num = user_num
        # self.item_num = item_num
        # self.dev = device
        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.use_cross_att = args.use_cross_att
        self.adapter_type = getattr(args, 'adapter_type', 'cross_att')

        # load llm embedding as item embedding
        # llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset, "pca_itm_emb_np.pkl"), "rb"))
        llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "itm_emb_np.pkl"), "rb"))
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
        self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))    
        self.llm_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.adapter = nn.Linear(llm_item_emb.shape[1], args.hidden_size)
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        id_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
        self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))
        self.id_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.id_item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)
        
        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        if self.use_cross_att:
            if self.adapter_type == 'mlp':
                self.llm2id = MLPAdapter(args.hidden_size, args.dropout_rate)
                self.id2llm = MLPAdapter(args.hidden_size, args.dropout_rate)
            else:  # default to cross_att
                self.llm2id = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)
                self.id2llm = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)

        if args.freeze: # freeze the llm embedding
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
        self._init_weights()

    
    def _get_embedding(self, log_seqs):

        id_seq_emb = self.id_item_emb(log_seqs)
        llm_seq_emb = self.llm_item_emb(log_seqs)
        llm_seq_emb = self.adapter(llm_seq_emb)

        item_seq_emb = torch.cat([id_seq_emb, llm_seq_emb], dim=-1)

        return item_seq_emb


    def log2feats(self, log_seqs, positions):

        id_seqs = self.id_item_emb(log_seqs)
        id_seqs *= self.id_item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        id_seqs += self.pos_emb(positions.long())
        id_seqs = self.emb_dropout(id_seqs)

        llm_seqs = self.llm_item_emb(log_seqs)
        llm_seqs = self.adapter(llm_seqs)
        llm_seqs *= self.id_item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        llm_seqs += self.pos_emb(positions.long())
        llm_seqs = self.emb_dropout(llm_seqs)

        if self.use_cross_att:
            cross_id_seqs = self.llm2id(llm_seqs, id_seqs, log_seqs)
            cross_llm_seqs = self.id2llm(id_seqs, llm_seqs, log_seqs)
        else:
            cross_id_seqs = id_seqs
            cross_llm_seqs = llm_seqs

        id_log_feats = self.backbone(cross_id_seqs, log_seqs)
        llm_log_feats = self.backbone(cross_llm_seqs, log_seqs)

        log_feats = torch.cat([id_log_feats, llm_log_feats], dim=-1)

        return log_feats

# class DualColMod(SASRec_seq):
#     def __init__(self, user_num, item_num, device, args):
#         super().__init__(user_num, item_num, device, args)

#         self.mask_token = item_num + 1
#         self.num_heads = args.num_heads
#         self.use_cross_att = args.use_cross_att
#         self.hgc_layers = args.hgc_layers if hasattr(args, 'hgc_layers') else 2
#         self.device = device

#         # Load and process frequency data
#         self.cooccurrence_dict = self._load_frequency_data(args.dataset)
        
#         # Rest of the initialization remains the same
#         llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "itm_emb_np.pkl"), "rb"))
#         llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
#         llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
#         self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))    
#         self.llm_item_emb.weight.requires_grad = True

#         id_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
#         id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
#         id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
#         self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))    
#         self.id_item_emb.weight.requires_grad = True

#         self.adapter = nn.Sequential(
#             nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
#             nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
#         )
        
#         self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size)
#         self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        
#         if self.use_cross_att:
#             self.llm2id = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)
#             self.id2llm = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)
        
#         self.hgc_dropout = nn.Dropout(p=args.dropout_rate)
        
#         if args.freeze:
#             self.freeze_modules = ["llm_item_emb"]
#             self._freeze()

#         self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
#         self._init_weights()

#         self.enable_id = args.enable_id

#     def _load_frequency_data(self, dataset):
#         """Load frequency data from text file and convert to dictionary"""
#         freq_dict = {}
#         freq_path = os.path.join("data", dataset, "handled", "frequency.txt")
        
#         with open(freq_path, 'r') as f:
#             for line in f:
#                 parts = line.strip().split('\t')
#                 if len(parts) == 3:
#                     item1, item2, freq = int(parts[0]), int(parts[1]), int(parts[2])
#                     if item1 not in freq_dict:
#                         freq_dict[item1] = {}
#                     if item2 not in freq_dict:
#                         freq_dict[item2] = {}
#                     freq_dict[item1][item2] = freq
#                     freq_dict[item2][item1] = freq  # Make it symmetric
        
#         return freq_dict

#     def build_co_occurrence_graph(self, log_seqs):
#         """Build co-occurrence graph from frequency data"""
#         batch_size, seq_len = log_seqs.shape
#         adj = torch.zeros((batch_size, seq_len, seq_len), device=self.device)
        
#         for i in range(batch_size):
#             seq = log_seqs[i].cpu().numpy()
#             for j in range(seq_len):
#                 item_j = seq[j]
#                 if item_j in self.cooccurrence_dict:
#                     for k in range(seq_len):
#                         if j == k:
#                             continue  # Skip self-connections
#                         item_k = seq[k]
#                         if item_k in self.cooccurrence_dict[item_j]:
#                             adj[i, j, k] = self.cooccurrence_dict[item_j][item_k]
        
#         # Add self-connections
#         # for i in range(batch_size):
#         #     for j in range(seq_len):
#         #         adj[i, j, j] = 1.0  # Add self-connection
        
#         # Normalize adjacency matrix
#         row_sum = adj.sum(dim=-1, keepdim=True)
#         adj = adj / (row_sum + 1e-8)  # Add small epsilon to avoid division by zero
        
#         return adj

#     def build_modality_graph(self, llm_embeddings):
#         """Build modality similarity graph using cosine similarity"""
#         # Compute cosine similarity
#         norm_emb = F.normalize(llm_embeddings, p=2, dim=-1)
#         sim = torch.matmul(norm_emb, norm_emb.transpose(-1, -2))
        
#         # Apply threshold and add self-connections
#         mask = sim > 0.1
#         # mask = sim > 0.0
#         sim = sim * mask.float()
        
#         # Add self-connections
#         # batch_size, seq_len, _ = sim.shape
#         # eye = torch.eye(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
#         # sim = sim + eye
        
#         # Normalize
#         row_sum = sim.sum(dim=-1, keepdim=True)
#         sim = sim / (row_sum + 1e-8)
        
#         return sim

#     def hypergraph_convolution(self, adjacency, embedding):
#         """Perform hypergraph convolution with residual connections"""
#         embedding = self.hgc_dropout(embedding)
        
#         for _ in range(self.hgc_layers):
#             embedding = torch.bmm(adjacency, embedding) + embedding
            
#         return embedding

#     def log2feats(self, log_seqs, positions):

#         # Get initial embeddings
#         id_seqs = self.id_item_emb(log_seqs)
#         init_id_seqs = id_seqs.clone()
#         id_seqs *= self.id_item_emb.embedding_dim ** 0.5
#         id_seqs += self.pos_emb(positions.long())
#         id_seqs = self.emb_dropout(id_seqs)

#         llm_seqs = self.llm_item_emb(log_seqs)
#         init_llm_seqs = llm_seqs.clone()
#         llm_seqs = self.adapter(llm_seqs)
#         llm_seqs *= self.id_item_emb.embedding_dim ** 0.5
#         llm_seqs += self.pos_emb(positions.long())
#         llm_seqs = self.emb_dropout(llm_seqs)
        
#         # Build graphs
#         co_adj = self.build_co_occurrence_graph(log_seqs)
#         mo_adj = self.build_modality_graph(llm_seqs)
        
#         pure_co_adj = co_adj / (mo_adj + 1e-8) # eliminate modality information in collaborative graph
#         row_sum = pure_co_adj.sum(dim=-1, keepdim=True)
#         pure_co_adj = pure_co_adj / (row_sum + 1e-8)

#         # for i in range(co_adj.shape[0]):
#         #     print(manhattan_similarity(co_adj[i].detach().cpu().numpy(), mo_adj[i].detach().cpu().numpy()))
#         #     print("=================================================")
        
#         # Apply hypergraph convolution
#         with torch.no_grad(): # 这两个hypergraph_convolution会产生inplace operation影响backpropagation的报错。
#             id_seqs = self.hypergraph_convolution(pure_co_adj, id_seqs)
#             llm_seqs = self.hypergraph_convolution(mo_adj, llm_seqs)

#         bce_loss_func = nn.BCEWithLogitsLoss(reduction='sum')
#         # pairwise_align_loss = bce_loss_func(llm_seqs, id_seqs)
#         pairwise_align_loss = 0.0 # 暂且都设置为0
        
#         # Cross attention if enabled
#         if self.use_cross_att:
#             cross_id_seqs = self.llm2id(llm_seqs, id_seqs, log_seqs)
#             cross_llm_seqs = self.id2llm(id_seqs, llm_seqs, log_seqs)
#             cross_id_seqs = 1 * cross_id_seqs + 0 * id_seqs
#             cross_llm_seqs = 1 * cross_llm_seqs + 0 * llm_seqs
#         else:
#             cross_id_seqs = id_seqs
#             cross_llm_seqs = llm_seqs

#         # Process through backbone
#         id_log_feats = self.backbone(cross_id_seqs, log_seqs)
#         llm_log_feats = self.backbone(cross_llm_seqs, log_seqs)

#         if self.enable_id:
#             return pairwise_align_loss, id_log_feats, llm_log_feats
#         else:
#             log_feats = torch.cat([id_log_feats, llm_log_feats], dim=-1)
#             return log_feats        

#     def _get_embedding(self, log_seqs):
#         id_seq_emb = self.id_item_emb(log_seqs)
#         llm_seq_emb = self.llm_item_emb(log_seqs)
#         llm_seq_emb = self.adapter(llm_seq_emb)

#         item_seq_emb = torch.cat([id_seq_emb, llm_seq_emb], dim=-1)
#         return item_seq_emb


class DualColMod(SASRec_seq):
    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)

        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.use_cross_att = args.use_cross_att
        self.adapter_type = getattr(args, 'adapter_type', 'cross_att')
        self.hgc_layers = args.hgc_layers if hasattr(args, 'hgc_layers') else 2
        self.device = device

        # Load and process frequency data
        self.cooccurrence_dict = self._load_frequency_data(args.dataset)
        
        # Rest of the initialization remains the same
        llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "itm_emb_np.pkl"), "rb"))
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
        self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))    
        self.llm_item_emb.weight.requires_grad = True

        id_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
        self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))    
        self.id_item_emb.weight.requires_grad = True

        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )
        
        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        
        if self.use_cross_att:
            if self.adapter_type == 'mlp':
                self.llm2id = MLPAdapter(args.hidden_size, args.dropout_rate)
                self.id2llm = MLPAdapter(args.hidden_size, args.dropout_rate)
            else:  # default to cross_att
                self.llm2id = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)
                self.id2llm = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)
        
        self.hgc_dropout = nn.Dropout(p=args.dropout_rate)
        
        if args.freeze:
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
        self._init_weights()

        self.enable_id = args.enable_id

    def _load_frequency_data(self, dataset):
        """Load frequency data from text file and convert to dictionary"""
        freq_dict = {}
        freq_path = os.path.join("data", dataset, "handled", "frequency.txt")
        
        with open(freq_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    item1, item2, freq = int(parts[0]), int(parts[1]), int(parts[2])
                    if item1 not in freq_dict:
                        freq_dict[item1] = {}
                    if item2 not in freq_dict:
                        freq_dict[item2] = {}
                    freq_dict[item1][item2] = freq
                    freq_dict[item2][item1] = freq  # Make it symmetric
        
        return freq_dict

    def build_co_occurrence_graph(self, log_seqs):
        """Build co-occurrence graph from frequency data"""
        batch_size, seq_len = log_seqs.shape
        adj = torch.zeros((batch_size, seq_len, seq_len), device=self.device)
        
        for i in range(batch_size):
            seq = log_seqs[i].cpu().numpy()
            for j in range(seq_len):
                item_j = seq[j]
                if item_j in self.cooccurrence_dict:
                    for k in range(seq_len):
                        if j == k:
                            continue  # Skip self-connections
                        item_k = seq[k]
                        if item_k in self.cooccurrence_dict[item_j]:
                            adj[i, j, k] = self.cooccurrence_dict[item_j][item_k]
        
        # Add self-connections
        # for i in range(batch_size):
        #     for j in range(seq_len):
        #         adj[i, j, j] = 1.0  # Add self-connection
        
        # Normalize adjacency matrix
        row_sum = adj.sum(dim=-1, keepdim=True)
        adj = adj / (row_sum + 1e-8)  # Add small epsilon to avoid division by zero
        
        return adj

    def build_modality_graph(self, llm_embeddings):
        """Build modality similarity graph using cosine similarity"""
        # Compute cosine similarity
        norm_emb = F.normalize(llm_embeddings, p=2, dim=-1)
        sim = torch.matmul(norm_emb, norm_emb.transpose(-1, -2))
        
        # Apply threshold and add self-connections
        mask = sim > 0.7
        # mask = sim > 0.0
        # sim = sim * mask.float()
        sim = mask
        
        # Add self-connections
        # batch_size, seq_len, _ = sim.shape
        # eye = torch.eye(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        # sim = sim + eye
        
        # Normalize
        row_sum = sim.sum(dim=-1, keepdim=True)
        sim = sim / (row_sum + 1e-8)
        
        return sim

    def hypergraph_convolution(self, adjacency, embedding):
        """Perform hypergraph convolution with residual connections"""
        embedding = self.hgc_dropout(embedding)
        
        for _ in range(self.hgc_layers):
            embedding = torch.bmm(adjacency, embedding) + embedding
            
        return embedding

    def log2feats(self, log_seqs, positions):

        # Get initial embeddings
        id_seqs = self.id_item_emb(log_seqs)
        init_id_seqs = id_seqs.clone()
        id_seqs *= self.id_item_emb.embedding_dim ** 0.5
        id_seqs += self.pos_emb(positions.long())
        id_seqs = self.emb_dropout(id_seqs)

        llm_seqs = self.llm_item_emb(log_seqs)
        init_llm_seqs = llm_seqs.clone()
        llm_seqs = self.adapter(llm_seqs)
        llm_seqs *= self.id_item_emb.embedding_dim ** 0.5
        llm_seqs += self.pos_emb(positions.long())
        llm_seqs = self.emb_dropout(llm_seqs)
        
        # Build graphs
        co_adj = self.build_co_occurrence_graph(log_seqs)
        mo_adj = self.build_modality_graph(llm_seqs)
        
        # pure_co_adj = co_adj / (mo_adj + 1e-8) # eliminate modality information in collaborative graph
        # row_sum = pure_co_adj.sum(dim=-1, keepdim=True)
        # pure_co_adj = pure_co_adj / (row_sum + 1e-8)
        pure_co_adj = co_adj

        # mo_adj = mo_adj * co_adj # enhance modality graph with intersection set
        # mo_adj = torch.max(mo_adj, co_adj) # enhance modality graph with union set
        # mo_adj = mo_adj + co_adj
        mo_adj = torch.min(mo_adj, co_adj)
        

        # for i in range(co_adj.shape[0]):
        #     print(manhattan_similarity(co_adj[i].detach().cpu().numpy(), mo_adj[i].detach().cpu().numpy()))
        #     print("=================================================")
        
        # Apply hypergraph convolution
        with torch.no_grad(): # 这两个hypergraph_convolution会产生inplace operation影响backpropagation的报错。
            id_seqs = self.hypergraph_convolution(pure_co_adj, id_seqs)
            llm_seqs = self.hypergraph_convolution(mo_adj, llm_seqs)

        bce_loss_func = nn.BCEWithLogitsLoss(reduction='sum')
        # pairwise_align_loss = bce_loss_func(llm_seqs, id_seqs)
        pairwise_align_loss = 0.0 # 暂且都设置为0
        
        # Cross attention if enabled
        if self.use_cross_att:
            cross_id_seqs = self.llm2id(llm_seqs, id_seqs, log_seqs)
            cross_llm_seqs = self.id2llm(id_seqs, llm_seqs, log_seqs)
            cross_id_seqs = 1 * cross_id_seqs + 0 * id_seqs
            cross_llm_seqs = 1 * cross_llm_seqs + 0 * llm_seqs
        else:
            cross_id_seqs = id_seqs
            cross_llm_seqs = llm_seqs

        # Process through backbone
        id_log_feats = self.backbone(cross_id_seqs, log_seqs)
        llm_log_feats = self.backbone(cross_llm_seqs, log_seqs)

        # llm_log_feats = torch.zeros_like(llm_log_feats, device=llm_log_feats.device)
        id_log_feats = torch.zeros_like(id_log_feats, device=id_log_feats.device)

        if self.enable_id:
            return pairwise_align_loss, id_log_feats, llm_log_feats
        else:
            log_feats = torch.cat([id_log_feats, llm_log_feats], dim=-1)
            return log_feats        

    def _get_embedding(self, log_seqs):
        id_seq_emb = self.id_item_emb(log_seqs)
        llm_seq_emb = self.llm_item_emb(log_seqs)
        llm_seq_emb = self.adapter(llm_seq_emb)

        # llm_seq_emb = torch.zeros_like(llm_seq_emb, device=llm_seq_emb.device)
        id_seq_emb = torch.zeros_like(id_seq_emb, device=id_seq_emb.device)

        item_seq_emb = torch.cat([id_seq_emb, llm_seq_emb], dim=-1)
        return item_seq_emb