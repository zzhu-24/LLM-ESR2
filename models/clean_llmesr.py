import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.clean_backbones import CleanSASRecBackbone
from models.SASRec import SASRecBackbone
from models.utils import Contrastive_Loss2, MLPAdapter, Multi_CrossAttention
from utils.paths import get_dataset_paths


class LLMESRClean(nn.Module):
    """Clean ColMod-style LLM-ESR with explicit dual views and optional graph/adapters."""

    def __init__(self, user_num, item_num, device, args):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = device
        self.hidden_size = args.hidden_size
        self.fusion = args.fusion
        self.alpha = args.alpha
        self.beta = args.beta
        self.collab_llm_ratio = args.collab_llm_ratio
        self.use_align_loss = args.use_align_loss
        self.item_reg = args.item_reg
        self.use_pair_loss = getattr(args, "use_pair_loss", False)
        self.pair_loss_weight = args.pair_loss_weight
        self.user_sim_func = args.user_sim_func
        self.use_cross_att = (
            getattr(args, "use_adapter", False)
            or getattr(args, "use_cross_att", False)
            or getattr(args, "use_cross_attn", False)
        )
        self.adapter_type = getattr(args, "adapter_type", "cross_att")
        self.use_graph = getattr(args, "use_graph", True)
        self.use_co_graph = getattr(args, "use_co_graph", True)
        self.use_modality_graph = getattr(args, "use_modality_graph", True)
        self.hgc_layers = getattr(args, "hgc_layers", 2)
        self.modality_threshold = getattr(args, "modality_threshold", 0.1)
        self.graph_mix = getattr(args, "graph_mix", "intersection")
        self.graph_filter = getattr(args, "graph_filter", "none")
        self.use_intent_gap = getattr(args, "use_intent_gap", False) or getattr(args, "dynamic_align", False)
        self.dynamic_align = getattr(args, "dynamic_align", False)
        self.dynamic_align_scale = getattr(args, "dynamic_align_scale", 1.0)
        self.colmod_compat = getattr(args, "colmod_compat", False)
        self.split_backbone = getattr(args, "split_backbone", False)
        self.enable_id = getattr(args, "enable_id", False)

        if self.fusion not in {"concat", "sum", "gate"}:
            raise ValueError(f"Unsupported fusion: {self.fusion}")
        if self.user_sim_func not in {"cl", "kd"}:
            raise ValueError(f"Unsupported user_sim_func: {self.user_sim_func}")
        if self.graph_mix not in {None, "intersection", "union", "modality", "collab"}:
            raise ValueError(f"Unsupported graph_mix: {self.graph_mix}")
        if self.graph_filter not in {"none", "residual", "positive_residual", "normalized_residual"}:
            raise ValueError(f"Unsupported graph_filter: {self.graph_filter}")

        paths = get_dataset_paths(args.dataset, args.inter_file)
        id_weight = self._load_item_embedding(paths.id_item_emb, item_num)
        llm_weight = self._load_item_embedding(paths.llm_item_emb, item_num)

        self.id_item_emb = nn.Embedding.from_pretrained(id_weight, freeze=False, padding_idx=0)
        self.llm_item_emb = nn.Embedding.from_pretrained(llm_weight, freeze=args.freeze, padding_idx=0)
        self.id_adapter = self._make_projection(id_weight.size(1), args.hidden_size)
        self.llm_adapter = (
            self._make_colmod_projection(llm_weight.size(1), args.hidden_size)
            if self.colmod_compat
            else self._make_projection(llm_weight.size(1), args.hidden_size)
        )
        
        if self.use_intent_gap:
            sem_user_weight = self._load_user_embedding(paths.user_emb, user_num)
            collab_user_weight = self._load_user_embedding(paths.collab_user_emb, user_num)
            self.sem_user_emb = nn.Embedding.from_pretrained(sem_user_weight, freeze=True)
            self.collab_user_emb = nn.Embedding.from_pretrained(collab_user_weight, freeze=True)
            self.sem_user_adapter = self._make_projection(sem_user_weight.size(1), args.hidden_size)
            self.collab_user_adapter = self._make_projection(collab_user_weight.size(1), args.hidden_size)
        else:
            self.sem_user_emb = None
            self.collab_user_emb = None
            self.sem_user_adapter = None
            self.collab_user_adapter = None

        self.pos_emb = nn.Embedding(args.max_len + 100, args.hidden_size)
        self.emb_dropout = nn.Dropout(args.dropout_rate)
        self.hgc_dropout = nn.Dropout(args.dropout_rate)

        if self.use_cross_att:
            self.id_cross_adapter = self._make_view_adapter(args)
            self.llm_cross_adapter = self._make_view_adapter(args)
        else:
            self.id_cross_adapter = None
            self.llm_cross_adapter = None

        self.id_backbone = self._make_backbone(args)
        self.llm_backbone = (
            self._make_backbone(args)
            if self.split_backbone
            else self.id_backbone
        )

        if self.fusion == "concat":
            self.output_dim = 2 * args.hidden_size
            self.gate = None
        elif self.fusion == "sum":
            self.output_dim = args.hidden_size
            self.gate = None
        else:
            self.output_dim = args.hidden_size
            self.gate = nn.Sequential(
                nn.Linear(2 * args.hidden_size, args.hidden_size),
                nn.Sigmoid(),
            )

        self.cooccurrence = self._load_cooccurrence(paths.frequency)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.align_loss = Contrastive_Loss2(args.tau) if self.user_sim_func == "cl" else nn.MSELoss()
        self.pair_loss = nn.MSELoss()
        self.reg_loss = Contrastive_Loss2(args.tau)

        self._init_weights()

    @staticmethod
    def _load_item_embedding(path, item_num):
        if not path.exists():
            raise FileNotFoundError(f"Missing item embedding file: {path}")

        with path.open("rb") as f:
            emb = pickle.load(f)
        emb = np.asarray(emb, dtype=np.float32)
        expected_rows = item_num + 2

        if emb.shape[0] == item_num:
            emb = np.vstack([np.zeros((1, emb.shape[1]), dtype=np.float32), emb])
        if emb.shape[0] < expected_rows:
            emb = np.vstack([emb, np.zeros((expected_rows - emb.shape[0], emb.shape[1]), dtype=np.float32)])
        if emb.shape[0] > expected_rows:
            emb = emb[:expected_rows]

        emb[0] = 0.0
        emb[item_num + 1] = 0.0
        return torch.tensor(emb, dtype=torch.float32)

    @staticmethod
    def _load_user_embedding(path, user_num):
        if not path.exists():
            raise FileNotFoundError(f"Missing user embedding file: {path}")

        with path.open("rb") as f:
            emb = pickle.load(f)
        emb = np.asarray(emb, dtype=np.float32)

        if emb.shape[0] < user_num:
            emb = np.vstack([emb, np.zeros((user_num - emb.shape[0], emb.shape[1]), dtype=np.float32)])
        if emb.shape[0] > user_num:
            emb = emb[:user_num]

        return torch.tensor(emb, dtype=torch.float32)

    @staticmethod
    def _make_projection(input_dim, hidden_size):
        if input_dim == hidden_size:
            return nn.Identity()
        mid_dim = max(hidden_size, input_dim // 2)
        return nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, hidden_size),
        )

    @staticmethod
    def _make_colmod_projection(input_dim, hidden_size):
        if input_dim == hidden_size:
            return nn.Identity()
        return nn.Sequential(
            nn.Linear(input_dim, int(input_dim / 2)),
            nn.Linear(int(input_dim / 2), hidden_size),
        )

    def _make_backbone(self, args):
        if self.colmod_compat:
            return SASRecBackbone(self.dev, args)
        return CleanSASRecBackbone(
            hidden_size=args.hidden_size,
            num_layers=args.trm_num,
            num_heads=args.num_heads,
            dropout_rate=args.dropout_rate,
        )

    def _make_view_adapter(self, args):
        if self.adapter_type == "mlp":
            return MLPAdapter(args.hidden_size, args.dropout_rate)
        head_num = 2 if self.colmod_compat else args.num_heads
        return Multi_CrossAttention(args.hidden_size, args.hidden_size, head_num)

    @staticmethod
    def _load_cooccurrence(path):
        if not path.exists():
            return {}

        cooccurrence = {}
        with path.open("r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                item_a, item_b, freq = int(parts[0]), int(parts[1]), float(parts[2])
                cooccurrence.setdefault(item_a, {})[item_b] = freq
                cooccurrence.setdefault(item_b, {})[item_a] = freq
        return cooccurrence

    def _init_weights(self):
        modules = [
            self.id_adapter,
            self.llm_adapter,
            self.sem_user_adapter,
            self.collab_user_adapter,
            self.id_cross_adapter,
            self.llm_cross_adapter,
            self.id_backbone,
            self.gate,
            self.pos_emb,
        ]
        if self.split_backbone:
            modules.append(self.llm_backbone)

        for module in modules:
            if module is None or isinstance(module, nn.Identity):
                continue
            for name, param in module.named_parameters():
                if "weight" in name and param.dim() > 1:
                    nn.init.xavier_normal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def _item_views(self, item_ids):
        # id_emb = self.id_adapter(self.id_item_emb(item_ids))
        id_emb = self.id_item_emb(item_ids)
        llm_emb = self.llm_adapter(self.llm_item_emb(item_ids))
        return id_emb, llm_emb

    def _combine_views(self, id_repr, llm_repr):
        if self.fusion == "concat":
            return torch.cat([id_repr, llm_repr], dim=-1)
        if self.fusion == "sum":
            return id_repr + llm_repr

        gate = self.gate(torch.cat([id_repr, llm_repr], dim=-1))
        return gate * id_repr + (1.0 - gate) * llm_repr

    def _build_co_graph(self, item_seq):
        batch_size, seq_len = item_seq.shape
        adj = torch.zeros(batch_size, seq_len, seq_len, device=item_seq.device)
        if not self.cooccurrence:
            return adj

        for batch_idx in range(batch_size):
            items = item_seq[batch_idx].detach().cpu().tolist()
            for row, item_a in enumerate(items):
                neighbors = self.cooccurrence.get(item_a)
                if not neighbors:
                    continue
                for col, item_b in enumerate(items):
                    if row != col and item_b in neighbors:
                        adj[batch_idx, row, col] = neighbors[item_b]
        return self._normalize_graph(adj)

    def _build_modality_graph(self, llm_seq):
        sim = torch.matmul(F.normalize(llm_seq, p=2, dim=-1), F.normalize(llm_seq, p=2, dim=-1).transpose(-1, -2))
        sim = (sim > self.modality_threshold).float()
        return self._normalize_graph(sim)

    @staticmethod
    def _normalize_graph(adj):
        return adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)

    @staticmethod
    def _normalize_signed_graph(adj):
        return adj / (adj.abs().sum(dim=-1, keepdim=True) + 1e-8)

    def _mix_graphs(self, co_adj, modality_adj):
        if self.graph_mix is None:
            return modality_adj
        if self.graph_mix == "collab":
            return co_adj
        if self.graph_mix == "modality":
            return modality_adj
        if self.graph_mix == "union":
            return torch.max(co_adj, modality_adj)
        return torch.min(co_adj, modality_adj)

    def _filter_co_graph(self, co_adj, modality_adj):
        if self.graph_filter == "none":
            return co_adj

        if self.graph_filter == "normalized_residual":
            co_adj = self._normalize_graph(co_adj)
            modality_adj = self._normalize_graph(modality_adj)

        residual = co_adj / (modality_adj + 1e-8)
        if self.graph_filter in {"positive_residual", "normalized_residual"}:
            residual = torch.relu(residual)
            return self._normalize_graph(residual)
        return self._normalize_signed_graph(residual)

    def _graph_convolution(self, adj, seq_emb):
        hidden = self.hgc_dropout(seq_emb)
        for _ in range(self.hgc_layers):
            hidden = torch.bmm(adj, hidden) + hidden
        return hidden

    def _add_position(self, item_ids, seq_emb, positions):
        seq_emb = seq_emb * (seq_emb.size(-1) ** 0.5)
        seq_emb = seq_emb + self.pos_emb(positions.long())
        seq_emb = self.emb_dropout(seq_emb)
        return seq_emb.masked_fill(item_ids.eq(0).unsqueeze(-1), 0.0)

    def _encode_views(self, seq, positions):
        id_seq, llm_seq = self._item_views(seq)
        id_seq = self._add_position(seq, id_seq, positions)
        llm_seq = self._add_position(seq, llm_seq, positions)
        pairwise_align_loss = 0.0

        if self.use_graph:
            co_adj = self._build_co_graph(seq) if self.use_co_graph else torch.zeros(seq.size(0), seq.size(1), seq.size(1), device=seq.device)
            modality_adj = self._build_modality_graph(llm_seq) if self.use_modality_graph else co_adj
            if self.colmod_compat:
                filtered_co_adj = self._normalize_graph(co_adj / (modality_adj + 1e-8))
                mixed_adj = modality_adj
            else:
                filtered_co_adj = self._filter_co_graph(co_adj, modality_adj)
                mixed_adj = self._mix_graphs(filtered_co_adj, modality_adj)
            if self.colmod_compat:
                with torch.no_grad():
                    id_seq = self._graph_convolution(filtered_co_adj, id_seq)
                    llm_seq = self._graph_convolution(mixed_adj, llm_seq)
            else:
                id_seq = self._graph_convolution(filtered_co_adj, id_seq)
                llm_seq = self._graph_convolution(mixed_adj, llm_seq)

        if self.use_cross_att:
            id_source, llm_source = id_seq, llm_seq
            id_seq = self.id_cross_adapter(llm_source, id_source, seq)
            llm_seq = self.llm_cross_adapter(id_source, llm_source, seq)

        id_feats = self.id_backbone(id_seq, seq)
        llm_feats = self.llm_backbone(llm_seq, seq)
        return pairwise_align_loss, id_feats, llm_feats

    def log2feats(self, seq, positions, return_views=False):
        pairwise_align_loss, id_feats, llm_feats = self._encode_views(seq, positions)
        if return_views:
            if self.colmod_compat and self.enable_id:
                return pairwise_align_loss, id_feats, llm_feats
            return id_feats, llm_feats
        if self.colmod_compat and self.enable_id:
            return pairwise_align_loss, id_feats, llm_feats
        return self._combine_views(id_feats, llm_feats)

    def _get_embedding(self, item_ids):
        return self._combine_views(*self._item_views(item_ids))

    def forward(self, seq, pos, neg, positions, **kwargs):
        if self.colmod_compat and self.enable_id:
            _, id_feats, llm_feats = self.log2feats(seq, positions, return_views=True)
            log_feats = self._combine_views(id_feats, llm_feats)
        else:
            id_feats, llm_feats = self.log2feats(seq, positions, return_views=True)
            log_feats = self._combine_views(id_feats, llm_feats)
        pos_emb = self._get_embedding(pos)
        neg_emb = self._get_embedding(neg)

        pos_logits = (log_feats * pos_emb).sum(dim=-1)
        neg_logits = (log_feats * neg_emb).sum(dim=-1)
        valid_mask = pos.ne(0)

        loss = self.loss_func(pos_logits[valid_mask], torch.ones_like(pos_logits[valid_mask]))
        loss = loss + self.loss_func(neg_logits[valid_mask], torch.zeros_like(neg_logits[valid_mask]))

        if self.use_pair_loss:
            loss = loss + self.pair_loss_weight * self.pair_loss(id_feats[valid_mask], llm_feats[valid_mask])
        if self.use_align_loss and "sim_seq" in kwargs and "sim_positions" in kwargs:
            if self.colmod_compat:
                align_weight = self._dynamic_align_weight(kwargs.get("user_id"), seq.device)
                loss = loss + align_weight * self._user_align_loss_colmod(seq, positions, kwargs)
            else:
                align_weight = self._dynamic_align_weight(kwargs.get("user_id"), seq.device)
                loss = loss + align_weight * self._user_align_loss(seq, positions, id_feats, llm_feats, kwargs)
        if self.item_reg:
            loss = loss + self.beta * self._item_regularization(seq)

        return loss

    def _dynamic_align_weight(self, user_id, device):
        if not self.dynamic_align or user_id is None or self.sem_user_emb is None:
            return self.alpha

        user_id = user_id.long().clamp(min=0, max=self.user_num - 1)
        sem_user = self.sem_user_adapter(self.sem_user_emb(user_id))
        collab_user = self.collab_user_adapter(self.collab_user_emb(user_id))
        gap = 1.0 - F.cosine_similarity(sem_user, collab_user, dim=-1)
        weight = (2.0 * torch.sigmoid(-self.dynamic_align_scale * gap)).mean()
        return self.alpha * weight

    def _user_align_loss(self, seq, positions, id_feats, llm_feats, kwargs):
        sim_seq = kwargs["sim_seq"].reshape(-1, seq.size(1))
        sim_positions = kwargs["sim_positions"].reshape(-1, seq.size(1))
        sim_num = kwargs["sim_seq"].size(1)

        with torch.no_grad():
            sim_id_feats, sim_llm_feats = self.log2feats(sim_seq, sim_positions, return_views=True)
            sim_llm_repr = sim_llm_feats[:, -1, :].reshape(seq.size(0), sim_num, -1).mean(dim=1)

            if "sim_collab_seq" in kwargs and "sim_collab_positions" in kwargs:
                sim_collab_seq = kwargs["sim_collab_seq"].reshape(-1, seq.size(1))
                sim_collab_positions = kwargs["sim_collab_positions"].reshape(-1, seq.size(1))
                collab_num = kwargs["sim_collab_seq"].size(1)
                sim_id_feats, _ = self.log2feats(sim_collab_seq, sim_collab_positions, return_views=True)
                sim_id_repr = sim_id_feats[:, -1, :].reshape(seq.size(0), collab_num, -1).mean(dim=1)
            else:
                sim_id_repr = sim_id_feats[:, -1, :].reshape(seq.size(0), sim_num, -1).mean(dim=1)

        id_loss = self.align_loss(id_feats[:, -1, :], sim_id_repr)
        llm_loss = self.align_loss(llm_feats[:, -1, :], sim_llm_repr)
        return self.collab_llm_ratio * id_loss + llm_loss

    def _user_align_loss_colmod(self, seq, positions, kwargs):
        if not self.enable_id:
            log_feats = self.log2feats(seq, positions)[:, -1, :]
            sim_seq = kwargs["sim_seq"].reshape(-1, seq.size(1))
            sim_positions = kwargs["sim_positions"].reshape(-1, seq.size(1))
            sim_num = kwargs["sim_seq"].size(1)

            sim_log_feats = self.log2feats(sim_seq, sim_positions)[:, -1, :]
            sim_log_feats = sim_log_feats.detach().reshape(seq.size(0), sim_num, -1).mean(dim=1)
            return self.align_loss(log_feats, sim_log_feats)

        _, collab_feats, llm_feats = self.log2feats(seq, positions)
        collab_feats = collab_feats[:, -1, :].reshape(seq.size(0), -1)
        llm_feats = llm_feats[:, -1, :].reshape(seq.size(0), -1)

        sim_seq = kwargs["sim_seq"].reshape(-1, seq.size(1))
        sim_positions = kwargs["sim_positions"].reshape(-1, seq.size(1))
        sim_num = kwargs["sim_seq"].size(1)
        _, _, sim_llm_feats = self.log2feats(sim_seq, sim_positions)
        sim_llm_feats = sim_llm_feats[:, -1, :].detach().reshape(seq.size(0), sim_num, -1).mean(dim=1)

        sim_collab_seq = kwargs["sim_collab_seq"].reshape(-1, seq.size(1))
        sim_collab_positions = kwargs["sim_collab_positions"].reshape(-1, seq.size(1))
        collab_num = kwargs["sim_collab_seq"].size(1)
        _, sim_collab_feats, _ = self.log2feats(sim_collab_seq, sim_collab_positions)
        sim_collab_feats = sim_collab_feats[:, -1, :].detach().reshape(seq.size(0), collab_num, -1).mean(dim=1)

        id_loss = self.align_loss(collab_feats, sim_collab_feats)
        llm_loss = self.align_loss(llm_feats, sim_llm_feats)
        return self.collab_llm_ratio * id_loss + llm_loss

    def _item_regularization(self, seq):
        item_ids = torch.masked_select(seq, seq > 0)
        id_emb, llm_emb = self._item_views(item_ids)
        return self.reg_loss(llm_emb, id_emb)

    def predict(self, seq, item_indices, positions, **kwargs):
        if self.colmod_compat and self.enable_id:
            _, id_feats, llm_feats = self.log2feats(seq, positions)
            final_feat = self._combine_views(id_feats, llm_feats)[:, -1, :]
        else:
            final_feat = self.log2feats(seq, positions)[:, -1, :]
        item_emb = self._get_embedding(item_indices)
        return item_emb.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

    def get_user_emb(self, seq, positions, **kwargs):
        if self.colmod_compat and self.enable_id:
            _, id_feats, llm_feats = self.log2feats(seq, positions, return_views=True)
        else:
            id_feats, llm_feats = self.log2feats(seq, positions, return_views=True)
        return self._combine_views(id_feats[:, -1, :], llm_feats[:, -1, :])
