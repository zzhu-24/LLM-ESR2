import pickle

import numpy as np
import torch
import torch.nn as nn

from models.clean_backbones import CleanSASRecBackbone
from utils.paths import get_dataset_paths


class LLMESRClean(nn.Module):
    """Clean LLM-ESR variant: dual item views, one SR backbone, optional user alignment."""

    def __init__(self, user_num, item_num, device, args):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = device
        self.hidden_size = args.hidden_size
        self.fusion = args.fusion
        self.use_align_loss = args.use_align_loss
        self.alpha = args.alpha

        if self.fusion not in {"sum", "concat", "gate"}:
            raise ValueError(f"Unsupported fusion: {self.fusion}")

        paths = get_dataset_paths(args.dataset, args.inter_file)
        id_weight = self._load_item_embedding(paths.id_item_emb, item_num)
        llm_weight = self._load_item_embedding(paths.llm_item_emb, item_num)

        self.id_item_emb = nn.Embedding.from_pretrained(id_weight, freeze=False, padding_idx=0)
        self.llm_item_emb = nn.Embedding.from_pretrained(llm_weight, freeze=args.freeze, padding_idx=0)
        self.id_adapter = self._make_projection(id_weight.size(1), args.hidden_size)
        self.llm_adapter = self._make_projection(llm_weight.size(1), args.hidden_size)

        if self.fusion == "concat":
            model_dim = 2 * args.hidden_size
        else:
            model_dim = args.hidden_size

        self.gate = None
        if self.fusion == "gate":
            self.gate = nn.Sequential(
                nn.Linear(2 * args.hidden_size, args.hidden_size),
                nn.Sigmoid(),
            )

        self.pos_emb = nn.Embedding(args.max_len + 100, model_dim)
        self.emb_dropout = nn.Dropout(args.dropout_rate)
        self.backbone = CleanSASRecBackbone(
            hidden_size=model_dim,
            num_layers=args.trm_num,
            num_heads=args.num_heads,
            dropout_rate=args.dropout_rate,
        )
        self.loss_func = nn.BCEWithLogitsLoss()
        self.align_loss = nn.MSELoss()

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
            pad_rows = expected_rows - emb.shape[0]
            emb = np.vstack([emb, np.zeros((pad_rows, emb.shape[1]), dtype=np.float32)])
        if emb.shape[0] > expected_rows:
            emb = emb[:expected_rows]

        emb[0] = 0.0
        emb[item_num + 1] = 0.0
        return torch.tensor(emb, dtype=torch.float32)

    @staticmethod
    def _make_projection(input_dim, hidden_size):
        if input_dim == hidden_size:
            return nn.Identity()
        return nn.Sequential(
            nn.Linear(input_dim, max(hidden_size, input_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(hidden_size, input_dim // 2), hidden_size),
        )

    def _init_weights(self):
        for module in [self.id_adapter, self.llm_adapter, self.gate, self.pos_emb, self.backbone]:
            if module is None or isinstance(module, nn.Identity):
                continue
            for name, param in module.named_parameters():
                if "weight" in name and param.dim() > 1:
                    nn.init.xavier_normal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def _item_views(self, item_ids):
        id_emb = self.id_adapter(self.id_item_emb(item_ids))
        llm_emb = self.llm_adapter(self.llm_item_emb(item_ids))
        return id_emb, llm_emb

    def _get_embedding(self, item_ids):
        id_emb, llm_emb = self._item_views(item_ids)
        if self.fusion == "sum":
            return id_emb + llm_emb
        if self.fusion == "concat":
            return torch.cat([id_emb, llm_emb], dim=-1)

        gate = self.gate(torch.cat([id_emb, llm_emb], dim=-1))
        return gate * id_emb + (1.0 - gate) * llm_emb

    def log2feats(self, seq, positions):
        seq_emb = self._get_embedding(seq)
        seq_emb = seq_emb * (seq_emb.size(-1) ** 0.5)
        seq_emb = seq_emb + self.pos_emb(positions.long())
        seq_emb = self.emb_dropout(seq_emb)
        return self.backbone(seq_emb, seq)

    def forward(self, seq, pos, neg, positions, **kwargs):
        log_feats = self.log2feats(seq, positions)
        pos_emb = self._get_embedding(pos)
        neg_emb = self._get_embedding(neg)

        pos_logits = (log_feats * pos_emb).sum(dim=-1)
        neg_logits = (log_feats * neg_emb).sum(dim=-1)
        valid_mask = pos.ne(0)

        loss = self.loss_func(pos_logits[valid_mask], torch.ones_like(pos_logits[valid_mask]))
        loss = loss + self.loss_func(neg_logits[valid_mask], torch.zeros_like(neg_logits[valid_mask]))

        if self.use_align_loss and "sim_seq" in kwargs and "sim_positions" in kwargs:
            loss = loss + self.alpha * self._user_align_loss(seq, positions, kwargs)

        return loss

    def _user_align_loss(self, seq, positions, kwargs):
        user_repr = self.log2feats(seq, positions)[:, -1, :]
        sim_seq = kwargs["sim_seq"].reshape(-1, seq.size(1))
        sim_positions = kwargs["sim_positions"].reshape(-1, seq.size(1))
        sim_num = kwargs["sim_seq"].size(1)
        with torch.no_grad():
            sim_repr = self.log2feats(sim_seq, sim_positions)[:, -1, :]
            sim_repr = sim_repr.reshape(seq.size(0), sim_num, -1).mean(dim=1)
        return self.align_loss(user_repr, sim_repr)

    def predict(self, seq, item_indices, positions, **kwargs):
        final_feat = self.log2feats(seq, positions)[:, -1, :]
        item_emb = self._get_embedding(item_indices)
        return item_emb.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

    def get_user_emb(self, seq, positions, **kwargs):
        return self.log2feats(seq, positions)[:, -1, :]
