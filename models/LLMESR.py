# here put the import lib
import torch
import torch.nn as nn
# from models.DualLLMSRS import DualLLMSASRec, DualLLMGRU4Rec, DualLLMBert4Rec
from models.DualLLMSRS import *
from models.utils import Contrastive_Loss2



class LLMESR_SASRec(DualLLMSASRec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha
        self.user_sim_func = args.user_sim_func
        self.item_reg = args.item_reg

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        if self.item_reg:
            self.beta = args.beta
            self.reg = Contrastive_Loss2()

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, **kwargs)  # get the original loss
        
        log_feats = self.log2feats(seq, positions)[:, -1, :]
        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq.shape[1]), kwargs["sim_positions"].view(-1, seq.shape[1])
        sim_num = kwargs["sim_seq"].shape[1]
        sim_log_feats = self.log2feats(sim_seq, sim_positions)[:, -1, :]    # (bs*sim_num, hidden_size)
        sim_log_feats = sim_log_feats.detach().view(seq.shape[0], sim_num, -1)  # (bs, sim_num, hidden_size)
        sim_log_feats = torch.mean(sim_log_feats, dim=1)

        if self.user_sim_func == "cl":
            # align_loss = self.align(self.projector1(log_feats), self.projector2(sim_log_feats))
            align_loss = self.align(log_feats, sim_log_feats)
        elif self.user_sim_func == "kd":
            align_loss = self.align(log_feats, sim_log_feats)

        if self.item_reg:
            unfold_item_id = torch.masked_select(seq, seq>0)
            llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))
            id_item_emb = self.id_item_emb(unfold_item_id)
            reg_loss = self.reg(llm_item_emb, id_item_emb)
            loss += self.beta * reg_loss

        loss += self.alpha * align_loss

        return loss
    


class LLMESR_GRU4Rec(DualLLMGRU4Rec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha
        self.user_sim_func = args.user_sim_func
        self.item_reg = args.item_reg

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        if self.item_reg:
            self.beta = args.beta
            self.reg = Contrastive_Loss2()

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, **kwargs)  # get the original loss
        
        log_feats = self.log2feats(seq)[:, -1, :]
        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq.shape[1]), kwargs["sim_positions"].view(-1, seq.shape[1])
        sim_num = kwargs["sim_seq"].shape[1]
        sim_log_feats = self.log2feats(sim_seq)[:, -1, :]    # (bs*sim_num, hidden_size)
        sim_log_feats = sim_log_feats.detach().view(seq.shape[0], sim_num, -1)  # (bs, sim_num, hidden_size)
        sim_log_feats = torch.mean(sim_log_feats, dim=1)

        if self.user_sim_func == "cl":
            # align_loss = self.align(self.projector1(log_feats), self.projector2(sim_log_feats))
            align_loss = self.align(log_feats, sim_log_feats)
        elif self.user_sim_func == "kd":
            align_loss = self.align(log_feats, sim_log_feats)

        if self.item_reg:
            unfold_item_id = torch.masked_select(seq, seq>0)
            llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))
            id_item_emb = self.id_item_emb(unfold_item_id)
            reg_loss = self.reg(llm_item_emb, id_item_emb)
            loss += self.beta * reg_loss

        loss += self.alpha * align_loss

        return loss



class LLMESR_Bert4Rec(DualLLMBert4Rec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha
        self.user_sim_func = args.user_sim_func
        self.item_reg = args.item_reg

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        if self.item_reg:
            self.reg = Contrastive_Loss2()

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, **kwargs)  # get the original loss
        
        log_feats = self.log2feats(seq, positions)[:, -1, :]
        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq.shape[1]), kwargs["sim_positions"].view(-1, seq.shape[1])
        sim_num = kwargs["sim_seq"].shape[1]
        sim_log_feats = self.log2feats(sim_seq, sim_positions)[:, -1, :]
        sim_log_feats = sim_log_feats.detach().view(seq.shape[0], sim_num, -1)  # (bs, sim_num, hidden_size)
        sim_log_feats = torch.mean(sim_log_feats, dim=1)

        if self.user_sim_func == "cl":
            # align_loss = self.align(self.projector1(log_feats), self.projector2(sim_log_feats))
            align_loss = self.align(log_feats, sim_log_feats)
        elif self.user_sim_func == "kd":
            align_loss = self.align(log_feats, sim_log_feats)

        loss += self.alpha * align_loss

        return loss


class LLMESR_ColMod(DualColMod):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha
        self.user_sim_func = args.user_sim_func
        self.item_reg = args.item_reg
        self.args = args

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        if self.item_reg:
            self.beta = args.beta
            self.reg = Contrastive_Loss2()

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, **kwargs)  # get the original loss

        # # for theory verification
        # # ==================
        # return loss
        # # ==================
        
        if not self.enable_id:
            log_feats = self.log2feats(seq, positions)[:, -1, :]
            sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq.shape[1]), kwargs["sim_positions"].view(-1, seq.shape[1])
            sim_num = kwargs["sim_seq"].shape[1]
            sim_log_feats = self.log2feats(sim_seq, sim_positions)[:, -1, :]    # (bs*sim_num, hidden_size)
            sim_log_feats = sim_log_feats.detach().view(seq.shape[0], sim_num, -1)  # (bs, sim_num, hidden_size)
            sim_log_feats = torch.mean(sim_log_feats, dim=1)

            if self.user_sim_func == "cl":
                # align_loss = self.align(self.projector1(log_feats), self.projector2(sim_log_feats))
                align_loss = self.align(log_feats, sim_log_feats)
            elif self.user_sim_func == "kd":
                align_loss = self.align(log_feats, sim_log_feats)

            loss += self.alpha * align_loss

        else:
            pairwise_align_loss, collab_feats, llm_feats = self.log2feats(seq, positions)
            collab_feats = collab_feats[:, -1, :].clone().view(seq.shape[0], -1)
            llm_feats = llm_feats[:, -1, :].clone().view(seq.shape[0], -1)
            loss += self.args.pair_loss_weight * pairwise_align_loss

            sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq.shape[1]), kwargs["sim_positions"].view(-1, seq.shape[1])
            sim_num = kwargs["sim_seq"].shape[1]
            _, _, sim_llm_feats = self.log2feats(sim_seq, sim_positions)
            sim_llm_feats = sim_llm_feats[:, -1, :].detach().view(seq.shape[0], sim_num, -1)  # (bs, sim_num, hidden_size)
            sim_llm_feats = torch.mean(sim_llm_feats, dim=1)

            sim_seq, sim_positions = kwargs["sim_collab_seq"].view(-1, seq.shape[1]), kwargs["sim_collab_positions"].view(-1, seq.shape[1])
            sim_num = kwargs["sim_collab_seq"].shape[1]
            _, sim_collab_feats, _ = self.log2feats(sim_seq, sim_positions)
            sim_collab_feats = sim_collab_feats[:, -1, :].detach().view(seq.shape[0], sim_num, -1)  # (bs, sim_num, hidden_size)
            sim_collab_feats = torch.mean(sim_collab_feats, dim=1)

            if self.user_sim_func == "cl":
                # align_loss = self.align(self.projector1(log_feats), self.projector2(sim_log_feats))
                align_loss = self.args.collab_llm_ratio * self.align(collab_feats, sim_collab_feats) \
                            + self.align(llm_feats, sim_llm_feats)
            elif self.user_sim_func == "kd":
                align_loss = self.args.collab_llm_ratio * self.align(collab_feats, sim_collab_feats) \
                            + self.align(llm_feats, sim_llm_feats)

            loss += self.alpha * align_loss

            

        if self.item_reg:
            unfold_item_id = torch.masked_select(seq, seq>0)
            llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))
            id_item_emb = self.id_item_emb(unfold_item_id)
            reg_loss = self.reg(llm_item_emb, id_item_emb)
            loss += self.beta * reg_loss

        return loss


class LLMESR_IntentColMod(DualIntentColMod):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha
        self.user_sim_func = args.user_sim_func
        self.item_reg = args.item_reg
        self.args = args

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        if self.item_reg:
            self.beta = args.beta
            self.reg = Contrastive_Loss2()

        self._init_weights()

    def forward(self,
                seq,
                pos,
                neg,
                positions,
                **kwargs):

        pairwise_align_loss, collab_log_feats, llm_log_feats = self.log2feats(
            seq,
            positions,
            sim_seq=kwargs["sim_seq"],
            sim_positions=kwargs["sim_positions"],
            return_alignment=True,
        )
        log_feats = torch.cat([collab_log_feats, llm_log_feats], dim=-1)

        valid_mask = (seq > 0)
        position_index = torch.arange(
            seq.shape[1], device=seq.device).unsqueeze(0).expand_as(seq)
        last_index = position_index.masked_fill(~valid_mask, -1).max(dim=1).values
        valid_users = (last_index >= 0)
        safe_last_index = last_index.clamp_min(0)
        batch_index = torch.arange(seq.shape[0], device=seq.device)

        final_feats = log_feats[batch_index, safe_last_index]
        final_pos = pos[batch_index, safe_last_index]
        final_neg = neg[batch_index, safe_last_index]
        pos_embs = self._get_embedding(final_pos)
        neg_embs = self._get_embedding(final_neg)
        pos_logits = (final_feats * pos_embs).sum(dim=-1)
        neg_logits = (final_feats * neg_embs).sum(dim=-1)
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        loss = self.loss_func(pos_logits[valid_users], pos_labels[valid_users]) \
             + self.loss_func(neg_logits[valid_users], neg_labels[valid_users])
        loss += self.args.pair_loss_weight * pairwise_align_loss

        # Keep semantic-neighbor self-distillation in addition to using the
        # same neighbors to construct the dynamic collaborative prefix tokens.
        sim_seq = kwargs["sim_seq"]
        sim_positions = kwargs["sim_positions"]
        sim_num = sim_seq.shape[1]
        flat_sim_seq = sim_seq.reshape(-1, seq.shape[1])
        flat_sim_positions = sim_positions.reshape(-1, seq.shape[1])
        with torch.no_grad():
            _, _, sim_llm_feats = self.log2feats(
                flat_sim_seq, flat_sim_positions)
        sim_llm_feats = sim_llm_feats[:, -1, :].reshape(
            seq.shape[0], sim_num, -1).mean(dim=1)
        anchor_llm_feats = llm_log_feats[batch_index, safe_last_index]
        loss += self.alpha * self.align(anchor_llm_feats, sim_llm_feats)

        if self.item_reg:
            unfold_item_id = torch.masked_select(seq, seq > 0)
            llm_item_emb = self.first_adapter(self.llm_item_emb(unfold_item_id))
            id_item_emb = self.id_item_emb(unfold_item_id)
            reg_loss = self.reg(llm_item_emb, id_item_emb)
            loss += self.beta * reg_loss

        return loss
