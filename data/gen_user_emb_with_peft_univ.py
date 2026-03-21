# gen_user_emb_with_peft.py
# ============================================================
# Unified script for:
# 1) Training a LoRA+BERT user embedder on item-id sequences
#    with contrastive learning and data augmentation.
# 2) Training a LoRA+BERT user embedder on text prompts
#    with contrastive learning and data augmentation.
# 3) Exporting user embeddings for both pipelines:
#    - enable_id=True: item-id -> (proj, cls)
#    - enable_id=False: text prompt -> mean pooled BERT
# Notes:
# - All comments are in English as requested.
# - Requires: torch, transformers, peft, tqdm, numpy
# ============================================================

import os
import json
import pickle
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertModel, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

# -----------------------------
# Argparse
# -----------------------------
parser = argparse.ArgumentParser(description='PEFT(LoRA) training and user embedding export')

# Path
parser.add_argument('--adapter_dir_itemid', type=str, default='./lora_adapters_itemid', help='Save dir for item-id LoRA adapters')
parser.add_argument('--extras_path_itemid', type=str, default='./user_embedder_itemid_extras.pt', help='Save path for item-id extras')
parser.add_argument('--adapter_dir_text', type=str, default='./lora_adapters_text', help='Save dir for text LoRA adapters')

# Common
parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g. ml-1m)')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased', help='Base BERT name')
# parser.add_argument('--inter_path', type=str, default='./handled/inter.txt', help='Path to interactions file')
# parser.add_argument('--id_map_path', type=str, default='./handled/id_map.json', help='Path to id_map.json')
# parser.add_argument('--item_meta_path', type=str, default='./handled/item2attributes.json', help='Path to item2attributes.json')
# parser.add_argument('--user_index_out', type=str, default='./handled/user_index.json', help='Output path for row_index->user_id map')

# Training
parser.add_argument('--train', action='store_true', default=True, help='Enable training (default: True)')
parser.add_argument('--no-train', action='store_false', dest='train', help='Disable training')
parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--temperature', type=float, default=0.2, help='Contrastive temperature')
parser.add_argument('--max_seq_len', type=int, default=64, help='Max sequence length for item-id pipeline')
parser.add_argument('--proj_dim', type=int, default=128, help='Projection dimension')
parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
parser.add_argument('--target_modules', nargs='+', default=['query','value'], help='LoRA target module name fragments')
parser.add_argument('--last_k', type=int, default=0, help='Use only the last K interactions per user (0=use all)')

# Export
parser.add_argument('--export', action='store_true', default=True, help='Export user embeddings')
parser.add_argument('--no-export', action='store_false', dest='export', help='Disable export')
parser.add_argument('--enable_id', action='store_true', help='Use item-id pipeline (otherwise text pipeline)')
parser.add_argument('--export_batch_size', type=int, default=1, help='Batch size for export')
parser.add_argument('--out_text_path', type=str, default='./handled/usr_emb_peft_np.pkl', help='Output for text pipeline')
parser.add_argument('--out_id_proj_path', type=str, default='./handled/usr_emb_collab_proj_np.pkl', help='Output for PROJ embeddings')
parser.add_argument('--out_id_cls_path', type=str, default='./handled/usr_emb_collab_cls_np.pkl', help='Output for CLS embeddings')

# Augmentation parameters
parser.add_argument('--drop_prob_id', type=float, default=0.1, help='Drop prob for item-id pipeline')
parser.add_argument('--shuffle_prob_id', type=float, default=0.15, help='Shuffle prob for item-id pipeline')
parser.add_argument('--mask_prob_id', type=float, default=0.05, help='Mask prob for item-id pipeline')

parser.add_argument('--drop_prob_text', type=float, default=0.2, help='Drop prob for text pipeline')
parser.add_argument('--shuffle_prob_text', type=float, default=0.15, help='Shuffle prob for text pipeline')
parser.add_argument('--truncate_prob_text', type=float, default=0.3, help='Truncate prob for text pipeline')

args = parser.parse_args()
DEVICE = args.device

# -----------------------------
# Utilities
# -----------------------------

def get_dataset_path(*path_parts):
    """Construct path relative to dataset directory."""
    return os.path.join(args.dataset, *path_parts)
    
def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def infer_num_items_from_id_map(id_map: dict) -> int:
    """Infer item vocabulary size from id_map['id2item'] keys. Reserve 0 for PAD."""
    max_id = max(int(k) for k in id_map["id2item"].keys())
    return max_id + 1

def load_user_histories(inter_path: str) -> Dict[int, List[int]]:
    """Load user histories from file."""
    User = defaultdict(list)
    with open(inter_path, 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            User[int(u)].append(int(i))
    train = {}
    for u in User:
        if len(User[u]) < 3:
            train[u] = User[u]
        else:
            train[u] = User[u][:-2]
    return train

# -----------------------------
# Text prompt pipeline
# -----------------------------
prompt_template = "The user has visited following fashions: \n<HISTORY> \nplease conclude the user's perference."

def build_text_prompt(history: List[int], id_map: dict, item_meta: dict, augment=False) -> str:
    titles = []
    for item in history:
        try:
            titles.append(item_meta[id_map["id2item"][str(item)]]["title"])
        except:
            continue
    if not titles:
        titles = ["[EMPTY]"]
    if augment:
        titles = augment_titles(titles)
    hist_str = ", ".join(titles)
    if len(hist_str) > 8000:
        hist_str = hist_str[-8000:]
    return prompt_template.replace("<HISTORY>", hist_str)

# -----------------------------
# Text data augmentation
# -----------------------------
def text_drop(titles, p):
    return [t for t in titles if np.random.rand() > p] or [titles[0]]

def text_shuffle(titles, window=3, p=0.15):
    titles = titles[:]
    for i in range(len(titles)):
        if np.random.rand() < p:
            j = min(len(titles)-1, i+np.random.randint(1, window+1))
            titles[i], titles[j] = titles[j], titles[i]
    return titles

def text_truncate(titles, p):
    if len(titles) < 2 or np.random.rand() > p:
        return titles
    cut = np.random.randint(1, len(titles))
    return titles[:cut] if np.random.rand() < 0.5 else titles[cut:]

def augment_titles(titles):
    t = text_drop(titles, args.drop_prob_text)
    t = text_shuffle(t, p=args.shuffle_prob_text)
    t = text_truncate(t, args.truncate_prob_text)
    return t

# -----------------------------
# Item-id pipeline model
# -----------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim)
        )
    def forward(self, x):
        return self.net(x)

class UserEmbedder(nn.Module):
    def __init__(self, bert_model: BertModel, num_items: int, hidden_size: int, proj_dim: int):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, hidden_size, padding_idx=0)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.bert = bert_model
        self.proj = ProjectionHead(hidden_size, proj_dim)
    def forward(self, item_ids, mask):
        B = item_ids.size(0)
        emb = self.item_embedding(item_ids)
        cls_emb = self.cls_token.expand(B, 1, -1)
        inputs_embeds = torch.cat([cls_emb, emb], dim=1)
        cls_mask = torch.ones(B, 1, dtype=mask.dtype, device=mask.device)
        mask = torch.cat([cls_mask, mask], dim=1)
        out = self.bert(inputs_embeds=inputs_embeds, attention_mask=mask)
        cls_vec = out.last_hidden_state[:, 0, :]
        return self.proj(cls_vec), cls_vec

def build_lora_on_bert(bert):
    cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    return get_peft_model(bert, cfg)

# -----------------------------
# Item-id augmentation
# -----------------------------
def drop(seq, p):
    return [x for x in seq if np.random.rand() > p] or [seq[0]]

def shuffle(seq, window=3, p=0.15):
    seq = seq[:]
    for i in range(len(seq)):
        if np.random.rand() < p:
            j = min(len(seq)-1, i+np.random.randint(1, window+1))
            seq[i], seq[j] = seq[j], seq[i]
    return seq

def mask(seq, p):
    return [0 if np.random.rand() < p else x for x in seq]

def augment_id(seq):
    s = drop(seq, args.drop_prob_id)
    s = shuffle(s, p=args.shuffle_prob_id)
    s = mask(s, args.mask_prob_id)
    return s

# -----------------------------
# Dataset for training
# -----------------------------
class UserDataset(Dataset):
    def __init__(self, histories, last_k=0, maxlen=64, enable_id=True, id_map=None, meta=None):
        self.uids = sorted(histories.keys())
        self.hist = histories
        self.last_k = last_k
        self.maxlen = maxlen
        self.enable_id = enable_id
        self.id_map = id_map
        self.meta = meta
    def __len__(self): return len(self.uids)
    def __getitem__(self, idx):
        u = self.uids[idx]
        h = self.hist[u]
        if self.last_k > 0: h = h[-self.last_k:]
        if self.enable_id:
            base = [x for x in h if x>=1][:self.maxlen]
            v1 = augment_id(base)
            v2 = augment_id(base)
            return v1,v2
        else:
            t1 = build_text_prompt(h, self.id_map, self.meta, augment=True)
            t2 = build_text_prompt(h, self.id_map, self.meta, augment=True)
            return t1,t2

def pad_batch(seqs):
    maxlen = min(max(len(s) for s in seqs) if seqs else 1, args.max_seq_len)
    B = len(seqs)
    p = torch.zeros(B,maxlen,dtype=torch.long)
    m = torch.zeros(B,maxlen,dtype=torch.long)
    for i,s in enumerate(seqs):
        p[i,:min(len(s), maxlen)] = torch.tensor(s[:maxlen])
        m[i,:min(len(s), maxlen)] = 1
    return p,m

# -----------------------------
# Contrastive loss
# -----------------------------
def nt_xent(z1,z2,temp):
    B=z1.size(0)
    z=torch.cat([z1,z2],0)
    z=F.normalize(z,1)
    sim=z@z.T/temp
    sim.fill_diagonal_(-9e15)
    t=torch.arange(B,device=z.device)
    t=torch.cat([t+B,t],0)
    return F.cross_entropy(sim,t)

# -----------------------------
# Training loops
# -----------------------------
def train_itemid(histories,num_items):

    adapter_dir_itemid = get_dataset_path(args.adapter_dir_itemid)
    os.makedirs(adapter_dir_itemid, exist_ok=True)
    extras_path_itemid = get_dataset_path(args.extras_path_itemid)
    

    bert=BertModel.from_pretrained(args.pretrained_model)
    bert=build_lora_on_bert(bert)
    model=UserEmbedder(bert,num_items,bert.config.hidden_size,args.proj_dim).to(DEVICE)
    for n,p in model.named_parameters():
        p.requires_grad = "lora" in n or n.startswith(("item_embedding","cls_token","proj"))
    ds=UserDataset(histories,args.last_k,args.max_seq_len,True)
    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True,collate_fn=lambda b:(pad_batch([x[0] for x in b]),pad_batch([x[1] for x in b])),drop_last=True)
    opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=args.lr)
    for ep in range(args.epochs):
        tot=0;n=0
        for (p1,m1),(p2,m2) in tqdm(dl):
            p1,m1,p2,m2=p1.to(DEVICE),m1.to(DEVICE),p2.to(DEVICE),m2.to(DEVICE)
            z1,_=model(p1,m1); z2,_=model(p2,m2)
            loss=nt_xent(z1,z2,args.temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item()*p1.size(0); n+=p1.size(0)
        print(f"[ItemID] Epoch {ep+1}: loss={tot/n:.6f}")
    model.bert.save_pretrained(adapter_dir_itemid)
    torch.save({"item_embedding":model.item_embedding.state_dict(),
                "cls_token":model.cls_token.detach().cpu(),
                "proj":model.proj.state_dict()},extras_path_itemid)

def train_text(histories,id_map,meta):

    adapter_dir_text = get_dataset_path(args.adapter_dir_text)
    os.makedirs(adapter_dir_text, exist_ok=True)

    tok=BertTokenizer.from_pretrained(args.pretrained_model)
    bert=BertModel.from_pretrained(args.pretrained_model)
    bert=build_lora_on_bert(bert).to(DEVICE)
    for n,p in bert.named_parameters(): p.requires_grad="lora" in n
    ds=UserDataset(histories,args.last_k,enable_id=False,id_map=id_map,meta=meta)
    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True,drop_last=True)
    opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,bert.parameters()),lr=args.lr)
    for ep in range(args.epochs):
        tot=0;n=0
        for t1,t2 in tqdm(dl):
            enc1=tok(list(t1),return_tensors="pt",padding=True,truncation=True,max_length=512).to(DEVICE)
            enc2=tok(list(t2),return_tensors="pt",padding=True,truncation=True,max_length=512).to(DEVICE)
            z1=bert(**enc1).last_hidden_state.mean(1)
            z2=bert(**enc2).last_hidden_state.mean(1)
            loss=nt_xent(z1,z2,args.temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item()*len(t1); n+=len(t1)
        print(f"[Text] Epoch {ep+1}: loss={tot/n:.6f}")
    bert.save_pretrained(adapter_dir_text)

# -----------------------------
# Export
# -----------------------------
def export_itemid(histories,num_items):

    adapter_dir_itemid = get_dataset_path(args.adapter_dir_itemid)
    extras_path_itemid = get_dataset_path(args.extras_path_itemid)


    config=AutoConfig.from_pretrained(args.pretrained_model)
    bert=BertModel.from_pretrained(args.pretrained_model,config=config).to(DEVICE)
    model=UserEmbedder(bert,num_items,bert.config.hidden_size,args.proj_dim).to(DEVICE)
    model.bert=PeftModel.from_pretrained(model.bert, adapter_dir_itemid).to(DEVICE)
    payload=torch.load(extras_path_itemid,map_location=DEVICE)
    model.item_embedding.load_state_dict(payload["item_embedding"])
    with torch.no_grad(): model.cls_token.copy_(payload["cls_token"].to(DEVICE))
    model.proj.load_state_dict(payload["proj"]); model.eval()
    seqs=[[x for x in histories[u] if x>=1][:64] for u in sorted(histories.keys())]
    embs_proj=[];embs_cls=[]
    with torch.no_grad():
        for i in range(0,len(seqs),args.export_batch_size):
            batch=seqs[i:i+args.export_batch_size]
            p,m=pad_batch(batch); p,m=p.to(DEVICE),m.to(DEVICE)
            z,cls=model(p,m)
            embs_proj.append(z); embs_cls.append(cls)
    proj=torch.concatenate(embs_proj).cpu().numpy(); cls=torch.concatenate(embs_cls).cpu().numpy()

    out_id_proj_path = get_dataset_path('handled', 'usr_emb_collab_proj_np.pkl')
    out_id_cls_path = get_dataset_path('handled', 'usr_emb_collab_cls_np.pkl')

    pickle.dump(proj,open(out_id_proj_path,"wb"))
    pickle.dump(cls,open(out_id_cls_path,"wb"))
    print("[Export ItemID]",proj.shape,cls.shape)

def export_text(histories,id_map,meta):

    adapter_dir = get_dataset_path(args.adapter_dir_text)
    out_text_path = get_dataset_path('handled', 'usr_emb_peft_np.pkl')

    tok=BertTokenizer.from_pretrained(args.pretrained_model)
    bert=BertModel.from_pretrained(args.pretrained_model).to(DEVICE)
    bert=PeftModel.from_pretrained(bert,adapter_dir).to(DEVICE); bert.eval()
    texts=[build_text_prompt(histories[u],id_map,meta) for u in sorted(histories.keys())]
    embs=[]
    for i in range(0,len(texts),args.export_batch_size):
        batch=texts[i:i+args.export_batch_size]
        enc=tok(batch,return_tensors="pt",padding=True,truncation=True,max_length=512).to(DEVICE)
        out=bert(**enc).last_hidden_state.mean(1)
        embs.append(out.cpu().numpy())
    arr=np.concatenate(embs)

    pickle.dump(arr,open(out_text_path,"wb"))
    print("[Export Text]",arr.shape)

# -----------------------------
# Main
# -----------------------------
def main():
    inter_path = get_dataset_path('handled', 'inter.txt')
    id_map_path = get_dataset_path('handled', 'id_map.json')
    item_meta_path = get_dataset_path('handled', 'item2attributes.json')
    user_index_out = get_dataset_path('handled', 'user_index.json')

    id_map=load_json(id_map_path)
    meta=load_json(item_meta_path)
    hist=load_user_histories(inter_path)
    num_items=infer_num_items_from_id_map(id_map)
    if args.train:
        if args.enable_id: train_itemid(hist,num_items)
        else: train_text(hist,id_map,meta)
    if args.export:
        with open(user_index_out,"w") as f:
            json.dump({i:u for i,u in enumerate(sorted(hist.keys()))},f)
        if args.enable_id: export_itemid(hist,num_items)
        else: export_text(hist,id_map,meta)

if __name__=="__main__":
    main()
