from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.loss import BPRLoss, CELoss

class Gru4Rec(nn.Module):
    def __init__(
        self, 
        n_items,

        d_model,
        inner_dim,
        num_layers,
        dropout,

        pad_idx=0,
        loss_type: Literal["bpr", "ce"] = "ce"
    ):
        super(Gru4Rec, self).__init__()
        self.n_items = n_items
        self.d_model = d_model
        self.inner_dim = inner_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.item_emb = nn.Embedding(n_items + 1, d_model) # zero for padding
        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(d_model, inner_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(inner_dim, d_model)

        self.pad_idx = pad_idx
        self.loss_type = loss_type
        self.loss_func = self.get_loss_func()

    def get_loss_func(self):
        if self.loss_type == "bpr":
            return BPRLoss()
        elif self.loss_type == "ce":
            return CELoss()
        else:
            raise ValueError("Invalid loss type.")    

    def encode_seqs(self, his_seqs):
        # his_seqs: (batch_size, seq_len)
        x = self.item_emb(his_seqs) # (batch_size, seq_len, d_model)
        x = self.emb_dropout(x)
        x, _ = self.gru(x)
        x = self.fc(x)
        return x
    
    def extract(self, his_emb, target_indices):
        assert his_emb.shape[0] == target_indices.shape[0]
        res = []
        for i in range(his_emb.shape[0]):
            res.append(his_emb[i, target_indices[i], :])
        return torch.stack(res, dim=0)
    
    def forward(self, his_seqs, next_items, neg_items=None):
        # his_seqs: (batch_size, seq_len), next_items: (batch_size), neg_items: (batch_size)
        his_emb = self.encode_seqs(his_seqs)
        target_indices = (his_seqs != self.pad_idx).sum(dim=-1) - 1
        target_emb = self.extract(his_emb, target_indices)

        if self.loss_type == "bpr" and neg_items is not None:
            pos_emb = self.item_emb(next_items)
            neg_emb = self.item_emb(neg_items)
            pos_scores = torch.sum(target_emb * pos_emb, dim=-1)
            neg_scores = torch.sum(target_emb * neg_emb, dim=-1)
            loss = self.loss_func(pos_scores, neg_scores)
        elif self.loss_type == "ce":
            scores = target_emb @ self.item_emb.weight[1:].t()
            loss = F.cross_entropy(scores, next_items - 1)
        else:
            raise ValueError("Invalid loss type.")
        return loss
    
    def inference(self, his_seqs, topk):
        # his_seqs: (batch_size, seq_len)
        his_emb = self.encode_seqs(his_seqs)
        target_indices = (his_seqs != self.pad_idx).sum(dim=-1)
        target_emb = self.extract(his_emb, target_indices)

        scores = target_emb @ self.item_emb.weight[1:].t()
        _, indices = torch.topk(scores, topk, dim=-1, largest=True, sorted=True)
        return indices + 1
    
    def predict(self, his_seqs, test_items):
        # his_seqs: (batch_size, seq_len), test_items: (batch_size)
        his_emb = self.encode_seqs(his_seqs)
        target_indices = (his_seqs != self.pad_idx).sum(dim=-1)
        target_emb = self.extract(his_emb, target_indices)
        test_emb = self.item_emb(test_items)
        scores = torch.sum(target_emb * test_emb, dim=-1)
        return scores