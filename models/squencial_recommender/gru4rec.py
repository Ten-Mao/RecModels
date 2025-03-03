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

        self.pad_idx = pad_idx
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=pad_idx) # zero for padding
        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(d_model, inner_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(inner_dim, d_model)


        self.loss_type = loss_type
        self.loss_func = self.get_loss_func()
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight)
            if module.padding_idx is not None:
                nn.init.constant_(module.weight[module.padding_idx], 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

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
        # his_emb: (batch_size, seq_len, d_model), target_indices: (batch_size)
        assert his_emb.shape[0] == target_indices.shape[0]
        res = []
        for i in range(his_emb.shape[0]):
            res.append(his_emb[i, target_indices[i], :])
        return torch.stack(res, dim=0)
    
    def forward(self, interactions):
        # his_seqs: (batch_size, seq_len), next_items: (batch_size), next_neg_items: (batch_size)
        his_seqs = interactions["his_seqs"]
        next_items = interactions["next_items"].to(torch.long)
        next_neg_items = interactions.get("next_neg_items", None)
        if next_neg_items is not None:
            next_neg_items = next_neg_items.to(torch.long)
        his_emb = self.encode_seqs(his_seqs)
        target_indices = (his_seqs != self.pad_idx).sum(dim=-1) - 1
        target_emb = self.extract(his_emb, target_indices)

        if self.loss_type == "bpr":
            assert next_neg_items is not None
            pos_emb = self.item_emb(next_items)
            neg_emb = self.item_emb(next_neg_items)
            pos_scores = torch.sum(target_emb * pos_emb, dim=-1)
            neg_scores = torch.sum(target_emb * neg_emb, dim=-1)
            loss = self.loss_func(pos_scores, neg_scores)
        elif self.loss_type == "ce":
            scores = target_emb @ self.item_emb.weight[1:].t()
            loss = F.cross_entropy(scores, next_items - 1)
        else:
            raise ValueError("Invalid loss type.")
        return loss
    
    def inference(self, interactions):
        # his_seqs: (batch_size, seq_len)
        his_seqs = interactions["his_seqs"]
        his_emb = self.encode_seqs(his_seqs)
        target_indices = (his_seqs != self.pad_idx).sum(dim=-1) - 1
        target_emb = self.extract(his_emb, target_indices)

        scores = target_emb @ self.item_emb.weight[1:].t()
        return scores
    
    def predict(self, interactions):
        # his_seqs: (batch_size, seq_len), test_items: (batch_size)
        his_seqs = interactions["his_seqs"]
        test_items = interactions["test_items"].to(torch.long)
        
        his_emb = self.encode_seqs(his_seqs)
        target_indices = (his_seqs != self.pad_idx).sum(dim=-1) - 1
        target_emb = self.extract(his_emb, target_indices)
        test_emb = self.item_emb(test_items)
        scores = torch.sum(target_emb * test_emb, dim=-1)
        return scores