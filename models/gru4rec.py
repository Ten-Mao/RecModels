from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

class Gru4Rec(nn.Module):
    def __init__(
        self, 
        n_items,
        emb_dropout,
        d_model,
        inner_dim,
        num_layers,
        gru_dropout,
    ):
        super(Gru4Rec, self).__init__()
        self.n_items = n_items
        self.d_model = d_model
        self.inner_dim = inner_dim
        self.num_layers = num_layers

        self.pad_idx = 0
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=self.pad_idx) # zero for padding
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.gru = nn.GRU(d_model, inner_dim, num_layers, dropout=gru_dropout, batch_first=True)
        self.fc = nn.Linear(inner_dim, d_model)
        self.loss_func = nn.CrossEntropyLoss()

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
        # his_seqs: (batch_size, seq_len), next_items: (batch_size)
        his_seqs = interactions["his_seqs"].to(torch.long)
        next_items = interactions["next_items"].to(torch.long)

        his_emb = self.encode_seqs(his_seqs)
        target_indices = (his_seqs != self.pad_idx).sum(dim=-1) - 1
        target_emb = self.extract(his_emb, target_indices)

        scores = target_emb @ self.item_emb.weight[1:].t()
        loss = F.cross_entropy(scores, next_items - 1)

        return loss
    
    def inference(self, interactions):
        # his_seqs: (batch_size, seq_len)
        his_seqs = interactions["his_seqs"].to(torch.long)
        his_emb = self.encode_seqs(his_seqs)
        target_indices = (his_seqs != self.pad_idx).sum(dim=-1) - 1
        target_emb = self.extract(his_emb, target_indices)

        scores = target_emb @ self.item_emb.weight[1:].t()
        return scores
    
    def predict(self, interactions):
        # his_seqs: (batch_size, seq_len), test_items: (batch_size)
        his_seqs = interactions["his_seqs"].to(torch.long)
        test_items = interactions["test_items"].to(torch.long)
        
        his_emb = self.encode_seqs(his_seqs)
        target_indices = (his_seqs != self.pad_idx).sum(dim=-1) - 1
        target_emb = self.extract(his_emb, target_indices)
        test_emb = self.item_emb(test_items)
        scores = torch.sum(target_emb * test_emb, dim=-1)
        return scores