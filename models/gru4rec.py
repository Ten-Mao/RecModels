import torch
import torch.nn as nn
import torch.nn.functional as F

class Gru4Rec(nn.Module):
    def __init__(
        self, 
        n_items,

        d_model,
        inner_dim,
        num_layers,
        dropout,
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
    
    def encode_seqs(self, x):
        # x: (batch_size, seq_len)
        x = self.item_emb(x) # (batch_size, seq_len, d_model)
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
    
    def forward(self, x, next_items):
        # x: (batch_size, seq_len), next_items: (batch_size)
        his_emb = self.encode_seqs(x)
        target_indices = (x != self.pad_idx).sum(dim=-1) - 1
        target_emb = self.extract(his_emb, target_indices)

        scores = target_emb @ self.item_emb.weight[1:].t()
        loss = F.cross_entropy(scores, next_items - 1)
        return loss
    
    def inference(self, x, topk):
        # x: (batch_size, seq_len)
        his_emb = self.encode_seqs(x)
        target_indices = (x != self.pad_idx).sum(dim=-1)
        target_emb = self.extract(his_emb, target_indices)

        scores = target_emb @ self.item_emb.weight[1:].t()
        _, indices = torch.topk(scores, topk, dim=-1, largest=True, sorted=True)
        return indices + 1