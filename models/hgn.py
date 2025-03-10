from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.loss import BPRLoss, CELoss

class HGN(nn.Module):
    def __init__(
        self,
        n_items,
        n_users,

        d_model,
        max_len,
        dropout=0.1,
        pool_type: Literal["max", "avg"] = "avg",
    ):
        super(HGN, self).__init__()

        # Embedding Layer Parameters
        self.n_items = n_items
        self.n_users = n_users
        self.d_model = d_model

        self.pad_idx = 0
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=self.pad_idx) # zero for padding
        self.user_emb = nn.Embedding(n_users, d_model)

        # features gating
        self.w1 = nn.Linear(d_model, d_model, bias=False)
        self.w2 = nn.Linear(d_model, d_model, bias=False)
        self.b = nn.Parameter(torch.zeros(d_model), requires_grad=True)
        self.activation1 = nn.Sigmoid()

        # instance gating
        self.w3 = nn.Linear(d_model, 1, bias=False)
        self.max_len = max_len
        self.w4 = nn.Linear(d_model, self.max_len, bias=False)
        self.activation2 = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout)
        self.pool_type = pool_type

        self.loss_func = nn.CrossEntropyLoss()
    
    def features_gating(self, item_emb, user_emb):
        # item_emb: (batch_size, max_len, d_model), user_emb: (batch_size, d_model)

        gate = self.activation1(self.w1(item_emb) + self.w2(user_emb).unsqueeze(1) + self.b)
        return item_emb * gate
    
    def instance_gating(self, item_emb, user_emb):
        # item_emb: (batch_size, max_len, d_model), user_emb: (batch_size, d_model)

        gate = self.activation2(self.w3(item_emb).squeeze(-1) + self.w4(user_emb))
        return item_emb * (gate.unsqueeze(-1))
 
    def padding_pool(self, gated_emb, padding_mask):
        # gated_emb: (batch_size, max_len, d_model), padding_mask: (batch_size, max_len, 1)
        if self.pool_type == "max":
            gated_emb = gated_emb.masked_fill(~padding_mask, float("-inf"))  
            gated_emb = gated_emb.permute(0, 2, 1)
            pooled_emb = gated_emb.max(dim=-1)
        elif self.pool_type == "avg":
            gated_emb = gated_emb.masked_fill(~padding_mask, 0.)
            gated_emb = gated_emb.permute(0, 2, 1)
            pooled_emb = gated_emb.sum(dim=-1) / padding_mask.squeeze(-1).sum(dim=-1).unsqueeze(-1)
        else:
            raise ValueError("Invalid pool type")
        return pooled_emb

    def encode_seqs(self, his_seqs, user_seqs):
        # his_seqs: (batch_size, max_len), user_seqs: (batch_size)

        # Item Embedding
        his_emb = self.item_emb(his_seqs)   # [batch_size, max_len, d_model]
        user_emb = self.user_emb(user_seqs) # [batch_size, d_model]

        features_gated_emb = self.features_gating(his_emb, user_emb) # [batch_size, max_len, d_model]
        instance_gated_emb = self.instance_gating(features_gated_emb, user_emb) # [batch_size, max_len, d_model]

        # Pooling
        padding_mask = (his_seqs != self.pad_idx).unsqueeze(-1) # [batch_size, max_len, 1]
        
        pooled_emb = self.padding_pool(instance_gated_emb, padding_mask) # [batch_size, d_model]
        
        his_embs = his_emb.masked_fill(~padding_mask, 0.)

        final_emb = self.dropout(user_emb + pooled_emb + his_embs.sum(dim=1))
        return final_emb

    def forward(self, interactions):
        # his_seqs: (batch_size, max_len), 
        # user_seqs: (batch_size), 
        # next_items: (batch_size)

        his_seqs = interactions["his_seqs"].to(torch.long)
        user_seqs = interactions["user_seqs"].to(torch.long)
        next_items = interactions["next_items"].to(torch.long)

        final_emb = self.encode_seqs(his_seqs, user_seqs) # [batch_size, d_model]

        scores = final_emb @ self.item_emb.weight[1:].t()
        loss = self.loss_func(scores, next_items - 1)
        
        return loss
    
    def inference(self, interactions):
        # his_seqs: (batch_size, max_len), user_seqs: (batch_size)
        his_seqs = interactions["his_seqs"].to(torch.long)
        user_seqs = interactions["user_seqs"].to(torch.long)

        final_emb = self.encode_seqs(his_seqs, user_seqs)
        scores = final_emb @ self.item_emb.weight[1:].t()
        return scores
    
    def predict(self, interactions):
        # his_seqs: (batch_size, max_len), user_seqs: (batch_size), test_items: (batch_size)
        his_seqs = interactions["his_seqs"].to(torch.long)
        user_seqs = interactions["user_seqs"].to(torch.long)
        test_items = interactions["test_items"].to(torch.long)
        
        final_emb = self.encode_seqs(his_seqs, user_seqs)
        test_emb = self.item_emb(test_items)
        scores = torch.sum(final_emb * test_emb, dim=-1)

        return scores