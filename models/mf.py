from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

class MF(nn.Module):
    def __init__(
        self,
        n_items,
        n_users,

        d_model,
        dropout,

    ):
        super(MF, self).__init__()

        # Embedding Layer Parameters
        self.n_items = n_items
        self.n_users = n_users

        self.d_model = d_model

        self.pad_idx = 0
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=self.pad_idx) # zero for padding
        self.user_emb = nn.Embedding(n_users, d_model)

        self.dropout = nn.Dropout(dropout)
        self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, interactions):
        # users: (batch_size), his_seqs: (batch_size, seq_len), next_items: (batch_size)
        users = interactions["users"].to(torch.long)
        his_seqs = interactions["his_seqs"].to(torch.long)
        targets = interactions["next_items"].to(torch.long)

        user_emb = self.user_emb(users) # [batch_size, d_model]
        his_emb = self.item_emb(his_seqs).sum(dim=1) / (his_seqs != 0).sum(dim=1, keepdim=True) # [batch_size, d_model] 

        pred_emb = self.dropout(user_emb + his_emb)

        scores = pred_emb @ self.item_emb.weight[1:].t()
        loss = self.loss_func(scores, targets - 1)

        return loss
    
    def inference(self, interactions):
        # users: (batch_size), his_seqs: (batch_size, seq_len)
        users = interactions["users"].to(torch.long)
        his_seqs = interactions["his_seqs"].to(torch.long)

        user_emb = self.user_emb(users) # [batch_size, d_model]
        his_emb = self.item_emb(his_seqs).sum(dim=1) / (his_seqs != 0).sum(dim=1, keepdim=True) # [batch_size, d_model] 

        pred_emb = self.dropout(user_emb + his_emb)

        scores = pred_emb @ self.item_emb.weight[1:].t()
        return scores
    
    def predict(self, interactions):
        # users: (batch_size), his_seqs: (batch_size, seq_len), test_items: (batch_size)
        users = interactions["users"].to(torch.long)
        his_seqs = interactions["his_seqs"].to(torch.long)
        test_items = interactions["test_items"].to(torch.long)

        user_emb = self.user_emb(users) # [batch_size, d_model]
        his_emb = self.item_emb(his_seqs).sum(dim=1) / (his_seqs != 0).sum(dim=1, keepdim=True) # [batch_size, d_model] 
        pred_emb = self.dropout(user_emb + his_emb)
        test_emb = self.item_emb(test_items)

        scores = torch.sum(pred_emb * test_emb, dim=-1)
        return scores