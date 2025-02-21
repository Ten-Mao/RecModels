from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.loss import BPRLoss

class BPR(nn.Module):
    def __init__(
        self,
        n_items,
        n_users,

        d_model,
        loss_type: Literal["bpr"] = "bpr"

    ):
        super(BPR, self).__init__()

        # Embedding Layer Parameters
        self.n_items = n_items
        self.n_users = n_users

        self.d_model = d_model

        self.item_emb = nn.Embedding(n_items, d_model)
        self.user_emb = nn.Embedding(n_users, d_model)

        self.loss_type = loss_type
        assert self.loss_type == "bpr"
        self.loss_func = self.get_loss_func()
    
    def get_loss_func(self):
        if self.loss_type == "bpr":
            return BPRLoss()
        else:
            raise ValueError("Invalid loss type.")
    
    def forward(self, user_seqs, next_items, neg_items):
        # user_seqs: (batch_size), next_items: (batch_size), neg_items: (batch_size)
        user_emb = self.user_emb(user_seqs)
        pos_emb = self.item_emb(next_items - 1)
        neg_emb = self.item_emb(neg_items - 1)

        pos_scores = torch.sum(user_emb * pos_emb, dim=-1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=-1)

        loss = self.loss_func(pos_scores, neg_scores)

        return loss
    
    def inference(self, user_seqs, topk):
        # user_seqs: (batch_size)
        user_emb = self.user_emb(user_seqs)
        item_emb = self.item_emb.weight

        scores = user_emb @ item_emb.t()
        _, indices = torch.topk(scores, topk, dim=-1, largest=True, sorted=True)
        return indices + 1
    
    def predict(self, user_seqs, test_items):
        # user_seqs: (batch_size), test_items: (batch_size)
        user_emb = self.user_emb(user_seqs)
        test_emb = self.item_emb(test_items - 1)

        scores = torch.sum(user_emb * test_emb, dim=-1)
        return scores