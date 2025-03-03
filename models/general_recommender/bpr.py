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
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight)
    
    def get_loss_func(self):
        if self.loss_type == "bpr":
            return BPRLoss()
        else:
            raise ValueError("Invalid loss type.")
    
    def forward(self, interactions):
        # users: (batch_size), items: (batch_size), neg_items: (batch_size, neg_samples)
        users = interactions["users"].to(torch.long)
        items = interactions["items"].to(torch.long)
        neg_items = interactions["neg_items"].to(torch.long)

        user_emb = self.user_emb(users)
        pos_emb = self.item_emb(items)
        neg_emb = self.item_emb(neg_items)

        pos_scores = torch.sum(user_emb * pos_emb, dim=-1).unsqueeze(-1).repeat(1, neg_emb.shape[1]) # [batch_size, neg_samples]
        neg_scores = torch.sum(user_emb * neg_emb, dim=-1) # [batch_size, neg_samples]

        loss = self.loss_func(pos_scores, neg_scores)

        return loss
    
    def inference(self, interactions):
        # users: (batch_size)
        users = interactions["users"].to(torch.long)
        user_emb = self.user_emb(users)
        item_emb = self.item_emb.weight

        scores = user_emb @ item_emb.t()
        return scores
    
    def predict(self, interactions):
        # users: (batch_size), test_items: (batch_size)
        users = interactions["users"].to(torch.long)
        test_items = interactions["test_items"].to(torch.long)
        user_emb = self.user_emb(users)
        test_emb = self.item_emb(test_items)

        scores = torch.sum(user_emb * test_emb, dim=-1)
        return scores