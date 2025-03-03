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
        pool_type: Literal["max", "avg"] = "avg",

        pad_idx: int = 0,
        loss_type: Literal["bpr", "ce"] = "bpr",
    ):
        super(HGN, self).__init__()

        # Embedding Layer Parameters
        self.n_items = n_items
        self.n_users = n_users
        self.d_model = d_model

        self.pad_idx = pad_idx
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=pad_idx) # zero for padding
        self.user_emb = nn.Embedding(n_users, d_model)

        # features gating
        self.w1 = nn.Linear(d_model, d_model, bias=False)
        self.w2 = nn.Linear(d_model, d_model, bias=False)
        self.b = nn.Parameter(torch.zeros(d_model), requires_grad=True)
        self.activation1 = nn.Sigmoid()

        # instance gating
        self.w3 = nn.Linear(d_model, d_model, bias=False)
        self.w4 = nn.Linear(d_model, d_model, bias=False)
        self.activation2 = nn.Sigmoid()


        self.pool_type = pool_type

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
        elif isinstance(module, nn.Parameter):
            nn.init.constant_(module, 0)

    def get_loss_func(self):
        if self.loss_type == "bpr":
            return BPRLoss()
        elif self.loss_type == "ce":
            return CELoss()
        else:
            raise ValueError("Invalid loss type")
    
    def features_gating(self, item_emb, user_emb):
        # item_emb: (batch_size, max_len, d_model), user_emb: (batch_size, d_model)

        gate = self.activation1(self.w1(item_emb) + self.w2(user_emb).unsqueeze(1) + self.b)
        
        return item_emb * gate
    
    def instance_gating(self, item_emb, user_emb):
        # item_emb: (batch_size, max_len, d_model), user_emb: (batch_size, d_model)

        gate = self.activation2(self.w3(item_emb) + self.w4(user_emb).unsqueeze(1))
        
        return item_emb * gate
 
    def padding_pool(self, gated_emb, padding_mask):
        # gated_emb: (batch_size, d_model, max_len), padding_mask: (batch_size, max_len, 1)
        if self.pool_type == "max":
            gated_emb = torch.where(padding_mask, gated_emb.permute(0, 2, 1), torch.tensor(-float("inf")).to(gated_emb.device))
            pooled_emb = gated_emb.max(dim=-1)
        elif self.pool_type == "avg":
            gated_emb = torch.where(padding_mask, gated_emb.permute(0, 2, 1), torch.tensor(0.).to(gated_emb.device))
            pooled_emb = gated_emb.sum(dim=-1) / padding_mask.sum(dim=-1).unsqueeze(1)
        else:
            raise ValueError("Invalid pool type")
        return pooled_emb

    def encode_seqs(self, his_seqs, user_seqs):
        # his_seqs: (batch_size, max_len), user_seqs: (batch_size)

        # Item Embedding
        his_emb = self.item_emb(his_seqs)
        user_emb = self.user_emb(user_seqs)

        features_gated_emb = self.features_gating(his_emb, user_emb)
        instance_gated_emb = self.instance_gating(features_gated_emb, user_emb)

        # Pooling
        instance_gated_emb = instance_gated_emb.permute(0, 2, 1)
        padding_mask = (his_seqs != self.pad_idx).unsqueeze(-1)
        
        pooled_emb = self.padding_pool(instance_gated_emb, padding_mask)
        
        his_embs = torch.where(padding_mask, his_emb, torch.tensor(0.).to(his_emb.device))

        final_emb = user_emb + pooled_emb + his_embs.sum(dim=1)

        return final_emb

    def forward(self, interactions):
        # his_seqs: (batch_size, max_len), 
        # user_seqs: (batch_size), 
        # next_items: (batch_size), 
        # next_neg_items: (batch_size, neg_samples)
        his_seqs = interactions["his_seqs"].to(torch.long)
        user_seqs = interactions["user_seqs"].to(torch.long)
        next_items = interactions["next_items"].to(torch.long)
        next_neg_items = interactions.get("next_neg_items", None)
        if next_neg_items is not None:
            next_neg_items = next_neg_items.to(torch.long)

        final_emb = self.encode_seqs(his_seqs, user_seqs)
        if self.loss_type == "bpr":
            assert next_neg_items is not None
            final_emb = final_emb.unsqueeze(1) # [batch_size, 1, d_model]
            pos_emb = self.item_emb(next_items).unsqueeze(1) # [batch_size, 1, d_model]
            neg_emb = self.item_emb(next_neg_items) # [batch_size, neg_samples, d_model]
            pos_scores = torch.sum(final_emb * pos_emb, dim=-1).repeat(1, neg_emb.shape[1]) # [batch_size, neg_samples]
            neg_scores = torch.sum(final_emb * neg_emb, dim=-1) # [batch_size, neg_samples]
            loss = self.loss_func(pos_scores, neg_scores)
        elif self.loss_type == "ce":
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