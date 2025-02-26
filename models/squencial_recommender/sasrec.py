from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.FeedForward import FeedForward
from layers.LayerNorm import LayerNorm
from layers.MultiHeadAttention import MultiHeadAttention
from util.loss import BPRLoss, CELoss

class SASRec(nn.Module):
    def __init__(
        self, 
        n_items, 
        emb_dropout,
        max_len, 
        
        d_model, 
        n_heads, 
        attn_dropout, 
        inner_dim, 
        ffn_activation,
        ffn_dropout,
        eps, 
        num_layers,

        pad_idx=0,
        loss_type: Literal["bpr", "ce"] = "ce"
    ):
        super(SASRec, self).__init__()
        # Embedding Layer Parameters
        self.n_items = n_items            # the number of items must be equal to the number of unique items in the dataset + 1
        self.max_len = max_len
        self.emb_dropout = emb_dropout

        # Transformer Layer Parameters
        self.d_model = d_model
        self.n_heads = n_heads
        self.attn_dropout = attn_dropout
        self.inner_dim = inner_dim
        self.ffn_dropout = ffn_dropout
        self.ffn_activation = ffn_activation
        self.eps = eps
        self.num_layers = num_layers

        assert pad_idx == 0
        self.pad_idx = pad_idx

        # Embedding Layer
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=pad_idx) # zero for padding
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Transformer Layer
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.attn_norm = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.ffn_norm = nn.ModuleList()
        self.ffn = nn.ModuleList()

        for _ in range(num_layers):
            self.attn_norm.append(LayerNorm(d_model, eps=eps))
            self.attn.append(MultiHeadAttention(d_model, n_heads, attn_dropout, causal=True, q_mask=True))
            self.ffn_norm.append(LayerNorm(d_model, eps=eps))
            self.ffn.append(FeedForward(d_model, inner_dim, ffn_dropout, ffn_activation))

        self.final_norm = LayerNorm(d_model, eps=eps)    

        self.loss_type = loss_type
        self.loss_func = self.get_loss_func()

        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight)
            if module.padding_idx is not None:
                nn.init.constant_(module.weight[module.padding_idx], 0)

    def get_loss_func(self):
        if self.loss_type == "bpr":
            return BPRLoss()
        elif self.loss_type == "ce":
            return CELoss()
        else:
            raise ValueError("Invalid loss type.")    

    def encode_seqs(self, his_seqs):
        key_padding_mask = query_padding_mask = (his_seqs != self.pad_idx)

        # Embedding Layer
        item_emb = self.item_emb(his_seqs)
        item_emb = item_emb * np.sqrt(self.d_model)
        pos = (
            torch.arange(his_seqs.shape[-1])
            .unsqueeze(0)
            .repeat(his_seqs.shape[0], 1)
            .to(item_emb.device)
        )
        pos_emb = self.pos_emb(pos)
        x = item_emb + pos_emb


        # Transformer Layer
        x = self.emb_dropout(x)
        x = x * query_padding_mask.int().unsqueeze(-1)

        for i in range(self.num_layers):
            q, k, v = self.attn_norm[i](x), x, x
            x = self.attn[i](q, k, v, key_padding_mask=key_padding_mask, query_padding_mask=query_padding_mask)

            x = self.ffn_norm[i](x)
            x = self.ffn[i](x)

            x = x * query_padding_mask.int().unsqueeze(-1)

        x = self.final_norm(x)

        return x   
    
    def extract(self, his_emb, target_indices):
        assert his_emb.shape[0] == target_indices.shape[0]
        res = []
        for i in range(his_emb.shape[0]):
            res.append(his_emb[i, target_indices[i], :])
        return torch.stack(res, dim=0)
    
    def forward(self, interactions):
        # his_seqs: [batch_size, seq_len], next_items: [batch_size], next_neg_items: [batch_size]
        his_seqs = interactions["his_seqs"]
        next_items = interactions["next_items"].to(torch.long)
        next_neg_items = interactions.get("next_neg_items", None)
        if next_neg_items is not None:
            next_neg_items = next_neg_items.to(torch.long)
        target_indices = (his_seqs != self.pad_idx).sum(dim=-1) - 1
        his_emb = self.encode_seqs(his_seqs)
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
            loss = self.loss_func(scores, next_items - 1)
        else:
            raise ValueError("Invalid loss type.")
        return loss
    
    def inference(self, interactions):
        # his_seqs: [batch_size, seq_len]
        his_seqs = interactions["his_seqs"]

        target_indices = (his_seqs == self.pad_idx).sum(dim=-1) - 1
        his_emb = self.encode_seqs(his_seqs)
        target_emb = self.extract(his_emb, target_indices)
        scores = target_emb @ self.item_emb.weight[1:].t()
        return scores

    def predict(self, interactions):
        # his_seqs: [batch_size, seq_len], test_items: [batch_size]
        his_seqs = interactions["his_seqs"]
        test_items = interactions["test_items"].to(torch.long)
        target_indices = (his_seqs == self.pad_idx).sum(dim=-1) - 1
        his_emb = self.encode_seqs(his_seqs)
        target_emb = self.extract(his_emb, target_indices)
        test_emb = self.item_emb(test_items)
        scores = torch.sum(target_emb * test_emb, dim=-1)
        return scores

