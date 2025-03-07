from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.FeedForward import FeedForward
from layers.LayerNorm import LayerNorm
from layers.MultiHeadAttention import MultiHeadAttention
from util.loss import BPRLoss, CELoss

class Bert4Rec(nn.Module):
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
        super(Bert4Rec, self).__init__()
        # Embedding Layer Parameters
        self.n_items = n_items            # the number of items must be equal to the number of unique items in the dataset + 1
        self.max_len = max_len
        self.emb_dropout = emb_dropout

        # Transformer Layer Parameters
        self.d_model = d_model
        self.n_heads = n_heads
        self.attn_dropout = attn_dropout
        self.inner_dim = inner_dim
        self.ffn_activation = ffn_activation
        self.ffn_dropout = ffn_dropout
        self.eps = eps
        self.num_layers = num_layers

        assert pad_idx == 0
        self.pad_idx = pad_idx
        self.mask_idx = n_items + 1

        # Embedding Layer
        self.item_emb = nn.Embedding(n_items + 2, d_model, padding_idx=pad_idx) # zero for padding and last for [MASK]
        self.pos_emb = nn.Embedding(max_len + 1, d_model) # last for injecting target item in the inference phase

        # Transformer Layer
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.attn_norm = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.ffn_norm = nn.ModuleList()
        self.ffn = nn.ModuleList()

        for _ in range(num_layers):
            self.attn_norm.append(LayerNorm(d_model, eps=eps))
            self.attn.append(MultiHeadAttention(d_model, n_heads, attn_dropout, causal=False, q_mask=False))
            self.ffn_norm.append(LayerNorm(d_model, eps=eps))
            self.ffn.append(FeedForward(d_model, inner_dim, ffn_dropout, ffn_activation))

        self.final_ffn = nn.Linear(d_model, d_model)
        self.final_activation = nn.GELU()
        self.final_norm1 = LayerNorm(d_model, eps=eps)        
        self.final_norm2 = LayerNorm(d_model, eps=eps)      

        self.loss_type = loss_type
        self.loss_func = self.get_loss_func()
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight)
            if module.padding_idx is not None:
                nn.init.constant_(module.weight[module.padding_idx], 0)
        elif isinstance(module, nn.Linear):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def get_loss_func(self):
        if self.loss_type == "bpr":
            return BPRLoss()
        elif self.loss_type == "ce":
            return CELoss()
        else:
            raise ValueError("Invalid loss type.")    
        
    def encode_seqs(self, his_seqs):
        key_padding_mask = (his_seqs != self.pad_idx)

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

        for i in range(self.num_layers):
            x = self.attn_norm[i](x)
            q = k = v = x
            x = self.attn[i](q, k, v, key_padding_mask=key_padding_mask)

            x = self.ffn_norm[i](x)
            x = self.ffn[i](x)
        
        x = self.final_norm1(x)
        x = self.final_ffn(x)
        x = self.final_activation(x)
        x = self.final_norm2(x)

        return x   
    
    def extract(self, his_emb, target_indices):
        assert his_emb.shape[0] == target_indices.shape[0]
        res = []
        for i in range(his_emb.shape[0]):
            res.append(his_emb[i, target_indices[i], :])
        return torch.stack(res, dim=0)
    
    def inject(self, his_seqs):
        tgt_pad = torch.full((his_seqs.shape[0], 1), self.pad_idx, dtype=torch.long, device=his_seqs.device)
        x_pad = torch.cat([his_seqs, tgt_pad], dim=-1)
        target_indices = (his_seqs != self.pad_idx).sum(dim=-1)
        x_pad[torch.arange(his_seqs.shape[0]), target_indices] = self.mask_idx 
        return x_pad, target_indices

    def forward(self, interactions):
        # masked_his_seqs: [batch_size, seq_len], 
        # mask_indices: [batch_size, max_mask_len], 
        # mask_items: [batch_size, max_mask_len], 
        # mask_neg_items: [batch_size, max_mask_len, neg_samples]

        masked_his_seqs = interactions["masked_his_seqs"].to(torch.long)
        mask_indices = interactions["mask_indices"].to(torch.long)
        mask_items = interactions["mask_items"].to(torch.long)
        mask_neg_items = interactions.get("mask_neg_items", None)
        if mask_neg_items is not None:
            mask_neg_items = mask_neg_items.to(torch.long)

        his_emb = self.encode_seqs(masked_his_seqs)

        mask_onehot = F.one_hot(mask_indices.reshape(-1), num_classes=masked_his_seqs.shape[1]).to(his_emb.dtype)
        mask_onehot = mask_onehot.reshape(mask_indices.shape[0], mask_indices.shape[1], -1)
        # mask_onehot: [batch_size, max_mask_len, seq_len] -> [batch_size, max_mask_len, d_model]
        pred_emb = mask_onehot @ his_emb 

        if self.loss_type == "bpr":
            assert mask_neg_items is not None
            pred_emb = pred_emb.unsqueeze(-2) # [batch_size, max_mask_len, 1, d_model]
            pos_emb = self.item_emb(mask_items).unsqueeze(-2) # [batch_size, max_mask_len, 1, d_model]
            neg_emb = self.item_emb(mask_neg_items) # [batch_size, max_mask_len, neg_samples, d_model]
            pos_scores = torch.sum(pred_emb * pos_emb, dim=-1).repeat(1, 1, neg_emb.shape[-2]) # [batch_size, max_mask_len, neg_samples]
            neg_scores = torch.sum(pred_emb * neg_emb, dim=-1) # [batch_size, max_mask_len, neg_samples]
            mask = (mask_items != self.pad_idx).unsqueeze(-1).repeat(1, 1, neg_emb.shape[-2]) # [batch_size, max_mask_len, neg_samples]
            loss = self.loss_func(pos_scores, neg_scores, mask=mask)
        elif self.loss_type == "ce":
            scores = pred_emb @ self.item_emb.weight[1:-1].t() # [batch_size, max_mask_len, n_items]
            scores = scores.reshape(-1, self.n_items)
            mask_items = mask_items.reshape(-1)
            loss = self.loss_func(scores, mask_items - 1, ignore_index=self.pad_idx - 1)
        else:
            raise ValueError("Invalid loss type.")

        return loss
    
    def inference(self, interactions):
        # his_seqs: [batch_size, seq_len]
        his_seqs = interactions["his_seqs"].to(torch.long)
        his_seqs, target_indices = self.inject(his_seqs)
        his_emb = self.encode_seqs(his_seqs)
        target_emb = self.extract(his_emb, target_indices)
        scores = target_emb @ self.item_emb.weight[1:-1].t() # [batch_size, n_items]
        return scores
    
    def predict(self, interactions):
        # his_seqs: [batch_size, seq_len], test_items: [batch_size]
        his_seqs = interactions["his_seqs"].to(torch.long)
        test_items = interactions["test_items"].to(torch.long)

        his_seqs, target_indices = self.inject(his_seqs)
        his_emb = self.encode_seqs(his_seqs)
        target_emb = self.extract(his_emb, target_indices)
        test_emb = self.item_emb(test_items)
        scores = torch.sum(target_emb * test_emb, dim=-1)
        return scores

