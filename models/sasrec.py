import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.feedforward import FeedForward
from layers.layernorm import LayerNorm
from layers.multiheadattention import MultiHeadAttention

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

        pad_idx=0
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

    def encode_seqs(self, x):
        key_padding_mask = query_padding_mask = (x != self.pad_idx)

        # Embedding Layer
        item_emb = self.item_emb(x)
        item_emb = item_emb * np.sqrt(self.d_model)
        pos = (
            torch.arange(x.shape[-1])
            .unsqueeze(0)
            .repeat(x.shape[0], 1)
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
    
    def forward(self, x, next_items):
        # x: [batch_size, seq_len], next_items: [batch_size]
        target_indices = (x != self.pad_idx).sum(dim=-1) - 1
        his_emb = self.encode_seqs(x)
        target_emb = self.extract(his_emb, target_indices)
        scores = target_emb @ self.item_emb.weight[1:].t()
        loss = F.cross_entropy(scores, next_items - 1)
        return loss
    
    def inference(self, x, topk):
        target_indices = (x == self.pad_idx).sum(dim=-1) - 1
        his_emb = self.encode_seqs(x)
        target_emb = self.extract(his_emb, target_indices)
        scores = target_emb @ self.item_emb.weight[1:].t()
        _, indices = torch.topk(scores, topk, dim=-1, largest=True, sorted=True)
        return indices + 1

