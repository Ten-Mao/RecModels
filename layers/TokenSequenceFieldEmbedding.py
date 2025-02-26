from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

class TSFEmbedding(nn.Module):
    def __init__(
        self,

        token_sequence_field_value_num_list,
        emb_dim,
        padding_idx=0,
        mode:Literal["mean", "sum", "max"]="mean"
    ):
        super(TSFEmbedding, self).__init__()
        self.token_sequence_field_value_num_list = token_sequence_field_value_num_list
        self.num_token_sequence_fields = len(token_sequence_field_value_num_list)
        self.emb_dim = emb_dim
        self.padding_idx = padding_idx
        self.token_sequence_field_emb = [
            nn.Embedding(num + 1, emb_dim, padding_idx=padding_idx) for num in token_sequence_field_value_num_list
        ]
        self.mode = mode
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight)
            if module.padding_idx is not None:
                nn.init.constant_(module.weight[module.padding_idx], 0)
    
    def forward(self, x):
        # x: [(batch_size, seq_len_i), ...] len(x) == num_token_sequence_fields
        if x is None:
            return None
        assert len(x) == self.num_token_sequence_fields, "The length of x must be equal to the number of token sequence fields"
        emb_list = []
        for i, seq in enumerate(x):
            seq_emb = self.token_sequence_field_emb[i](seq)         # (batch_size, seq_len_i, emb_dim)
            padding_mask = (seq != self.padding_idx).unsqueeze(-1)  # (batch_size, seq_len_i, 1)
            if self.mode == "mean":
                seq_emb = torch.where(padding_mask, seq_emb, torch.tensor(0.0).to(seq_emb.device))
                seq_emb = seq_emb.sum(dim=1) / padding_mask.sum(dim=1).float()
            elif self.mode == "sum":
                seq_emb = torch.where(padding_mask, seq_emb, torch.tensor(0.0).to(seq_emb.device))
                seq_emb = seq_emb.sum(dim=1)
            elif self.mode == "max":
                seq_emb = torch.where(padding_mask, seq_emb, torch.tensor(-float("inf")).to(seq_emb.device))
                seq_emb = seq_emb.max(dim=1)
            else:
                raise ValueError("Invalid mode")
            emb_list.append(seq_emb.unsqueeze(1))
        return torch.cat(emb_list, dim=1) # (batch_size, num_token_sequence_fields, emb_dim)
        