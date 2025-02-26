import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TFEmbedding(nn.Module):
    def __init__(
        self,

        token_field_value_num_list, 
        emb_dim,
        padding_idx=0,
    ):
        super(TFEmbedding, self).__init__()
        self.token_field_value_num_list = token_field_value_num_list
        self.num_token_fields = len(token_field_value_num_list)
        self.emb_dim = emb_dim
        self.padding_idx = padding_idx
        self.token_field_emb = [
            nn.Embedding(num + 1, emb_dim, padding_idx=padding_idx) for num in token_field_value_num_list
        ]
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight)
            if module.padding_idx is not None:
                nn.init.constant_(module.weight[module.padding_idx], 0) # Zero out the padding index


    def forward(self, x):
        # x: (batch_size, num_token_field)
        assert x.shape[-1] == self.num_token_fields, "The last dimension of x must be equal to the number of token fields"
        token_field_emb_list = [emb(x[:, i]).unsqueeze(1) for i, emb in enumerate(self.token_field_emb)]
        return torch.cat(token_field_emb_list, dim=1) # (batch_size, num_token_field, emb_dim)