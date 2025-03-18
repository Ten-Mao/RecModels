from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

class LightGCN(nn.Module):
    def __init__(
        self,
        n_items,
        n_users, 

        d_model,
        interaction_matrix: torch.sparse_coo_tensor,
        num_layers,
        dropout=0.1,
    ):
        super(LightGCN, self).__init__()
        # Embedding Layer Parameters
        self.n_items = n_items
        self.n_users = n_users
        self.d_model = d_model

        self.pad_idx = 0
        self.item_emb = nn.Embedding(1 + n_items, d_model, padding_idx=self.pad_idx)
        self.user_emb = nn.Embedding(n_users, d_model)

        self.norm_adjaency_matrix = self.normalize_adjacency_matrix(interaction_matrix)

        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        
        self.loss_func = nn.CrossEntropyLoss()
        
    def normalize_adjacency_matrix(self, adj):
        # adj: sp.csrmatrix, shape: [n_users + n_items, n_users + n_items]
        coo = adj.tocoo().astype(np.float32)
        row = torch.tensor(coo.row, dtype=torch.long)
        col = torch.tensor(coo.col, dtype=torch.long)
        index = torch.stack([row, col], dim=0)
        data = torch.tensor(coo.data, dtype=torch.float32)
        L = torch.sparse_coo_tensor(
            indices=index,
            values=data,
            size=torch.Size(coo.shape),
        )

        return L

    def get_all_embeddings(self):
        return torch.cat([self.user_emb.weight, self.item_emb.weight[1:]], dim=0)
    
    def encode_users_items(self):
        all_embeddings = self.get_all_embeddings()
        embedding_list = [all_embeddings]
        self.norm_adjaency_matrix = self.norm_adjaency_matrix.to(all_embeddings.device)
        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(self.norm_adjaency_matrix, all_embeddings)
            embedding_list.append(all_embeddings)
        
        all_embeddings = torch.stack(embedding_list, dim=1)
        all_embeddings = all_embeddings.mean(dim=1)
        user_latent, item_latent = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return user_latent, item_latent
    
    def forward(self, interactions):
        # user_seqs: (batch_size), his_seqs: [batch_size, seq_len], next_items: [batch_size]
        user_seqs = interactions["user_seqs"].to(torch.long)
        his_seqs = interactions["his_seqs"].to(torch.long)
        next_items = interactions["next_items"].to(torch.long)

        user_latent, item_latent = self.encode_users_items()
        his_embed = self.item_emb(his_seqs).sum(dim=1) / (his_seqs != self.pad_idx).sum(dim=1, keepdim=True).float()
        user_embed = user_latent[user_seqs]
        pred_embed = self.dropout(user_embed + his_embed)

        scores = pred_embed @ item_latent.t()
        loss = self.loss_func(scores, next_items - 1)
        return loss

    def inference(self, interactions):
        # user_seqs: (batch_size), his_seqs: [batch_size, seq_len]
        user_seqs = interactions["user_seqs"].to(torch.long)
        his_seqs = interactions["his_seqs"].to(torch.long)

        user_latent, item_latent = self.encode_users_items()
        his_embed = self.item_emb(his_seqs).sum(dim=1) / (his_seqs != self.pad_idx).sum(dim=1, keepdim=True).float()
        user_embed = user_latent[user_seqs]
        pred_embed = self.dropout(user_embed + his_embed)

        scores = pred_embed @ item_latent.t()
        return scores
    

        
