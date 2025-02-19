from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from util.loss import BPRLoss, CELoss

class LightGCN(nn.Module):
    def __init__(
        self,
        n_items,
        n_users, 

        d_model,
        interaction_matrix: torch.sparse_coo_tensor,
        num_layers,
        eps=1e-7,

        loss_type: Literal["bpr", "ce"] = "bpr"
    ):
        super(LightGCN, self).__init__()
        # Embedding Layer Parameters
        self.n_items = n_items
        self.n_users = n_users
        self.d_model = d_model

        self.item_emb = nn.Embedding(n_items, d_model)
        self.user_emb = nn.Embedding(n_users, d_model)

        self.norm_adjaency_matrix = self.normalize_adjacency_matrix(interaction_matrix, eps)

        self.num_layers = num_layers
        
        self.loss_type = loss_type
        assert self.loss_type == "bpr" "the only loss type supported by LightGCN is BPR"
        self.loss_func = self.get_loss_func()

    def get_loss_func(self):
        if self.loss_type == "bpr":
            return BPRLoss()
        elif self.loss_type == "ce":
            return CELoss()
        else:
            raise ValueError("Invalid loss type.")    
        
    def normalize_adjacency_matrix(self, adj, eps):
        # adj: (n_users, n_items)
        indices = adj.indices()     # (2, nnz)
        values = adj.values()       # (nnz,)

        indices1 = indices.clone()
        indices1[1] = indices[1] + self.n_users

        indices2 = indices.clone()
        indices2 = indices2[[1, 0]]
        indices2[0] = indices2[0] + self.n_users

        indices_A = torch.cat([indices1, indices2], dim=1)
        values_A = torch.cat([values, values], dim=0)
        size_A = torch.Size([self.n_users + self.n_items, self.n_users + self.n_items])

        A = sp.coo_matrix(
            (values_A.cpu().numpy(), indices_A.cpu().numpy()),
            shape=(self.n_users + self.n_items, self.n_users + self.n_items)
        )
        sumArr = (A > 0).sum(axis=1)
        D = np.array(sumArr.flatten())[0] + eps
        D = sp.diags(np.power(D, -0.5))
        L = D @ A @ D

        L = torch.sparse_coo_tensor(
            indices=indices_A,
            values=torch.tensor(L.data),
            size=size_A
        )

        return L

    def get_all_embeddings(self):
        return torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
    
    def encode_users_items(self):
        all_embeddings = self.get_all_embeddings()
        embedding_list = [all_embeddings]
        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(self.norm_adjaency_matrix, all_embeddings)
            embedding_list.append(all_embeddings)
        
        all_embeddings = torch.stack(embedding_list, dim=1)
        all_embeddings = all_embeddings.mean(dim=1)
        user_latent, item_latent = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return user_latent, item_latent
    
    def forward(self, user, pos_item, neg_item):
        # user: (batch_size), pos_item: (batch_size), neg_item: (batch_size)
        user_latent, item_latent = self.encode_users_items()

        tgt_user_latent = user_latent[user]
        pos_item_latent = item_latent[pos_item-1]
        neg_item_latent = item_latent[neg_item-1]

        pos_scores = torch.sum(tgt_user_latent * pos_item_latent, dim=-1)
        neg_scores = torch.sum(tgt_user_latent * neg_item_latent, dim=-1)

        loss = self.loss_func(pos_scores, neg_scores)
        return loss

    def inference(self, user, topk):
        # user: (batch_size)
        user_latent, item_latent = self.encode_users_items()
        user_latent = user_latent[user]
        scores = torch.matmul(user_latent, item_latent.t())
        _, indices = torch.topk(scores, topk, dim=-1, largest=True, sorted=True)
        return indices + 1

        
