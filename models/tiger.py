from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Rqvae import RQVAE

class TIGER(nn.Module):

    def __init__(
        self,

        # rqvae parameters
        
        item_emb_path: str="data/Beauty2014/Beauty2014.emb-llama-td.npy",

        in_dims: List[int]=[4096, 2048, 1024, 512, 256, 128, 64, 32],
        codebook_dim: int=32,
        codebook_sizes: List[int]=[256, 256, 256, 256],

        sinkhorn_open: bool=True,
        sinkhorn_epsilons: List[float]=[0.0, 0.0, 0.0, 0.003],
        sinkhorn_iter: int=50,

        kmeans_init_open: bool=False,

        mu: float=0.25,
    ):
        
        self.item_emb_path = item_emb_path
        self.in_dims = in_dims

        self.codebook_dim = codebook_dim
        self.codebook_sizes = codebook_sizes
        
        self.sinkhorn_open = sinkhorn_open
        self.sinkhorn_epsilons = sinkhorn_epsilons
        self.sinkhorn_iter = sinkhorn_iter

        self.kmeans_init_open = kmeans_init_open

        self.mu = mu

        self.rqvae = RQVAE(
            item_emb_path=self.item_emb_path,
            in_dims=self.in_dims,
            codebook_dim=self.codebook_dim,
            codebook_sizes=self.codebook_sizes,
            sinkhorn_open=self.sinkhorn_open,
            sinkhorn_epsilons=self.sinkhorn_epsilons,
            sinkhorn_iter=self.sinkhorn_iter,
            cf_loss_open=False,
            diversity_loss_open=False,
            kmeans_init_open=self.kmeans_init_open,
            mu=self.mu,
        )
