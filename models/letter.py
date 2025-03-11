from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Rqvae import RQVAE

class LETTER(nn.Module):

    def __init__(
        self,

        # rqvae parameters
        
        item_emb_path: str="data/Beauty2014/Beauty2014.emb-llama-td.npy",
        cf_emb_path: str="data/Beauty2014/Beauty2014-32d-sasrec.pt",

        in_dims: List[int]=[4096, 2048, 1024, 512, 256, 128, 64, 32],
        codebook_dim: int=32,
        codebook_sizes: List[int]=[256, 256, 256, 256],

        sinkhorn_open: bool=True,
        sinkhorn_epsilons: List[float]=[0.0, 0.0, 0.0, 0.003],
        sinkhorn_iter: int=50,

        kmeans_cluster: int=10,

        mu: float=0.25,
        alpha: float=0.001,
        beta: float=0.0001,
    ):
        
        self.item_emb_path = item_emb_path
        self.cf_emb_path = cf_emb_path
        self.in_dims = in_dims

        self.codebook_dim = codebook_dim
        self.codebook_sizes = codebook_sizes
        
        self.sinkhorn_open = sinkhorn_open
        self.sinkhorn_epsilons = sinkhorn_epsilons
        self.sinkhorn_iter = sinkhorn_iter

        self.kmeans_cluster = kmeans_cluster

        self.mu = mu
        self.alpha = alpha
        self.beta = beta

        self.rqvae = RQVAE(
            item_emb_path=self.item_emb_path,
            cf_emb_path=self.cf_emb_path,
            in_dims=self.in_dims,
            codebook_dim=self.codebook_dim,
            codebook_sizes=self.codebook_sizes,
            sinkhorn_open=self.sinkhorn_open,
            sinkhorn_epsilons=self.sinkhorn_epsilons,
            sinkhorn_iter=self.sinkhorn_iter,
            cf_loss_open=True,
            diversity_loss_open=True,
            kmeans_init_open=True,
            kmeans_cluster=self.kmeans_cluster,
            mu=self.mu,
            alpha=self.alpha,
            beta=self.beta,
        )