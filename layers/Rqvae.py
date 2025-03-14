from collections import defaultdict
import random
from typing import List, Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from k_means_constrained import KMeansConstrained
from layers import MLPLayers


class VectorQuantier(nn.Module):
    
    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,

        sinkhorn_open: bool=True,
        sinkhorn_epsilon: float=3e-3,
        sinkhorn_iter: int=50,

        diversity_loss_open: bool=False,
        kmeans_init_open: bool=False,
        kmeans_cluster: int=10,




        mu: float=0.25,
        beta: float=0.0001,
    ):
        super(VectorQuantier, self).__init__()

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.labels = torch.arange(codebook_size)


        self.sinkhorn_open = sinkhorn_open
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iter = sinkhorn_iter

        self.diversity_loss_open = diversity_loss_open
        self.kmeans_init_open = kmeans_init_open
        self.kmeans_cluster = kmeans_cluster


        self.mu = mu
        self.beta = beta
    
    def kmeans_init(self, x):
        # x: [N, codebook_dim]
        if not self.kmeans_init_open:
            print("kmeans_init_open is False, no need to init kmeans")
            return
        size_min = min(x.shape[0] // (self.codebook_size * 2), 50)
        try:
            clf = KMeansConstrained(
                n_clusters=self.codebook_size,
                size_min=size_min, 
                size_max=size_min * 4, 
                max_iter=10, 
                n_init=10,
                n_jobs=10, 
                verbose=False
            )
            clf.fit(x)
        except Exception as e:
            print("KMeansConstrained Error when kmeans_init: ", e)
            clf = KMeansConstrained(
                n_clusters=self.codebook_size,
                size_min=size_min, 
                size_max=size_min * 4, 
                max_iter=10, 
                n_init=10,
                n_jobs=10, 
                verbose=False
            )
            clf.fit(x)
        self.codebook.weight.data.copy_(torch.tensor(clf.cluster_centers_, dtype=torch.float32))
        self.labels = torch.tensor(clf.labels_, dtype=torch.long)

        quant, _, _ = self.inference(x)
        return quant

    def update_labels(self):
        if self.diversity_loss_open:
            size_min = min(self.codebook_size // (self.kmeans_cluster * 2), 10)
            try:
                clf = KMeansConstrained(
                    n_clusters=self.kmeans_cluster,
                    size_min=size_min, 
                    size_max=size_min * 6, 
                    max_iter=10, 
                    n_init=10,
                    n_jobs=10, 
                    verbose=False
                )
                clf.fit(self.codebook.weight)
            except Exception as e:
                print("KMeansConstrained Error when kmeans_init: ", e)
                clf = KMeansConstrained(
                    n_clusters=self.kmeans_cluster,
                    size_min=size_min, 
                    size_max=size_min * 6, 
                    max_iter=10, 
                    n_init=10,
                    n_jobs=10, 
                    verbose=False
                )
                clf.fit(self.codebook.weight)
            self.codebook.weight.data.copy_(torch.tensor(clf.cluster_centers_, dtype=torch.float32))
            self.labels = torch.tensor(clf.labels_, dtype=torch.long)


    def forward(self, x):
        # x [N, codebook_dim]
        d2term = (torch.sum(x**2, dim=-1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=-1, keepdim=True).t()
            - 2 * torch.matmul(x, self.codebook.weight.t())
        ) # [N, codebook_size]

        if self.sinkhorn_open and self.sinkhorn_epsilon > 0:
            # Sinkhorn normalization -> [-1, 1]
            max_d = torch.max(d2term, dim=-1, keepdim=True)
            min_d = torch.min(d2term, dim=-1, keepdim=True)
            mid_d = (max_d + min_d) / 2
            d2term_normed = (d2term - mid_d) / (max_d - mid_d + 1e-8)
            d2term_normed = d2term_normed.double()

            # Sinkhorn iteration 概率均衡化
            d2term_standard = torch.exp(-d2term_normed / self.sinkhorn_epsilon)
            B = d2term_standard.shape[0]
            K = d2term_standard.shape[1] 
            d2term_standard = d2term_standard / d2term_standard.sum(dim=1, keepdim=True).sum(dim=0, keepdim=True)

            for _ in range(self.sinkhorn_iter):
                # column normalization
                d2term_standard = d2term_standard / d2term_standard.sum(dim=0, keepdim=True)
                d2term_normed /= K

                # row normalization
                d2term_standard = d2term_standard / d2term_standard.sum(dim=1, keepdim=True)
                d2term_normed /= B
            
            d2term_normed *= B

            probs = d2term_normed # [batch_size, codebook_size]
            indices = torch.argmax(probs, dim=-1) # [batch_size]
        
        else:
            probs = F.softmax(-d2term, dim=-1)
            indices = torch.argmax(probs, dim=-1)
        
        quant_hard = self.codebook(indices) # [N, codebook_dim]
        quant = (quant_hard - x).detach() + x # [N, codebook_dim]

        codebook_loss = torch.mean((quant_hard - x.detach())**2, dim=-1) # [N]
        commitment_loss = torch.mean((quant_hard.detach() - x)**2, dim=-1) # [N]
        loss = codebook_loss + self.mu*commitment_loss # [N]

        if self.diversity_loss_open and self.beta > 0:
            # 计算每一个物品对应的cluster
            cluster_indices = [self.labels[idx.item()] for idx in indices]

            # 计算每一个cluster对应物品集进行码表对应后的term index list
            cluster_index_list = {}
            for idx, idx_cluster in enumerate(cluster_indices):
                cluster_index_list[idx_cluster] = cluster_index_list.get(idx_cluster, []) + [indices[idx]]
            
            # 计算每一个item对应的cluster的 term index list
            index_pos_list = [cluster_index_list[cluster] for cluster in cluster_indices]

            # 对每一个item在同一个cluster中随机采样一个不是自己的item 对应的 term index
            pos_sample = []
            for idx, pos_index_list in enumerate(index_pos_list):
                sample = random.choice(pos_index_list)
                while sample == indices[idx]:
                    sample = random.choice(pos_index_list)
                pos_sample.append(sample)
            
            pos = torch.tensor(pos_sample).to(x.device)

            sim = quant_hard @ self.codebook.weight.t() # [N, codebook_size]

            # 防止diversity loss打击自己
            sim_self = torch.zeros_like(sim)
            for idx in range(sim_self.shape[0]):
                sim_self[idx, indices[idx]] = 1e12
            sim = sim - sim_self

            diversity_loss = F.cross_entropy(sim, pos, reduction="none") # [N]

            loss += self.beta * diversity_loss # [N]

        return quant, indices, probs, loss

    @torch.no_grad()
    def inference(self, x):
        # x [N, codebook_dim]
        d2term = (torch.sum(x**2, dim=-1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=-1, keepdim=True).t()
            - 2 * torch.matmul(x, self.codebook.weight.t())
        )

        if self.sinkhorn_open and self.sinkhorn_epsilon > 0:
            # Sinkhorn normalization -> [-1, 1]
            max_d = torch.max(d2term, dim=-1, keepdim=True)
            min_d = torch.min(d2term, dim=-1, keepdim=True)
            mid_d = (max_d + min_d) / 2
            d2term_normed = (d2term - mid_d) / (max_d - mid_d + 1e-8)
            d2term_normed = d2term_normed.double()

            # Sinkhorn iteration 概率均衡化
            d2term_standard = torch.exp(-d2term_normed / self.sinkhorn_epsilon)
            B = d2term_standard.shape[0]
            K = d2term_standard.shape[1] 
            d2term_standard = d2term_standard / d2term_standard.sum(dim=1, keepdim=True).sum(dim=0, keepdim=True)

            for _ in range(self.sinkhorn_iter):
                # column normalization
                d2term_standard = d2term_standard / d2term_standard.sum(dim=0, keepdim=True)
                d2term_normed /= K

                # row normalization
                d2term_standard = d2term_standard / d2term_standard.sum(dim=1, keepdim=True)
                d2term_normed /= B
            
            d2term_normed *= B

            probs = d2term_normed # [batch_size, codebook_size]
            indices = torch.argmax(probs, dim=-1) # [batch_size]
        
        else:
            probs = F.softmax(-d2term, dim=-1)
            indices = torch.argmax(probs, dim=-1)

        quant_hard = self.codebook(indices) # [N, codebook_dim]
        quant = (quant_hard - x).detach() + x # [N, codebook_dim]

        return quant, indices, probs
            


class ResidualVectorQuantier(nn.Module):

    def __init__(
        self,
        codebook_sizes: List[int]=[256, 256, 256, 256],
        codebook_dim: int=32,

        sinkhorn_open: bool=True,
        sinkhorn_epsilons: List[float]=[0.0, 0.0, 0.0, 0.003],
        sinkhorn_iter: int=50,

        diversity_loss_open: bool=False,
        kmeans_init_open: bool=False,
        kmeans_cluster: int=10,

        mu: float=0.25,
        beta: float=0.0001,
    ):
        super(ResidualVectorQuantier, self).__init__()

        self.codebook_sizes = codebook_sizes
        self.codebook_dim = codebook_dim

        self.sinkhorn_open = sinkhorn_open
        self.sinkhorn_epsilons = sinkhorn_epsilons
        self.sinkhorn_iter = sinkhorn_iter

        self.diversity_loss_open = diversity_loss_open
        self.kmeans_init_open = kmeans_init_open
        self.kmeans_cluster = kmeans_cluster

        self.mu = mu
        self.beta = beta

        self.codebooks = nn.ModuleList(
            VectorQuantier(
                codebook_size=codebook_size_i,
                codebook_dim=self.codebook_dim,
                sinkhorn_open=self.sinkhorn_open,
                sinkhorn_epsilon=sinkhorn_epsilon_i,
                sinkhorn_iter=self.sinkhorn_iter,
                diversity_loss_open=self.diversity_loss_open,
                kmeans_init_open=self.kmeans_init_open,
                kmeans_cluster=self.kmeans_cluster,
                mu=self.mu,
                beta=self.beta
            )
            for codebook_size_i, sinkhorn_epsilon_i in zip(self.codebook_sizes, self.sinkhorn_epsilons)
        )


    def kmeans_init(self, x):
        # x: [N, codebook_dim]
        if not self.kmeans_init_open:
            print("kmeans_init_open is False, no need to init kmeans")
            return
        residual = x
        for codebook_i in self.codebooks:
            x_q = codebook_i.kmeans_init(residual)
            residual = residual - x_q

    def update_labels(self):
        if self.diversity_loss_open:
            for codebook_i in self.codebooks:
                codebook_i.update_labels()
    
    def forward(self, x):
        # x [N, codebook_dim]
        residual = x
        quant = 0
        indices_list = []
        probs_list = []
        loss_list = []
        for codebook_i in self.codebooks:
            quant_i, indices_i, probs_i, loss_i = codebook_i(residual)
            indices_list.append(indices_i)
            probs_list.append(probs_i)
            loss_list.append(loss_i)
            residual = residual - quant_i
            quant += quant_i
        
        indices = torch.stack(indices_list, dim=-1) # [N, codebook_num]
        probs = torch.stack(probs_list, dim=-1) # [N, codebook_size, codebook_num]
        loss = torch.stack(loss_list, dim=-1).mean(dim=-1) # [N]

        return quant, indices, probs, loss

    @torch.no_grad()
    def inference(self, x):
        # x [N, codebook_dim]
        residual = x
        quant = 0
        indices_list = []
        probs_list = []
        for codebook_i in self.codebooks:
            quant_i, indices_i, probs_i = codebook_i.inference(residual)
            indices_list.append(indices_i)
            probs_list.append(probs_i)
            residual = residual - quant_i
            quant += quant_i
        
        indices = torch.stack(indices_list, dim=-1) # [N, codebook_num]
        probs = torch.stack(probs_list, dim=-1) # [N, codebook_size, codebook_num]

        return quant, indices, probs



class RQVAE(nn.Module):

    def __init__(
        self,
        item_emb_path: str="data/Beauty2014/Beauty2014.emb-llama-td.npy",
        cf_emb_path: str="data/Beauty2014/Beauty2014-32d-sasrec.pt",

        in_dims: List[int]=[4096, 2048, 1024, 512, 256, 128, 64, 32],
        codebook_dim: int=32,
        codebook_sizes: List[int]=[256, 256, 256, 256],

        sinkhorn_open: bool=True,
        sinkhorn_epsilons: List[float]=[0.0, 0.0, 0.0, 0.003],
        sinkhorn_iter: int=50,

        cf_loss_open: bool=False,
        diversity_loss_open: bool=False,
        kmeans_init_open: bool=False,
        kmeans_cluster: int=10,

        mu: float=0.25,
        alpha: float=0.001,
        beta: float=0.0001,
    ):
        
        super(RQVAE, self).__init__()
        self.in_dims = in_dims
        self.codebook_dim = codebook_dim
        self.codebook_sizes = codebook_sizes

        self.sinkhorn_open = sinkhorn_open
        self.sinkhorn_epsilons = sinkhorn_epsilons
        self.sinkhorn_iter = sinkhorn_iter

        self.cf_loss_open = cf_loss_open
        self.diversity_loss_open = diversity_loss_open
        self.kmeans_init_open = kmeans_init_open
        self.kmeans_cluster = kmeans_cluster

        self.mu = mu
        self.alpha = alpha
        self.beta = beta

        self.padding_idx = 0
        self.item_emb_data = np.load(item_emb_path)
        self.item_emb_data = np.concatenate([np.zeros((1, self.item_emb_data.shape[1])), self.item_emb_data], axis=0)
        self.item_emb = nn.Embedding(self.item_emb_data.shape[0] + 1, self.in_dims[0], padding_idx=self.padding_idx)
        self.item_emb.weight.data.copy_(torch.tensor(self.item_emb_data, dtype=torch.float32))
        self.item_emb.weight.requires_grad = False

        if self.cf_loss_open:
            self.cf_emb_data = np.load(cf_emb_path)
            self.cf_emb = nn.Embedding(self.cf_emb_data.shape[0], self.in_dims[0], padding_idx=self.padding_idx)
            self.cf_emb.weight.data.copy_(torch.tensor(self.cf_emb_data, dtype=torch.float32))
            self.cf_emb.weight.requires_grad = False


        self.encoder = MLPLayers(in_dims, activation_fn="relu", last_activation=False)
        self.rq = ResidualVectorQuantier(
            codebook_sizes=self.codebook_sizes,
            codebook_dim=self.codebook_dim,
            sinkhorn_open=self.sinkhorn_open,
            sinkhorn_epsilons=self.sinkhorn_epsilons,
            sinkhorn_iter=self.sinkhorn_iter,
            diversity_loss_open=self.diversity_loss_open,
            kmeans_init_open=self.kmeans_init_open,
            kmeans_cluster=self.kmeans_cluster,
            mu=self.mu,
            beta=self.beta
        )
        self.decoder = MLPLayers(in_dims[::-1], activation_fn="relu", last_activation=False)

        self.item_indices = None

    @torch.no_grad()
    def kmeans_init(self):
        if not self.kmeans_init_open:
            print("kmeans_init_open is False, no need to init kmeans")
            return
        x_in = self.encoder(self.item_emb.weight[1:]) # [N, in_dims[-1]]
        self.rq.kmeans_init(x_in)

    @torch.no_grad()
    def update_labels(self):
        if self.diversity_loss_open:
            self.rq.update_labels()

    @torch.no_grad()
    def compute_unique_key_ratio(self):
        x_in = self.encoder(self.item_emb.weight[1:]) # [N, in_dims[-1]]
        _, item_indices, _ = self.rq.inference(x_in)
        item_indices = item_indices.cpu().numpy()
        inverse_map = defaultdict(list)
        for i, indices in enumerate(item_indices):
            key = ",".join(map(str, indices))
            inverse_map[key].append(i)
        print(
            f"Unique key number: {len(inverse_map)}, ratio: {len(inverse_map)/(len(item_indices))}."
        )

    @torch.no_grad()
    def get_all_indices(self):
        return self.item_indices
    
    def set_all_indices(self):

        def get_collision_item(all_indices_str):
            indexstr2id = {}
            for i, indexstr in enumerate(all_indices_str):
                if indexstr not in indexstr2id:
                    indexstr2id[indexstr] = []
                indexstr2id[index].append(i + 1)

            collision_item_groups = []

            for index in indexstr2id:
                if len(indexstr2id[index]) > 1:
                    collision_item_groups.append(indexstr2id[index])

            return collision_item_groups



        x_in = self.encoder(self.item_emb.weight[1:])
        _, item_indices, _ = self.rq.inference(x_in)
        item_indices = item_indices.cpu().numpy() # [N, codebook_num]
        prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>","<f_{}>"]
        item_indices_str = []
        for i, indices in enumerate(item_indices):
            item_indices_str.append(",".join([prefix[j].format(indices[j]) for j in range(len(indices))]))
        
        sinkhorn_open = self.sinkhorn_open
        sinkhorn_epsilons = self.sinkhorn_epsilons

        self.sinkhorn_epsilons = [0.0, 0.0, 0.0, 0.003]
        for i, vq in  enumerate(self.rq.codebooks):
            vq.sinkhorn_open = True
            vq.sinkhorn_epsilon = self.sinkhorn_epsilons[i]
        
        iter_num = 20
        while iter_num > 0 and len(item_indices) != len(set(item_indices_str)):
            collision_item_groups = get_collision_item(item_indices_str)
            for collision_group in collision_item_groups:
                _, new_indices, _ = self.inference(torch.tensor(collision_group, dtype=torch.long, device=x_in.device))[1]
                new_indices = new_indices.cpu().numpy()
                for i, new_index in enumerate(new_indices):
                    item_indices_str[collision_group[i] - 1] = ",".join([prefix[j].format(new_index[j]) for j in range(len(new_index))])
                    item_indices[collision_group[i] - 1] = new_index
            iter_num -= 1
        
        self.sinkhorn_epsilons = sinkhorn_epsilons
        for i, vq in  enumerate(self.rq.codebooks):
            vq.sinkhorn_open = sinkhorn_open
            vq.sinkhorn_epsilon = sinkhorn_epsilons[i]
        
        self.item_indices = torch.tensor(item_indices, dtype=torch.long, device=x_in.device)
        self.item_indices = torch.cat([torch.zeros((1, self.item_indices.shape[1]), dtype=torch.long, device=self.item_indices.device), self.item_indices], dim=0) # [N+1, codebook_num]
        
        print(
            f"Unique key number: {len(set(item_indices_str))}, ratio: {len(set(item_indices_str))/(len(item_indices) - 1)}."
        )

    def forward(self, x):
        # x: (batch_size) or (batch_size, seq_len)
        shape_len = len(x.shape)
        assert shape_len == 1 or shape_len == 2, "x shape must be (batch_size) or (batch_size, seq_len)"
        if shape_len == 2:
            bsz, seq_len = x.shape
            x_reshape = x.reshape(-1)
        else:
            x_reshape = x

        mask = (x_reshape != self.padding_idx).float() # [N]

        x_emb = self.item_emb(x_reshape) # [N, in_dims[0]]
        x_in = self.encoder(x_emb) # [N, in_dims[-1]]
        quant, indices, probs, quant_loss = self.rq(x_in)
        x_out = self.decoder(quant) # [N, in_dims[0]]

        recon_loss = torch.mean((x_out - x_emb)**2, dim=-1) * mask
        recon_loss = torch.sum(recon_loss) / torch.sum(mask)

        quant_loss = quant_loss * mask
        quant_loss = torch.sum(quant_loss) / torch.sum(mask)

        loss = recon_loss + quant_loss

        if self.cf_loss_open and self.alpha > 0:
            x_cf_emb = self.cf_emb(x_reshape) # [N, in_dims[-1]]
            sim = quant @ x_cf_emb.t() # [N, N]
            label = torch.arange(quant.shape[0])
            cf_loss = F.cross_entropy(sim, label, reduction="none") # [N]
            cf_loss = cf_loss * mask
            cf_loss = torch.sum(cf_loss) / torch.sum(mask)
            loss += self.alpha * cf_loss
        else:
            cf_loss = torch.tensor(0.0)
            
        if shape_len == 2:
            quant = quant.reshape(bsz, seq_len, -1) # [bsz, seq_len, codebook_dim]
            indices = indices.reshape(bsz, seq_len, -1) # [bsz, seq_len, codebook_num]
            probs = probs.reshape(bsz, seq_len, probs.shape[-2], probs.shape[-1]) # [bsz, seq_len, codebook_size, codebook_num]
        return quant, indices, probs, loss, recon_loss, quant_loss, cf_loss
    
    @torch.no_grad()
    def inference(self, x):
        # x: (batch_size) or (batch_size, seq_len)
        shape_len = len(x.shape)
        assert shape_len == 1 or shape_len == 2, "x shape must be (batch_size) or (batch_size, seq_len)"
        if shape_len == 2:
            bsz, seq_len = x.shape
            x = x.reshape(-1)

        x_emb = self.item_emb(x) # [N, in_dims[0]]
        x_in = self.encoder(x_emb) # [N, in_dims[-1]]
        quant, indices, probs = self.rq.inference(x_in)
        
        if shape_len == 2:
            quant = quant.reshape(bsz, seq_len, -1) # [bsz, seq_len, codebook_dim]
            indices = indices.reshape(bsz, seq_len, -1) # [bsz, seq_len, codebook_num]
            probs = probs.reshape(bsz, seq_len, probs.shape[-2], probs.shape[-1]) # [bsz, seq_len, codebook_size, codebook_num]

        return quant, indices, probs
