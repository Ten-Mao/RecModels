from typing import List, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.loss import BPRLoss, CELoss

class Caser(nn.Module):
    def __init__(
        self,
        n_items,
        n_users,  

        d_model,
        n_horizontal_conv_per_h: int,
        n_vertical_conv: int,
        max_seq_len: int,
        activation_conv: Literal["relu", "tanh"] = "relu",
        dropout: float = 0.5,
        activation_fc: Literal["relu", "tanh"] = "relu",

        pad_idx: int = 0,
        loss_type: Literal["bpr", "ce"] = "ce",  
    ):
        super(Caser, self).__init__()
        self.n_items = n_items
        self.n_users = n_users
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=pad_idx) # zero for padding
        self.user_emb = nn.Embedding(n_users, d_model)
    
        self.n_horizontal_conv_per_h = n_horizontal_conv_per_h
        self.n_vertical_conv = n_vertical_conv
        self.max_seq_len = max_seq_len


        self.h_list = [ i + 1 for i in range(max_seq_len)]
        self.horizontal_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels=1, out_channels=n_horizontal_conv_per_h, kernel_size=(h_i, d_model))
                for h_i in self.h_list
            ]
        )
        self.activation_conv = self.get_activation(activation_conv)
        self.vertical_conv = nn.Conv2d(in_channels=1, out_channels=n_vertical_conv, kernel_size=(max_seq_len, 1))
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(n_horizontal_conv_per_h * max_seq_len + n_vertical_conv * d_model, d_model)
        self.activation_fc = self.get_activation(activation_fc)
        self.fc2 = nn.Linear(d_model * 2, d_model)


        self.loss_type = loss_type
        self.loss_func = self.get_loss_func()
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight)
            if module.padding_idx is not None:
                nn.init.constant_(module.weight[module.padding_idx], 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def get_loss_func(self):
        if self.loss_type == "bpr":
            return BPRLoss()
        elif self.loss_type == "ce":
            return CELoss()
        else:
            raise ValueError("Invalid loss type.")    

    def get_activation(self, activation: str):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        else:
            raise ValueError("Invalid activation function.")

    def encode_seqs(self, his_seqs, user_seqs):
        # his_seqs: [batch_size, seq_len], user_seqs: [batch_size]
        his_emb = self.item_emb(his_seqs)
        user_emb = self.user_emb(user_seqs)

        # horizontal convolutions
        out_horizontal = []
        for conv_i in self.horizontal_conv:
            out_conv_i = self.activation_conv(conv_i(his_emb.unsqueeze(1)).squeeze(-1))
            pool_conv_i = F.max_pool1d(out_conv_i, out_conv_i.shape[-1]).squeeze(-1)
            out_horizontal.append(pool_conv_i)
        out_horizontal = torch.cat(out_horizontal, dim=-1) # [batch_size, n_horizontal_conv_per_h * max_seq_len]

        # vertical convolution
        # [batch_size, n_vertical_conv, 1, d_model] -> [batch_size, n_vertical_conv * d_model]
        out_vertical = self.vertical_conv(his_emb.unsqueeze(1)).reshape(-1, self.n_vertical_conv * self.d_model)

        # concatenate conv and get latent vector
        out_conv = self.dropout(torch.cat([out_horizontal, out_vertical], dim=-1))
        conv_latent = self.activation_fc(self.fc1(out_conv))

        # output
        final_emb = self.fc2(torch.cat([conv_latent, user_emb], dim=-1))
        return final_emb

    def forward(self, interactions):
        # his_seqs: [batch_size, seq_len], user_seqs: [batch_size], next_items: [batch_size], next_neg_items: [batch_size]
        his_seqs = interactions["his_seqs"]
        user_seqs = interactions["user_seqs"]
        next_items = interactions["next_items"].to(torch.long)
        next_neg_items = interactions.get("next_neg_items", None)
        if next_neg_items is not None:
            next_neg_items = next_neg_items.to(torch.long)
        final_emb = self.encode_seqs(his_seqs, user_seqs)

        if self.loss_type == "bpr":
            assert next_neg_items is not None
            pos_emb = self.item_emb(next_items)
            neg_emb = self.item_emb(next_neg_items)
            pos_scores = torch.sum(final_emb * pos_emb, dim=-1)
            neg_scores = torch.sum(final_emb * neg_emb, dim=-1)
            loss = self.loss_func(pos_scores, neg_scores)
        elif self.loss_type == "ce":
            scores = final_emb @ self.item_emb.weight[1:].t()
            loss = self.loss_func(scores, next_items - 1)
        else:
            raise ValueError("Invalid loss type.")
        
        return loss
    
    def inference(self, interactions):
        # his_seqs: [batch_size, seq_len], user_seqs: [batch_size]
        his_seqs = interactions["his_seqs"]
        user_seqs = interactions["user_seqs"]

        final_emb = self.encode_seqs(his_seqs, user_seqs)
        scores = final_emb @ self.item_emb.weight[1:].t()
        return scores
    
    def predict(self, interactions):
        # his_seqs: [batch_size, seq_len], user_seqs: [batch_size], test_items: [batch_size]
        his_seqs = interactions["his_seqs"]
        user_seqs = interactions["user_seqs"]
        test_items = interactions["test_items"].to(torch.long)

        final_emb = self.encode_seqs(his_seqs, user_seqs)
        test_emb = self.item_emb(test_items)
        scores = torch.sum(final_emb * test_emb, dim=-1)
        return scores