from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.TokenFieldEmbedding import TFEmbedding
from layers.TokenSequenceFieldEmbedding import TSFEmbedding
from util.loss import BCELoss


class FM(nn.Module):
    def __init__(
        self,

        token_field_value_num_list,
        token_sequence_field_value_num_list,
        
        d_model,
        d_output=1,
        token_sequence_field_agg_method:Literal["mean", "sum", "max"]="mean",
        padding_idx=0,
        loss_type:Literal["bce"]="bce",
    ):
        super(FM, self).__init__()

        self.token_field_value_num_list = token_field_value_num_list
        self.token_sequence_field_value_num_list = token_sequence_field_value_num_list
        self.token_sequence_field_agg_method = token_sequence_field_agg_method
        self.d_model = d_model
        assert d_output == 1, "d_output must be 1 for FM"
        self.d_output = d_output
        self.padding_idx = padding_idx


        # global bias
        self.global_bias = nn.Parameter(torch.zeros(d_output), requires_grad=True)

        # first order linear
        self.token_field_emb_for_first_order = TFEmbedding(
            token_field_value_num_list=token_field_value_num_list,
            emb_dim=d_output,
            padding_idx=padding_idx,
        )

        self.token_seq_field_emb_for_first_order = TSFEmbedding(
            token_sequence_field_value_num_list=token_sequence_field_value_num_list,
            emb_dim=d_output,
            padding_idx=padding_idx,
            mode=token_sequence_field_agg_method,
        )

        # second order linear
        self.token_field_emb_for_second_order = TFEmbedding(
            token_field_value_num_list=token_field_value_num_list,
            emb_dim=d_model,
            padding_idx=padding_idx,
        )

        self.token_seq_field_emb_for_second_order = TSFEmbedding(
            token_sequence_field_value_num_list=token_sequence_field_value_num_list,
            emb_dim=d_model,
            padding_idx=padding_idx,
            mode=token_sequence_field_agg_method,
        )

        assert loss_type == "bce", "Only support BCE loss for FM"
        self.loss_type = loss_type
        self.loss_func = self.get_loss_func()

    def get_loss_func(self):
        if self.loss_type == "bce":
            return BCELoss()
        else:
            raise ValueError("Invalid loss type.")
     
    def second_order_embedding(self, token_field_values, token_sequence_field_values):
        # token_field_values: (batch_size, num_token_fields)
        # token_sequence_field_values: [(batch_size, seq_len_i), ...] the length of the list is the number of token sequence fields
        
        token_field_emb = self.token_field_emb_for_second_order(token_field_values) # (batch_size, num_token_fields, d_model)
        token_seq_field_emb = self.token_seq_field_emb_for_second_order(token_sequence_field_values) # (batch_size, num_token_seq_fields, d_model)
        return torch.cat([token_field_emb, token_seq_field_emb], dim=1) # (batch_size, num_fields, d_model)


    def second_order_linear(self, x, reduce_sum=False):
        # x: (batch_size, num_fields, emb_dim)
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        output = (square_of_sum - sum_of_square) # (batch_size, emb_dim)
        if reduce_sum:
            output = torch.sum(output, dim=-1, keepdim=True) # (batch_size, 1)
        return output

    def encode(self, token_field_values, token_sequence_field_values):
        # token_field_values: (batch_size, num_token_fields)
        # token_sequence_field_values: [(batch_size, seq_len_i), ...] the length of the list is the number of token sequence fields
        
        # first order
        first_order_output = torch.cat([
            self.token_field_emb_for_first_order(token_field_values),                   # (batch_size, num_token_fields, d_output)
            self.token_seq_field_emb_for_first_order(token_sequence_field_values),      # (batch_size, num_token_seq_fields, d_output)
        ], dim=1).sum(dim=1)                                                            # (batch_size, d_output)

        # second order
        second_order_output = self.second_order_linear(
            self.second_order_embedding(token_field_values, token_sequence_field_values),    # (batch_size, num_fields, d_model)
            reduce_sum=True
        )                                                                                    # (batch_size, 1)

        output = self.global_bias + first_order_output + second_order_output                 # (batch_size, d_output) 

        return output       

    def forward(self, token_field_values, token_sequence_field_values, labels):
        # token_field_values: (batch_size, num_token_fields)
        # token_sequence_field_values: [(batch_size, seq_len_i), ...] the length of the list is the number of token sequence fields
        # labels: (batch_size)
        
        output = self.encode(token_field_values, token_sequence_field_values) # (batch_size, d_output)
        output = output.squeeze(dim=-1) # (batch_size)
        loss = self.loss_func(output, labels)
        return loss
    
    def predict(self, token_field_values, token_sequence_field_values):
        # token_field_values: (batch_size, num_token_fields)
        # token_sequence_field_values: [(batch_size, seq_len_i), ...] the length of the list is the number of token sequence fields
        
        output = self.encode(token_field_values, token_sequence_field_values) # (batch_size, d_output)
        output = output.squeeze(dim=-1) # (batch_size)
        return F.sigmoid(output)