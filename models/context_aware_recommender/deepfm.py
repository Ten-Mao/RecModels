from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import MLPLayers
from models.context_aware_recommender.fm import FM
from util.loss import BCELoss

class DeepFM(nn.Module):
    def __init__(
        self,
        
        token_field_value_num_list,
        token_sequence_field_value_num_list,
        
        d_model,
        inner_dim,
        d_output=1,
        dropout=0.2,
        activation_fn:Literal["relu", "gelu", "sigmoid", "tanh"]="relu",
        bn=False,
        token_sequence_field_agg_method:Literal["mean", "sum", "max"]="mean",
        padding_idx=0,
        loss_type:Literal["bce"]="bce",
    ):
        super(DeepFM, self).__init__()
        self.token_field_value_num_list = token_field_value_num_list
        self.token_sequence_field_value_num_list = token_sequence_field_value_num_list
        self.token_sequence_field_agg_method = token_sequence_field_agg_method
        self.d_model = d_model
        self.inner_dim = inner_dim
        assert d_output == 1, "d_output must be 1 for DeepFM"
        self.d_output = d_output
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.padding_idx = padding_idx
        
        self.fm = FM(
            token_field_value_num_list=token_field_value_num_list,
            token_sequence_field_value_num_list=token_sequence_field_value_num_list,
            d_model=d_model,
            d_output=d_output,
            token_sequence_field_agg_method=token_sequence_field_agg_method,
            padding_idx=padding_idx,
            loss_type=loss_type,
        )

        self.dnn = MLPLayers(
            dim_list = [d_model * (len(token_field_value_num_list) + len(token_sequence_field_value_num_list)), inner_dim],
            dropout=dropout,
            activation_fn=activation_fn,
            bn=bn,
            last_activation=True,
        )

        self.dnn_output_layer = nn.Linear(inner_dim, d_output)

        assert loss_type == "bce", "DeepFM only supports BCE loss"
        self.loss_type = loss_type
        self.loss_func = self.get_loss_func()
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


    def get_loss_func(self):
        if self.loss_type == "bce":
            return BCELoss()
        else:
            raise ValueError(f"Only support BCE loss for DeepFM, but got {self.loss_type}")
    
    def encode(self, token_field_values, token_sequence_field_values):
        # token_field_values: (batch_size, num_token_field)
        # token_sequence_field_values: [(batch_size, seq_len_i), ...] the length of the list is the number of token sequence fields

        bsz = token_field_values.shape[0]

        fm_output = self.fm.encode(token_field_values, token_sequence_field_values)    # (batch_size, d_output)
        dnn_output = self.dnn(
            self.fm.second_order_embedding(
                token_field_values,
                token_sequence_field_values,
            ) # (batch_size, num_fields, d_model)
            .reshape((bsz, -1)) # (batch_size, num_fields * d_model)
        ) # (batch_size, inner_dim)
        dnn_output = self.dnn_output_layer(dnn_output) # (batch_size, d_output)
        
        output = fm_output + dnn_output

        return output

    def forward(self, interactions):
        # token_field_values: (batch_size, num_token_field)
        # token_sequence_field_values: [(batch_size, seq_len_i), ...] the length of the list is the number of token sequence fields
        token_field_values = interactions["token_field_values"].to(torch.long)
        token_sequence_field_values = [seq_field.to(torch.long) for seq_field in interactions["token_sequence_field_values"]]
        labels = interactions["labels"].to(torch.long)

        output = self.encode(token_field_values, token_sequence_field_values) # (batch_size, d_output)
        output = output.squeeze(dim=-1) # (batch_size)
        loss = self.loss_func(output, labels)
        return loss
    
    def predict(self, interactions):
        # token_field_values: (batch_size, num_token_field)
        # token_sequence_field_values: [(batch_size, seq_len_i), ...] the length of the list is the number of token sequence fields
        token_field_values = interactions["token_field_values"].to(torch.long)
        token_sequence_field_values = [seq_field.to(torch.long) for seq_field in interactions["token_sequence_field_values"]]
        
        output = self.encode(token_field_values, token_sequence_field_values) # (batch_size, d_output)
        output = output.squeeze(dim=-1)
        return F.sigmoid(output) # (batch_size)