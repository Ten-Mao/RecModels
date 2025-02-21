import torch
import torch.nn as nn
import torch.nn.functional as F

class FSFEmbedding(nn.Module):
    def __init__(
        self,
    ):
        super(FSFEmbedding, self).__init__()