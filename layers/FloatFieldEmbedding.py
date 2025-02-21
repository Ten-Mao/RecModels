import torch
import torch.nn as nn
import torch.nn.functional as F

class FFEmbedding(nn.Module):
    def __init__(
        self,
    ):
        super(FFEmbedding, self).__init__()