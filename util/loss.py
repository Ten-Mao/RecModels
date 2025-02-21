from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma
    
    def forward(self, pos_scores, neg_scores, mask=None):
        if mask is None:
            return -torch.log(torch.sigmoid(pos_scores - neg_scores) + self.gamma).mean()
        pos_scores = pos_scores * mask
        neg_scores = neg_scores * mask
        return -torch.log(torch.sigmoid(pos_scores - neg_scores) + self.gamma).sum() / mask.sum()

class BCELoss(_Loss):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)
        self.weight: Optional[torch.Tensor]
        self.pos_weight: Optional[torch.Tensor]

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            input,
            target,
            self.weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )



class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
    
    def forward(self,
        input: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        size_average: Optional[bool] = None,
        ignore_index: int = -100,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        return F.cross_entropy(
            input,
            target,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )