import torch


def ndcg_at_k(pred, target, k):
    """
    Compute Normalized Discounted Cumulative Gain at k.
    
    Args:
        pred (torch.Tensor): Tensor containing predicted item. shape: (batch_size, n_items) start from 1
        target (torch.Tensor): Tensor containing ground truth item. shape: (batch_size,) start from 1
        k (int): The rank to compute NDCG at.
    
    Returns:
        torch.Tensor: NDCG at k.
    """
    pred = pred[:, :k]
    target = target.unsqueeze(-1)
    hit_matrix = (pred == target).float()
    dcg = (hit_matrix / torch.log2(torch.arange(k, device=pred.device).float() + 2)).sum(dim=-1)
    idcg = 1.0
    ndcg = dcg / idcg
    return torch.mean(ndcg).numpy()

def recall_at_k(pred, target, k):
    """
    Compute Recall at k.
    
    Args:
        pred (torch.Tensor): Tensor containing predicted item. shape: (batch_size, n_items) start from 1
        target (torch.Tensor): Tensor containing ground truth item. shape: (batch_size,) start from 1
        k (int): The rank to compute Recall at.
    
    Returns:
        torch.Tensor: Recall at k.
    """
    pred = pred[:, :k]
    target = target.unsqueeze(-1)
    hit_matrix = (pred == target).float()
    recall = hit_matrix.sum(dim=-1)
    return torch.mean(recall).numpy()