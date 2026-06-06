from torchmetrics import R2Score
import torch

def compute_r2(y_true, y_pred):
    """
    Computes the r2 score for `y_true` and `y_pred`,
    returns `-1` when `y_pred` contains nan values
    """
    y_pred = torch.clamp(y_pred, -3e12, 3e12)
    metric = R2Score().to(y_true.device)
    metric.update(y_pred, y_true)  # same as sklearn.r2_score(y_true, y_pred)
    return metric.compute().item()

def compute_r2_sc(y_true, y_pred):
    return torch.Tensor([compute_r2(y_true[i], y_pred[i]) for i in range(y_pred.shape[0])]).mean()