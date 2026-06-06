import torch
def compute_l2_distance(target, transport):
    return torch.dist(
        torch.mean(target, dim=0), torch.mean(transport, dim=0), p=2)