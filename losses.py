import torch
import torch.nn as nn
import torch.nn.functional as F

def symmetric_cross_entropy(x, x_ema, alpha = 0.5):
    return -(1-alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)

def softmax_entropy_ema(x, x_ema) -> torch.Tensor:
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def information_maximization_loss(outputs):
    probs = outputs.softmax(dim = 1)
    ent = -torch.sum(probs * torch.log(probs + 1e-16), dim=1)
    mean_probs = probs.mean(dim=0)
    div = -torch.sum(mean_probs * torch.log(mean_probs + 1e-16))
    return ent.mean() - div

class L2SPLoss(nn.Module):
    def __init__(self, pre_trained_weights):
        super(L2SPLoss, self).__init__()
        self.pre_trained_weights = pre_trained_weights  # source model weights

    def forward(self, model):
        loss = 0.0
        for name, param in model.named_parameters():
            loss += F.mse_loss(param, self.pre_trained_weights[name].to(param.device))
        return loss
