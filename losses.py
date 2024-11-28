import torch

def symmetric_cross_entropy(x, x_ema, alpha = 0.5):
    return -(1-alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)

def softmax_entropy_ema(x, x_ema) -> torch.Tensor:
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def information_maximization_loss(probs):
    ent = (-torch.sum(probs * torch.log(probs + 1e-16), dim=1)).mean()
    mean_probs = probs.mean(dim=0)
    div = -torch.sum(mean_probs * torch.log(mean_probs + 1e-16))
    
    return ent - div
