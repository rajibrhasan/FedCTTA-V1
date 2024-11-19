import torch
import torch.nn as nn
import random

def split_indices_into_batches(total_indices, batch_size):
    """
    Splits indices into batches of given size.

    Args:
        total_indices (int): Total number of indices.
        batch_size (int): Number of indices per batch.

    Returns:
        list: List of batches, where each batch is a list of indices.
    """
    num_full_batches = total_indices // batch_size
    remaining_indices = total_indices % batch_size

    shuffled_indices = list(range(total_indices))
    random.shuffle(shuffled_indices)
    index_batches = []

    for batch_idx in range(num_full_batches):
        batch_indices = shuffled_indices[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        index_batches.append(batch_indices)

    return index_batches

def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:#isinstance(m, nn.BatchNorm2d): collect all 
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    print(nm, np)
    return params, names

def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model

def softmax_entropy(x):
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def get_available_corruptions(dataset, max_use_count):
    available_corruptions = []
    for corruption_type in dataset.keys():
        if dataset[corruption_type]['use_count'] < max_use_count:
            available_corruptions.append(corruption_type)
    return available_corruptions