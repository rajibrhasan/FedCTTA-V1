import torch
import torch.nn as nn
import random
from robustbench.data import load_cifar10c, load_cifar100c
    
@torch.no_grad()
def ema_update_model(model_to_update, model_to_merge, momentum, device, update_all=False):
    if momentum < 1.0:
        for param_to_update, param_to_merge in zip(model_to_update.parameters(), model_to_merge.parameters()):
            if param_to_update.requires_grad or update_all:
                param_to_update.data = momentum * param_to_update.data + (1 - momentum) * param_to_merge.data.to(device)
    return model_to_update

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

def get_available_corruptions(dataset, max_use_count):
    available_corruptions = []
    for corruption_type in dataset.keys():
        if dataset[corruption_type]['use_count'] < max_use_count:
            available_corruptions.append(corruption_type)
    return available_corruptions

def get_dataset(cfg, severity, dataset_name):
    dataset = {}
    for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
        dataset[corruption_type] = {}
        if dataset_name == 'cifar10':
            x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX, severity, cfg.MISC.DATA_DIR, True, [corruption_type]) 
        elif dataset_name == 'cifar100':
            x_test, y_test = load_cifar100c(cfg.CORRUPTION.NUM_EX, severity, cfg.MISC.DATA_DIR, True, [corruption_type])            
        dataset[corruption_type]['all_x'] = x_test
        dataset[corruption_type]['all_y'] = y_test
        dataset[corruption_type]['indices'] = split_indices_into_batches(len(x_test), cfg.MISC.BATCH_SIZE)
        dataset[corruption_type]['use_count'] = 0
        
    return dataset

def assign_clients(num_clients, domains):
    clients = list(range(num_clients))
    num_domains = len(domains)
    random.shuffle(clients)
    distributed_clients = {domains[i]: clients[i::num_domains] for i in range(num_domains)}
    return distributed_clients

def get_available_domains(selection_count, domains_per_timestep, max_selection_count):
    available_domains = [domain for domain, count in selection_count.items() if count < max_selection_count//2]
    if len(available_domains) < domains_per_timestep:
        sorted_domains = sorted(selection_count.items(), key=lambda item: item[1])  # Sort by selection count
        additional_domains = [domain for domain, count in sorted_domains if domain not in available_domains]
        available_domains.extend(additional_domains[:domains_per_timestep - len(available_domains)])

    return available_domains

def create_schedule_niid(num_clients, timesteps, domains, temporal_h, spatial_h):
    domain_per_step = int(spatial_h * num_clients)
    time_interval = int(1/temporal_h)
    assert time_interval > 0 and time_interval <= timesteps
    schedule = []
    selection_count = {domain: 0 for domain in domains}
    max_selection_count = timesteps*domain_per_step//(len(domains))

    # Create the schedule and client assignments
    for t in range(timesteps):
        if t % time_interval == 0:
            available_domains = get_available_domains(selection_count, domain_per_step, max_selection_count)
            selected_domains = random.sample(available_domains, domain_per_step)
            # Update selection count
            for domain in selected_domains:
                selection_count[domain] += int(num_clients * time_interval/domain_per_step)
            # Add to schedule
            schedule.append(selected_domains)

    random.shuffle(schedule)
    print(selection_count)

    clients_schedule = [[] for i in range(num_clients)]
    for t, selected_domains in enumerate(schedule):
        distributed_clients = assign_clients(num_clients, selected_domains)
        for domain, assigned_clients in distributed_clients.items():
            for client in assigned_clients:
                clients_schedule[client].extend([domain]*time_interval)
    return clients_schedule

def create_schedule_iid(num_clients, timesteps, domains, temporal_h):
    time_interval = int(1/temporal_h)
    max_count = timesteps//len(domains)
    use_count = {domain: 0 for domain in domains}
    schedule = []
    last_selection = None
    
    for t in range(timesteps):
        if t % time_interval == 0:
            available_domains = [domain for domain, count in use_count.items() if last_selection != domain and  count < max_count]
            last_selection = random.choice(available_domains)
            schedule.extend([last_selection]*min(time_interval, max_count - use_count[last_selection]))
            use_count[last_selection] += min(time_interval, max_count - use_count[last_selection])

    client_schedule = [schedule for i in range(num_clients)]
    return client_schedule


def cosine_similarity(bn_params1, bn_params2):
    assert bn_params1.keys() == bn_params2.keys(), "Keys must match"
    similarities = []
    for layer_name in bn_params1:
        similarity = nn.functional.cosine_similarity(bn_params1[layer_name].reshape(1, -1), bn_params2[layer_name].reshape(1, -1), dim = 1)
        similarity = torch.clamp(similarity, min=0, max=1)
        similarities.append(similarity.item())
    
    return sum(similarities)/len(similarities)