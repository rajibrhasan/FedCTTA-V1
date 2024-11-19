import yaml

import argparse
import torch
import random
from copy import deepcopy
from client import Client
from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from sklearn.decomposition import PCA
from methods.fed_avg import FedAvg
from fed_utils import split_indices_into_batches, get_available_corruptions


def main(cfg, severity, device):
    max_use_count = cfg.MISC.NUM_STEPS // (len(cfg.CORRUPTION.TYPE) * cfg.MISC.NUM_CLIENTS)
    dataset = {}
    clients = []
    
    for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
        x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                        severity, cfg.MISC.DATA_DIR, True,
                                        [corruption_type])
        dataset[corruption_type]['x'] = x_test
        dataset[corruption_type]['y'] = y_test
        dataset[corruption_type]['indices'] = split_indices_into_batches(len(x_test), cfg.MISC.BATCH_SIZE)
        dataset[corruption_type]['use_count'] = 0
    
    global_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions)

    for i in range(cfg.MISC.NUM_CLIENTS):
        clients.append(Client(f'client_{i}', deepcopy(global_model), cfg, device))
    
    for i in range(cfg.MISC.NUM_STEPS):
        selected_domain = random.choice(get_available_corruptions(dataset, max_use_count))
        for client in clients:
            cur_idx = dataset[selected_domain]['indice']['use_count']
            x  = dataset[selected_domain]['x'][cur_idx]
            y  = dataset[selected_domain]['y'][cur_idx]
            client.x = x
            client.y = y
            client.adapt()
            dataset[selected_domain]['use_count'] += 1

        
        w_locals = []
        for client in clients:
            w_locals.append(client.model.state_dict())

        ww = FedAvg(w_locals)
        for clients in clients:
            clients.model.load_state_dict(deepcopy(ww))
            clients.update_acc()

    acc = 0
    for client in clients:
        print(f'{client.name} accuracy: {client.acc_list.mean()}')
        acc += client.acc_list.mean()
    print(f'Global accuracy: {acc/len(clients)}')
            

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', type=str, default='configs/config_cifar10.yaml')
    args = argparser.parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for severity in config.CORRUPTION.SEVERITY:
        main(config, severity, device)
