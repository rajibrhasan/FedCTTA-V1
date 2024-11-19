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
from yacs.config import CfgNode as CfgNode
from fed_utils import split_indices_into_batches, get_available_corruptions
from conf import cfg, load_cfg_fom_args
from tqdm import tqdm

def main(severity, device):
    max_use_count = cfg.CORRUPTION.NUM_EX // (cfg.MISC.BATCH_SIZE * cfg.MISC.NUM_CLIENTS)
    print(max_use_count)
    dataset = {}
    clients = []
    
    for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
        dataset[corruption_type] = {}
        x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                        severity, cfg.MISC.DATA_DIR, True,
                                        [corruption_type])
                            
        dataset[corruption_type]['x'] = x_test
        dataset[corruption_type]['y'] = y_test
        dataset[corruption_type]['indices'] = split_indices_into_batches(len(x_test), cfg.MISC.BATCH_SIZE)
        dataset[corruption_type]['use_count'] = 0
    
    global_model = load_model(cfg.MODEL.ARCH, cfg.MISC.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions)

    for i in range(cfg.MISC.NUM_CLIENTS):
        clients.append(Client(f'client_{i}', deepcopy(global_model), cfg, device))
    
    for i in tqdm(range(cfg.MISC.NUM_STEPS)):
        w_locals = []
        selected_domain = dataset[random.choice(get_available_corruptions(dataset, max_use_count))]
        for client in clients:
            cur_idx = selected_domain['indices'][selected_domain['use_count']]
            x  = selected_domain['x'][cur_idx]
            y  = selected_domain['y'][cur_idx]

            client.x = x
            client.y = y
            client.adapt()
            w_locals.append(client.model.state_dict())

        selected_domain['use_count'] += 1

        ww = FedAvg(w_locals)
        for client in clients:
            client.model.load_state_dict(deepcopy(ww))
            client.update_acc()

    acc = 0
    for client in clients:
        print(f'{client.name} accuracy: {client.acc_list.mean()}')
        acc += client.acc_list.mean()
    print(f'Global accuracy: {acc/len(clients)}')
            

if __name__ == '__main__':

    load_cfg_fom_args("CIFAR-10C Evaluation")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(cfg)
    for severity in cfg.CORRUPTION.SEVERITY:
        main(severity, device)
