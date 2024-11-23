import yaml

import argparse
import torch
import torch.nn.functional as F
import random
from copy import deepcopy
from client import Client
from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from sklearn.decomposition import PCA
from methods.fed_avg import FedAvg
from yacs.config import CfgNode as CfgNode
from fed_utils import split_indices_into_batches, get_available_corruptions, get_dataset, create_schedule_iid, create_schedule_niid, cosine_similarity
from conf import cfg, load_cfg_fom_args
from tqdm import tqdm
import numpy as np
import logging

logger = logging.getLogger(__name__)

def main(severity, device):
    print(f"===============================Severity: {severity} || IID : {cfg.MISC.IID}===============================")
    max_use_count = cfg.CORRUPTION.NUM_EX // cfg.MISC.BATCH_SIZE 
    
    dataset = get_dataset(cfg, severity, cfg.CORRUPTION.DATASET)
    clients = []
    global_model = load_model(cfg.MODEL.ARCH, cfg.MISC.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions)

    for i in range(cfg.MISC.NUM_CLIENTS):
        clients.append(Client(f'client_{i}', deepcopy(global_model), cfg, device))

    if cfg.MISC.IID:
        client_schedule = create_schedule_iid(cfg.MISC.NUM_CLIENTS, cfg.MISC.NUM_STEPS, cfg.CORRUPTION.TYPE, cfg.MISC.TEMPORAL_H)
    else:
        client_schedule = create_schedule_niid(cfg.MISC.NUM_CLIENTS, cfg.MISC.NUM_STEPS, cfg.CORRUPTION.TYPE, cfg.MISC.TEMPORAL_H, cfg.MISC.SPATIAL_H)
    
    logger.info('Client schedule: \n')
    logger.info(client_schedule)
    
    for t in tqdm(range(cfg.MISC.NUM_STEPS)):
        w_locals = []

        for idx, client in enumerate(clients):
            selected_domain = dataset[client_schedule[idx][t]]
            cur_idx = selected_domain['indices'][selected_domain['use_count']]
            x = selected_domain['all_x'][cur_idx]
            y = selected_domain['all_y'][cur_idx]
            client.domain_list.append(client_schedule[idx][t])
            client.adapt(x, y)
            w_locals.append(deepcopy(client.model_ema.state_dict()))
            selected_domain['use_count'] += 1
        

        bn_params_list = [client.extract_bn_weights_and_biases() for client in clients]
        # feat_vec_list = [client.local_features for client in clients]
        # pvec_list = [client.pvec for client in clients]
        similarity_mat = torch.zeros((len(bn_params_list), len(bn_params_list)))
        for i, bn_params1 in enumerate(bn_params_list):
            for j, bn_params2 in enumerate(bn_params_list):
                similarity = cosine_similarity(bn_params1, bn_params2)
                similarity_mat[i,j] = similarity

        similarity_mat = F.softmax(similarity_mat, dim = -1)

        # if t % 10 == 0:
        #     print(similarity_mat)
        
        for i in range(len(clients)):
            ww = FedAvg(w_locals, similarity_mat[i])
            clients[i].model_ema.load_state_dict(deepcopy(ww))
            clients[i].update_acc()

    acc = 0
    for client in clients:
        client_acc = sum(client.correct_preds_before_adapt) / sum(client.total_preds)
        acc += client_acc
        print(f'{client.name} accuracy: {client_acc: 0.3f}')

    print(f'Global accuracy before adapt: {acc/len(clients) : 0.3f}')
    logger.info(f'Global accuracy before adapt: {acc/len(clients) : 0.3f}')

    acc = 0
    for client in clients:
        client_acc = sum(client.correct_preds_after_adapt) / sum(client.total_preds)
        acc += client_acc
        print(f'{client.name} accuracy: {client_acc: 0.3f}')

    print(f'Global accuracy after adapt: {acc/len(clients) : 0.3f}')
    logger.info(f'Global accuracy after adapt: {acc/len(clients) : 0.3f}')

if __name__ == '__main__':
    load_cfg_fom_args("CIFAR-10C Evaluation")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(cfg)
    for severity in cfg.CORRUPTION.SEVERITY:
        main(severity, device)
