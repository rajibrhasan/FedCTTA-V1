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
import wandb
import os

logger = logging.getLogger(__name__)


def main(severity, device):
    print(f"===============================Dataset: {cfg.CORRUPTION.DATASET} || Batch Size: {cfg.MISC.BATCH_SIZE} || Adaptation: {cfg.MODEL.ADAPTATION} || IID : {cfg.MISC.IID}===============================")
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
    
    # logger.info('Client schedule: \n')
    # logger.info(client_schedule)


    for t in tqdm(range(cfg.MISC.NUM_STEPS)):
        w_locals = []
        for idx, client in enumerate(clients):
            selected_domain = dataset[client_schedule[idx][t]]
            cur_idx = selected_domain['indices'][selected_domain['use_count']]
            x = selected_domain['all_x'][cur_idx]
            y = selected_domain['all_y'][cur_idx]
            client.domain_list.append(client_schedule[idx][t])
            client.adapt(x, y)
            w_locals.append(deepcopy(client.get_state_dict()))
            selected_domain['use_count'] += 1
        
        if cfg.MODEL.ADAPTATION == 'fedavg':
            w_avg = FedAvg(w_locals)
            for client in clients:
                client.set_state_dict(deepcopy(w_avg))
        
        elif cfg.MODEL.ADAPTATION == 'ours':
            if t % cfg.MISC.AGG_FREQ == 0:
                with torch.no_grad():
                    bn_params_list = [client.extract_bn_weights_and_biases() for client in clients]
                    # feat_vec_list = [client.local_features for client in clients]
                    # pvec_list = [client.pvec for client in clients]
                    similarity_mat = torch.zeros((len(bn_params_list), len(bn_params_list)))
                    for i, bn_params1 in enumerate(bn_params_list):
                        for j, bn_params2 in enumerate(bn_params_list):
                            similarity = cosine_similarity(bn_params1, bn_params2)
                            similarity_mat[i,j] = similarity

                scaled_similarity = np.array(similarity_mat / cfg.MISC.TEMP)
                # Apply softmax to normalize the similarity values for aggregation
                exp_scaled_similarity = np.exp(scaled_similarity - np.max(scaled_similarity, axis=1, keepdims=True))  # Subtract max for numerical stability
                # exp_scaled_similarity = np.exp(scaled_similarity)  # Subtract max for numerical stability
                normalized_similarity = exp_scaled_similarity / np.sum(exp_scaled_similarity, axis=1, keepdims=True)
                # if t  % 10 == 0:
                #     print(normalized_similarity)

                # wandb.log({"similarity_mat": similarity_mat})
                
                for i in range(len(clients)):
                    ww = FedAvg(w_locals, torch.tensor(normalized_similarity[i]))
                    clients[i].set_state_dict(deepcopy(ww))

    
    acc_st_before = 0
    acc_t_before = 0
    acc_m_before = 0
    for client in clients:
        client_acc_st = sum(client.correct_before['student']) / sum(client.total_preds)*100
        client_acc_t = sum(client.correct_before['teacher']) / sum(client.total_preds)*100
        client_acc_m = sum(client.correct_before['mixed']) / sum(client.total_preds)*100
        acc_st_before += client_acc_st
        acc_t_before += client_acc_t
        acc_m_before += client_acc_m
        print(f'{client.name} Student accuracy: {client_acc_st : 0.3f}')
        print(f'{client.name} Teacher accuracy: {client_acc_t: 0.3f}')
        print(f'{client.name} Mixed accuracy: {client_acc_m: 0.3f}')
    
    print(f'Global accuracy(Student): {acc_st_before/len(clients) : 0.3f}')
    print(f'Global accuracy(Teacher): {acc_t_before/len(clients) : 0.3f}')
    print(f'Global accuracy(Mixed): {acc_m_before/len(clients) : 0.3f}')


    # acc_st_after = 0
    # acc_t_after = 0
    # acc_m_after = 0
    # for client in clients:
    #     client_acc_st = sum(client.correct_after['student']) / sum(client.total_preds)*100
    #     client_acc_t = sum(client.correct_after['teacher']) / sum(client.total_preds)*100
    #     client_acc_m =  sum(client.correct_after['mixed']) / sum(client.total_preds)*100
    #     acc_st_after += client_acc_st
    #     acc_t_after += client_acc_t
    #     acc_m_after += client_acc_m
    #     print(f'{client.name} Student accuracy: {client_acc_st : 0.3f}')
    #     print(f'{client.name} Teacher accuracy: {client_acc_t: 0.3f}')
    #     print(f'{client.name} Mixed accuracy: {client_acc_m: 0.3f}')

    
    # print(f'Global accuracy(Student): {acc_st_after/len(clients) : 0.3f}')
    # print(f'Global accuracy(Teacher): {acc_t_after/len(clients) : 0.3f}')
    # print(f'Global accuracy(Mixed): {acc_m_after/len(clients) : 0.3f}')



if __name__ == '__main__':
    load_cfg_fom_args("CIFAR-10C Evaluation")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==========================================================================
    desc = f"Arch: {cfg.MODEL.ARCH} || Adaptation: {cfg.MODEL.ADAPTATION} \n Dataset: {cfg.CORRUPTION.DATASET}  || Timesteps: {cfg.MISC.NUM_STEPS} || Batch Size: {cfg.MISC.BATCH_SIZE}\n IID: {cfg.MISC.IID} || Temporal Heterogenity: {cfg.MISC.TEMPORAL_H} || Spatial Heterogenity: {cfg.MISC.SPATIAL_H}\n "
    wandb_api_key=os.environ.get('WANDB_API_KEY')
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        print("WANDB_API_KEY not found in environment variables.")
    wandb.init(
        project=cfg.CORRUPTION.DATASET + "_iid" if cfg.MISC.IID else "_niid",
        config = cfg,
        name = cfg.MODEL.ARCH + cfg.MODEL.ADAPTATION,
        notes = desc,
        dir= "output"
    )
    # ========================================================================== 
    
    for severity in cfg.CORRUPTION.SEVERITY:
        main(severity, device)
