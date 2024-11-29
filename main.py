import os
import yaml
import wandb
import torch
import logging
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

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


logger = logging.getLogger(__name__)

def process_acc(clients, printall):
    acc_st = 0
    acc_t = 0
    acc_m = 0
    global_correct_st = 0
    global_correct_t = 0
    global_correct_m  = 0
    global_total = 0
    for client in clients:
        global_correct_st += sum(client.correct_preds['student'])
        global_correct_t += sum(client.correct_preds['teacher'])
        global_correct_m += sum(client.correct_preds['mixed'])
        global_total += client.total_preds

        client_acc_st = sum(client.correct_preds['student']) / sum(client.total_preds)*100
        client_acc_t = sum(client.correct_preds['teacher']) / sum(client.total_preds)*100
        client_acc_m = sum(client.correct_preds['mixed']) / sum(client.total_preds)*100
        acc_st += client_acc_st
        acc_t += client_acc_t
        acc_m += client_acc_m

        if printall:
            print(f'{client.name} Student accuracy: {client_acc_st : 0.3f}')
            print(f'{client.name} Teacher accuracy: {client_acc_t: 0.3f}')
            print(f'{client.name} Mixed accuracy: {client_acc_m: 0.3f}')
    
    print(f'Global accuracy(Student): {acc_st/len(clients) : 0.3f}')
    print(f'Global accuracy(Teacher): {acc_t/len(clients) : 0.3f}')
    print(f'Global accuracy(Mixed): {acc_m/len(clients) : 0.3f}')
    print(f'Global Correct: {global_correct_st}(ST) || {global_correct_t}(T) ||{global_correct_m}(M)')
    print(f'Global Total: {global_total}')

    logger.info(f'Global accuracy(Student): {acc_st/len(clients) : 0.3f}')
    logger.info(f'Global accuracy(Teacher): {acc_t/len(clients) : 0.3f}')
    logger.info(f'Global accuracy(Mixed): {acc_m/len(clients) : 0.3f}')

def process_grad(clients):
    gloabal_grad = None
    local_grads = []
    for client in clients:
        local_grads.append(client.get_grad())
        if not gloabal_grad:
            global_grad = client.get_grad()
        else:
            global_grad += client.get_grad()
    
    global_grad /= len(clients)

    differences = []

    for i in range(len(clients)):
        c_dif = torch.sum((global_grad - local_grads[i]) ** 2).item()
        differences.append(c_dif)
    
    print(f'Gradient differences: {c_dif}')


def main(severity, device):
    print(f"==================Dataset: {cfg.CORRUPTION.DATASET} || Batch Size: {cfg.FED.BATCH_SIZE} || Adaptation: {cfg.MODEL.ADAPTATION} || IID : {cfg.FED.IID} || ADAPT_ALL : {cfg.MISC.ADAPT_ALL} || Similarity : {cfg.MISC.SIMILARITY}==================")
    max_use_count = cfg.CORRUPTION.NUM_EX // cfg.FED.BATCH_SIZE 
    print(cfg)
    
    dataset = get_dataset(cfg, severity, cfg.CORRUPTION.DATASET)
    clients = []
    global_model = load_model(cfg.MODEL.ARCH, cfg.MISC.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions)

    for i in range(cfg.FED.NUM_CLIENTS):
        clients.append(Client(f'client_{i}', deepcopy(global_model), cfg, device))

    client_schedule = create_schedule_iid(cfg.FED.NUM_CLIENTS, cfg.FED.NUM_STEPS, cfg.CORRUPTION.TYPE, cfg.FED.TEMPORAL_H)
  
    total_indices_sum = 0
    for t in tqdm(range(cfg.FED.NUM_STEPS)):
        w_locals = []
        for idx, client in enumerate(clients):
            selected_domain = dataset[client_schedule[idx][t]]
            cur_idx = selected_domain['indices'][selected_domain['use_count']]
            total_indices_sum += sum(cur_idx)
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
        
        elif 'ours' in cfg.MODEL.ADAPTATION:
            if t % cfg.FED.AGG_FREQ == 0:
                similarity_mat = torch.zeros((len(clients), len(clients)))
                with torch.no_grad():
                    if cfg.MISC.SIMILARITY == 'ema_probs':
                        probs_list = [client.class_probs_ema for client in clients]
                        for i, prob1 in enumerate(probs_list):
                            for j, prob2 in enumerate(probs_list):
                                similarity = F.cosine_similarity(prob1.reshape(1, -1), prob2.reshape(1,-1))
                                similarity_mat[i, j] = similarity.item()

                    elif cfg.MISC.SIMILARITY == 'weights':
                        params_list = [client.extract_bn_weights_and_biases() for client in clients]
                        for i, params1 in enumerate(params_list):
                            for j, params2 in enumerate(params_list):
                                similarity = cosine_similarity(params1, params2)
                                similarity_mat[i,j] = similarity
                    
                    else:
                        NotImplementedError(f"Similarity method {cfg.MISC.SIMILARITY} not implemented")

                temperature = cfg.MISC.EMA_PROBS_TEMP if cfg.MISC.SIMILARITY == 'ema_probs' else cfg.MISC.TEMP
                scaled_similarity = similarity_mat / temperature
                normalized_similarity = F.softmax(scaled_similarity, dim = 1)
                # # Apply softmax to normalize the similarity values for aggregation
                # exp_scaled_similarity = np.exp(scaled_similarity - np.max(scaled_similarity, axis=1, keepdims=True))  # Subtract max for numerical stability
                # # exp_scaled_similarity = np.exp(scaled_similarity)  # Subtract max for numerical stability
                # normalized_similarity = exp_scaled_similarity / np.sum(exp_scaled_similarity, axis=1, keepdims=True)
                # print(f'Timestep: {t} / {cfg.FED.NUM_STEPS}')

                if t % 399 == 0:
                    print(f'Timestep: {t} || Similarity Matrix')
                    print(normalized_similarity)
                    process_acc(clients, False)
                    process_grad(clients)
    

                # wandb.log({"similarity_mat": similarity_mat})
                for i in range(len(clients)):
                    ww = FedAvg(w_locals, normalized_similarity[i])
                    clients[i].set_state_dict(deepcopy(ww))

    process_acc(clients, True)
    print(total_indices_sum)

if __name__ == '__main__':
    load_cfg_fom_args("CIFAR-10C Evaluation")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ==========================================================================
    desc = f"Arch: {cfg.MODEL.ARCH} || Adaptation: {cfg.MODEL.ADAPTATION} || Similarity : {cfg.MISC.SIMILARITY}\n Dataset: {cfg.CORRUPTION.DATASET}  || Timesteps: {cfg.FED.NUM_STEPS} || Batch Size: {cfg.FED.BATCH_SIZE}\n IID: {cfg.FED.IID} || Temporal Heterogenity: {cfg.FED.TEMPORAL_H} || Spatial Heterogenity: {cfg.FED.SPATIAL_H}\n "
    wandb_api_key=os.environ.get('WANDB_API_KEY')
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        print("WANDB_API_KEY not found in environment variables.")
    
    iid_text = "iid" if cfg.FED.IID else "niid"
    bn_text = "fm" if cfg.MISC.ADAPT_ALL else "bn"
    bn_text += f"_Batch{cfg.FED.BATCH_SIZE}_RST_{cfg.OPTIM.RST}_AUG_{cfg.MISC.USE_AUG}_IMLOSS_{cfg.MISC.USE_IMLOSS}"
    wandb.init(
        project = f"{cfg.CORRUPTION.DATASET}_{cfg.MODEL.ADAPTATION}_{iid_text}_{cfg.MISC.SIMILARITY}",
        config = cfg,
        name = f"{cfg.MODEL.ADAPTATION}_{bn_text}",
        notes = desc,
        dir= "output"
    )
    # ========================================================================== 
    
    for severity in cfg.CORRUPTION.SEVERITY:
        main(severity, device)
