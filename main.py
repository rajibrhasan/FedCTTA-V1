import logging
from copy import deepcopy

import torch
import torch.nn.functional as F
from tqdm import tqdm
from yacs.config import CfgNode as CfgNode

from client import Client
from conf import cfg, load_cfg_fom_args
from fed_utils import (
    cosine_similarity,
    create_schedule_iid,
    create_schedule_niid,
    get_dataset,
)
from methods.fed_avg import FedAvg
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

logger = logging.getLogger(__name__)


def main(severity, device):
    """
    Main function to run the Federated Learning simulation

    Args:
        severity (int): Severity of the corruption
        device (torch.device): Device to run the simulation on
    """
    logger.log(f"{'='*30}Severity: {severity} || IID : {cfg.MISC.IID}{'='*30}")

    dataset = get_dataset(cfg, severity, cfg.CORRUPTION.DATASET)
    clients = []
    global_model = load_model(
        cfg.MODEL.ARCH,
        cfg.MISC.CKPT_DIR,
        cfg.CORRUPTION.DATASET,
        ThreatModel.corruptions,
    )

    for i in range(cfg.MISC.NUM_CLIENTS):
        clients.append(Client(f"client_{i}", deepcopy(global_model), cfg, device))

    if cfg.MISC.IID:
        client_schedule = create_schedule_iid(
            cfg.MISC.NUM_CLIENTS,
            cfg.MISC.NUM_STEPS,
            cfg.CORRUPTION.TYPE,
            cfg.MISC.TEMPORAL_H,
        )
    else:
        client_schedule = create_schedule_niid(
            cfg.MISC.NUM_CLIENTS,
            cfg.MISC.NUM_STEPS,
            cfg.CORRUPTION.TYPE,
            cfg.MISC.TEMPORAL_H,
            cfg.MISC.SPATIAL_H,
        )

    logger.info("Client schedule: \n")
    logger.info(client_schedule)

    for t in tqdm(range(cfg.MISC.NUM_STEPS)):
        w_locals = []

        for idx, client in enumerate(clients):
            selected_domain = dataset[client_schedule[idx][t]]
            cur_idx = selected_domain["indices"][selected_domain["use_count"]]
            x = selected_domain["all_x"][cur_idx]
            y = selected_domain["all_y"][cur_idx]
            client.domain_list.append(client_schedule[idx][t])
            client.adapt(x, y)
            w_locals.append(deepcopy(client.model.state_dict()))
            selected_domain['use_count'] += 1
        

        bn_params_list = [client.extract_bn_weights_and_biases() for client in clients]
        similarity_mat = torch.zeros((len(bn_params_list), len(bn_params_list)))
        for i, bn_params1 in enumerate(bn_params_list):
            for j, bn_params2 in enumerate(bn_params_list):
                similarity = cosine_similarity(bn_params1, bn_params2)
                similarity_mat[i, j] = similarity

        similarity_mat = F.softmax(similarity_mat, dim=-1)
        for i, _ in enumerate(clients):
            ww = FedAvg(w_locals, similarity_mat[i])
            clients[i].model.load_state_dict(deepcopy(ww))
            clients[i].update_acc()

    acc = 0
    for client in clients:
        client_acc = sum(client.correct_preds_before_adapt) / sum(client.total_preds)*100
        acc += client_acc
        logger.info("%s accuracy: %0.3f", client.name, client_acc)
    logger.info("Global accuracy before adapt: %0.3f", acc / len(clients))

    acc = 0
    for client in clients:
        client_acc = sum(client.correct_preds_after_adapt) / sum(client.total_preds)*100
        acc += client_acc
        logger.info("%s accuracy: %0.3f", client.name, client_acc)
    logger.info("Global accuracy after adapt: %0.3f", acc / len(clients))


if __name__ == "__main__":
    load_cfg_fom_args("CIFAR-10C Evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(cfg)
    for severity in cfg.CORRUPTION.SEVERITY:
        main(severity, device)

if __name__ == "__main__":
    load_cfg_fom_args("CIFAR-10C Evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(cfg)
    for severity in cfg.CORRUPTION.SEVERITY:
        main(severity, device)
