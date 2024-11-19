import numpy as np
import copy 

import torch 
from torch import nn, optim
import torch.nn.functional as F
from sklearn.decomposition import PCA
from fed_utils import collect_params, configure_model, softmax_entropy

class Client(object):
    def __init__(self, name, model, cfg, device):
        self.name = name 
        self.model = model
        self.model = configure_model(self.model)
        params, param_names = collect_params(model)
        self.optimizer = self.setup_optimizer(params, cfg)
        self.acc_list = []
        self.device = device
        self.p_vecs = []
        self.x = None
        self.y = None

    def adapt(self):
        self.update_p_vecs()
        self.model.to(self.device)
        outputs = self.model(self.x.to(self.device))
        loss = (softmax_entropy(outputs)).mean(0) 
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.model.to('cpu')
    
    def update_p_vecs(self):
        pca = PCA(n_components=1)  
        pca.fit(self.x.reshape(self.x.shape[0], -1))
        self.p_vecs.append(pca.components_[0])

    
    def update_acc(self, model = None):
        if model is not None:
            model.to(self.device)
            outputs = model(self.x.to(self.device))
        else:
            self.model.to(self.device)
            outputs = self.model(self.x.to(self.device))
       
        _, predicted = torch.max(outputs, 1)
        total = self.y.size(0)
        correct = (predicted == self.y.to(self.device)).sum().item()
        self.acc_list.append(correct / total)
        self.model.to('cpu')

    def setup_optimizer(params, cfg):
        """Set up optimizer for tent adaptation.
        For best results, try tuning the learning rate and batch size.
        """
        if cfg.optim_method == 'Adam':
            return optim.Adam(params,
                        lr=cfg.optim_lr,
                        betas=(cfg.optim_beta, 0.999),
                        weight_decay=cfg.optim_wd)
        
        elif cfg.optim_method == 'SGD':
            return optim.SGD(params,
                    lr=cfg.optim_lr,
                    momentum=cfg.optim_momentum,
                    dampening=cfg.optim_dampening,
                    weight_decay=cfg.optim_wd,
                    nesterov=cfg.optim_nesterov)
        else:
            raise NotImplementedError(f"Unknown optimizer: {cfg.optim_method}")
    
    def get_state_dict(self):
        return self.model.state_dict()
    
    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
    
    def get_model(self):
        return self.model
    
   

 
   
   