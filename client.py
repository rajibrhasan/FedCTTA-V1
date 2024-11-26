import numpy as np
from copy import deepcopy

import torch 
from torch import nn, optim
import torch.nn.functional as F
from sklearn.decomposition import PCA
from fed_utils import ema_update_model
from losses import symmetric_cross_entropy, softmax_entropy_ema, softmax_entropy
import wandb



@torch.no_grad()
def update_model_probs(x_ema, x, momentum=0.9):
    return momentum * x_ema + (1 - momentum) * x

class Client(object):
    def __init__(self, name, model, cfg, device):
        self.cfg = cfg
        self.name = name 
        self.model = deepcopy(model)
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.momentum_probs = cfg.MISC.MOMENTUM_PROBS
        
        
        self.configure_model()
        self.params, param_names = self.collect_params()
        self.optimizer = self.setup_optimizer() if len(self.params) > 0 else None

        self.model_ema = deepcopy(self.model)
        for param in self.model_ema.parameters():
            param.detach_()

        self.total_preds = []
        self.correct_before = {
            'student': [],
            'teacher': [], 
            'mixed': []

        }

        self.correct_after = {
            'student': [],
            'teacher': [], 
            'mixed': []

        }

        self.domain_list = []
        self.device = device
        self.pvec = None
        self.local_features = None

        self.class_probs_ema = 1 / self.num_classes * torch.ones(self.num_classes).to(self.device)


    def adapt(self, x, y):

        self.x = x
        self.y = y
        # self.update_pvec()
        self.model.to(self.device)
        self.model_ema.to(self.device)

        outputs = self.model(self.x.to(self.device))
        outputs_ema = self.model_ema(self.x.to(self.device))

        loss = symmetric_cross_entropy(outputs, outputs_ema).mean(0)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.model_ema = ema_update_model(
            model_to_update=self.model_ema,
            model_to_merge=self.model,
            momentum=self.cfg.MISC.MOMENTUM_SRC,
            device=self.device,
            update_all=True
        )

        if len(self.domain_list) % 10==0:
            wandb.log({f'{self.name}_loss': loss.item()})

        self.model.to('cpu')
        self.model_ema.to('cpu')

        _, st_pred = torch.max(outputs, 1)
        correct_st = (st_pred == self.y.to(self.device)).sum().item()
        self.correct_before['student'].append(correct_st)

        _, t_pred = torch.max(outputs_ema, 1)
        correct_t = (t_pred == self.y.to(self.device)).sum().item()
        self.correct_before['teacher'].append(correct_t)

        _, m_pred = torch.max(outputs_ema+outputs, 1)
        correct_m = (m_pred == self.y.to(self.device)).sum().item()
        self.correct_before['mixed'].append(correct_m)

        self.total_preds.append(len(self.y))

        self.class_probs_ema = update_model_probs(x_ema=self.class_probs_ema, x=outputs.softmax(1).mean(0), momentum=self.momentum_probs)

    def update_pvec(self):
        pca = PCA(n_components=1)  
        pca.fit(self.x.reshape(self.x.shape[0], -1))
        self.pvec = pca.components_[0]

    def update_acc(self):
        self.model.to(self.device)
        self.model_ema.to(self.device)
        with torch.no_grad():
            outputs = self.model(self.x.to(self.device))
            outputs_ema = self.model_ema(self.x.to(self.device))
    
            _, st_pred = torch.max(outputs, 1)
            correct_st = (st_pred == self.y.to(self.device)).sum().item()
            self.correct_after['student'].append(correct_st)

            _, t_pred = torch.max(outputs_ema, 1)
            correct_t = (t_pred == self.y.to(self.device)).sum().item()
            self.correct_after['teacher'].append(correct_t)
            self.total_preds.append(len(self.y))

            _, m_pred = torch.max(outputs_ema+outputs, 1)
            correct_m = (m_pred == self.y.to(self.device)).sum().item()
            self.correct_after['mixed'].append(correct_m)

        self.model.to('cpu')
        self.model_ema.to('cpu')
    

    def setup_optimizer(self):
        """Set up optimizer for tent adaptation.
        For best results, try tuning the learning rate and batch size.
        """
        if self.cfg.OPTIM.METHOD == 'Adam':
            return optim.Adam(self.params,
                        lr=self.cfg.OPTIM.LR,
                        betas=(self.cfg.OPTIM.BETA, 0.999),
                        weight_decay=self.cfg.OPTIM.WD)
        
        elif self.cfg.OPTIM.METHOD == 'SGD':
            return optim.SGD(self.params,
                    lr=self.cfg.OPTIM.LR,
                    momentum=self.cfg.OPTIM.MOMENTUM,
                    dampening=self.cfg.OPTIM.DAMPENING,
                    weight_decay=self.cfg.OPTIM.WD,
                    nesterov=self.cfg.OPTIM.NESTEROV)
        else:
            raise NotImplementedError(f"Unknown optimizer: {self.cfg.optim_method}")
    
    def configure_model(self):
        """Configure model"""
        self.model.train()
        # self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what we update
        self.model.requires_grad_(False)
        # enable all trainable
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            else:
                if self.cfg.MISC.ADAPT_ALL:
                    m.requires_grad_(True)
                
    def collect_params(self):
        """Collect all trainable parameters.
        Walk the model's modules and collect all parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names

    def extract_bn_weights_and_biases(self):
        bn_params = {}
        for name, layer in self.model.named_modules():
            for np, p in layer.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    bn_params[f"{name}.{np}"] = p.data.cpu()
                    # params.append(p)
                    # names.append(f"{nm}.{np}")
                # gamma = layer.weight.data.cpu()  # Scale (weight)
                # beta = layer.bias.data.cpu()    # Offset (bias)
                # weights = torch.cat((gamma, beta), dim =0)
                # bn_params[name] = weights
        return deepcopy(bn_params)

    def get_state_dict(self):
        return self.model.state_dict()
    
    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
    
    def get_model(self):
        return self.model
