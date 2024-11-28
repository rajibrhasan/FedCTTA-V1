import torch 
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from sklearn.decomposition import PCA
from fed_utils import ema_update_model
from losses import symmetric_cross_entropy, softmax_entropy_ema, softmax_entropy
from transforms_cotta import get_tta_transforms
import wandb

@torch.no_grad()
def update_model_probs(x_ema, x, momentum=0.9):
    return momentum * x_ema + (1 - momentum) * x

class Client(object):
    def __init__(self, name, model, cfg, device):
        self.cfg = cfg
        self.name = name 
        self.model = deepcopy(model)
        self.img_size = (32, 32) if "cifar" in self.cfg.CORRUPTION.DATASET else (224, 224)
        self.decay_factor = 0.94
        self.min_mom = 0.005
        
        self.configure_model()
        self.params, param_names = self.collect_params()
        # print(f"Learable params: {param_names}")
        self.optimizer = self.setup_optimizer() if len(self.params) > 0 else None

        self.model_ema = deepcopy(self.model)
        for param in self.model_ema.parameters():
            param.detach_()

        self.device = device
        self.class_probs_ema = 1 / cfg.MODEL.NUM_CLASSES * torch.ones(cfg.MODEL.NUM_CLASSES).to(self.device)
        # self.class_probs_ema = None
        self.tta_aug = get_tta_transforms(self.img_size)

        self.total_preds = []
        self.correct_preds = {
            'student': [],
            'teacher': [], 
            'mixed': []

        }

        self.domain_list = []

    def get_transform_images(self, x):
        x_temp = []
        for _ in range(self.cfg.MISC.N_AUGMENTATIONS):
            x_temp.append(self.tta_aug(x))
        
        x_temp = torch.stack(x_temp)
        return x_temp

    def adapt(self, x, y):
        self.x = x
        self.y = y
        self.model.to(self.device)
        # self.model_ema.to(self.device)
        outputs = []
        mom_pre = 0.1


        # outputs = self.model(self.x.to(self.device))
        # outputs_ema = self.model_ema(self.x.to(self.device))
        for img in x:
            self.model.eval()
            mom_new = (mom_pre * self.decay_factor)
            for m in self.model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm1d)):
                    m.train()
                    m.momentum = mom_new + self.min_mom
            mom_pre = mom_new
            _ = self.model(self.get_transform_images(img).to(self.device))
            outputs.append(self.model(img.to(self.device)))


        # self.model_ema = ema_update_model(
        #     model_to_update=self.model_ema,
        #     model_to_merge=self.model,
        #     momentum=self.cfg.MISC.MOMENTUM_TEACHER,
        #     device=self.device,
        #     update_all=True
        # )

        self.model.to('cpu')
        # self.model_ema.to('cpu')
        outputs = torch.stack(outputs).squeeze(1)

        _, st_pred = torch.max(outputs, 1)
        correct_st = (st_pred == self.y.to(self.device)).sum().item()
        self.correct_preds['student'].append(correct_st)

        # _, t_pred = torch.max(outputs_ema, 1)
        # correct_t = (t_pred == self.y.to(self.device)).sum().item()
        # self.correct_preds['teacher'].append(correct_t)

        # _, m_pred = torch.max(outputs_ema + outputs, 1)
        # correct_m = (m_pred == self.y.to(self.device)).sum().item()
        # self.correct_preds['mixed'].append(correct_m)

        self.total_preds.append(len(self.y))
        if self.class_probs_ema is None:
            self.class_probs_ema = outputs.softmax(1).mean(0)
        else:
            self.class_probs_ema = update_model_probs(x_ema = self.class_probs_ema, x = outputs.softmax(1).mean(0), momentum = self.cfg.MISC.MOMENTUM_PROBS)

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
