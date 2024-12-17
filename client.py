import torch 
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from sklearn.decomposition import PCA
from fed_utils import ema_update_model
from losses import symmetric_cross_entropy, softmax_entropy_ema, softmax_entropy, information_maximization_loss, L2SPLoss
from transforms_cotta import get_tta_transforms
import wandb
from sklearn.decomposition import PCA


@torch.no_grad()
def update_model_probs(x_ema, x, momentum=0.9):
    return momentum * x_ema + (1 - momentum) * x

class Client(object):
    def __init__(self, name, model, cfg, device):
        self.cfg = cfg
        self.name = name 
        self.model = deepcopy(model)
        self.src_model = deepcopy(model)
        self.img_size = (32, 32) if "cifar" in self.cfg.CORRUPTION.DATASET else (224, 224)
        
        self.configure_model()
        self.params, param_names = self.collect_params()
        # print(f"Learable params: {param_names}")
        self.optimizer = self.setup_optimizer() if len(self.params) > 0 else None
        self.model_ema = deepcopy(self.model)
        for param in self.model_ema.parameters():
            param.detach_()
        
        self.model_state = deepcopy(model.state_dict())
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

        self.ce_loss = nn.CrossEntropyLoss()
        self.l2_loss = L2SPLoss(self.model_state)
        self.pvec = None
        self.feat = None
        self.predictions = []
        self.features = []
        self.pvec = []

    def update_principal_vector(self, x):
        pca = PCA(n_components=1)
        projected_data = pca.fit_transform(x.reshape(len(x), -1))
        pvec = pca.components_[0]
        self.pvec.append(torch.tensor(pvec))

    def adapt(self, x, y):
        self.update_principal_vector(x)
        self.optimizer.zero_grad()
        self.x = x
        self.y = y
        self.model.to(self.device)
        self.model_ema.to(self.device)
        self.src_model.to(self.device)

        feat, outputs = self.model(self.x.to(self.device))
        _, outputs_ema = self.model_ema(self.x.to(self.device))
        # anchor_prob = torch.nn.functional.softmax(self.src_model(x.to(self.device)), dim=1).max(1)[0]

        self.predictions.append(outputs.detach().mean(0))
        self.features.append(feat.detach().mean(0))

        # if self.feat is None:
        #     self.feat = feat.mean(0)
        # else:
        #     self.feat = update_model_probs(self.feat, feat)

        if self.cfg.MISC.USE_AUG and anchor_prob.mean(0)<self.cfg.OPTIM.AP:
            outputs_emas = []
            for i in range(self.cfg.MISC.N_AUGMENTATIONS):
                outputs_  = self.model_ema(self.tta_aug(x).to(self.device)).detach()
                outputs_emas.append(outputs_)
        
            outputs_ema = torch.stack(outputs_emas).mean(0)

        # loss = symmetric_cross_entropy(outputs, outputs_ema).mean(0)
        loss = softmax_entropy(outputs).mean(0)
        # loss = self.ce_loss(outputs, y.to(self.device))
        im_loss = information_maximization_loss(outputs)
        mse_loss = self.l2_loss(self.model)

        wandb.log({f'{self.name}_L2': mse_loss})
        
        # if len(self.domain_list) % 10: 
        #     print(f'{self.name}_l2_loss: ', mse_loss.item())

    
        if self.cfg.MISC.USE_IMLOSS:
            loss += im_loss
    
        loss.backward()
        self.optimizer.step()
        wandb.log({f'{self.name}_loss': loss.item()})

        self.model_ema = ema_update_model(
            model_to_update=self.model_ema,
            model_to_merge=self.model,
            momentum=self.cfg.OPTIM.MT,
            device=self.device,
            update_all=True
        )

        self.model.to('cpu')
        self.model_ema.to('cpu')
        self.src_model.to('cpu')

        if len(self.domain_list) % 20 == 0 and self.cfg.OPTIM.RST> 0:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape) < self.cfg.OPTIM.RST).float()
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)

        _, st_pred = torch.max(outputs, 1)
        correct_st = (st_pred == self.y.to(self.device)).sum().item()
        self.correct_preds['student'].append(correct_st)

        _, t_pred = torch.max(outputs_ema, 1)
        correct_t = (t_pred == self.y.to(self.device)).sum().item()
        self.correct_preds['teacher'].append(correct_t)

        _, m_pred = torch.max(outputs_ema + outputs, 1)
        correct_m = (m_pred == self.y.to(self.device)).sum().item()
        self.correct_preds['mixed'].append(correct_m)

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

    def get_grad(self):
        gradients = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients.append(param.grad.clone().view(-1))  # Flatten each gradient tensor

        # Concatenate all gradients
        flat_gradients = torch.cat(gradients)
        return deepcopy(flat_gradients)

    def get_state_dict(self):
        return self.model.state_dict()
    
    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
    
    def get_model(self):
        return self.model
