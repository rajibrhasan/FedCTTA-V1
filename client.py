from copy import deepcopy

import torch
from sklearn.decomposition import PCA
from torch import nn, optim

from fed_utils import ema_update_model
from losses import symmetric_cross_entropy


class Client(object):
    """A class representing a client in a federated learning system."""

    def __init__(self, name, model, cfg, device):
        self.cfg = cfg
        self.name = name
        self.model = deepcopy(model)

        self.configure_model()
        self.params, _ = self.collect_params()
        self.optimizer = self.setup_optimizer() if len(self.params) > 0 else None

        self.model_ema = deepcopy(self.model)
        for param in self.model_ema.parameters():
            param.detach_()

        self.correct_preds_before_adapt = []
        self.correct_preds_after_adapt = []
        self.total_preds = []
        self.domain_list = []
        self.device = device
        self.pvec = None
        self.local_features = None

    def adapt(self, x, y):
        """
        Adapts the model to new data.

        Args:
            x (torch.Tensor): The input data.
            y (torch.Tensor): The target labels.

        Returns:
            None
        """
        self.x = x
        self.y = y
        self.update_pvec()
        self.model.to(self.device)
        self.model_ema.to(self.device)

        _, outputs = self.model(self.x.to(self.device))
        _, outputs_ema = self.model_ema(self.x.to(self.device))

        # self.local_features = feats.mean(0).detach().cpu()

        loss = softmax_entropy(outputs).mean(0)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.model_ema = ema_update_model(
            model_to_update=self.model_ema,
            model_to_merge=self.model,
            momentum=self.cfg.MISC.MOMENTUM_SRC,
            device=self.device,
            update_all=True,
        )

        self.model.to("cpu")
        self.model_ema.to("cpu")

        _, predicted = torch.max(outputs, 1)
        correct = (predicted == self.y.to(self.device)).sum().item()
        self.correct_preds_before_adapt.append(correct)
        self.total_preds.append(len(self.y))

    def update_pvec(self):
        """Updates the principal vector of the instance by performing PCA on the input data."""
        pca = PCA(n_components=1)
        pca.fit(self.x.reshape(self.x.shape[0], -1))
        self.pvec = pca.components_[0]

    def update_acc(self, model=None):
        """
        Updates the accuracy of the model by comparing the predicted outputs with the true labels.

        Args:
           model (torch.nn.Module, optional): The model to be evaluated.
           If None, the model_ema will be used.

        Returns:
            None
        """
        if model is not None:
            model.to(self.device)
            _, outputs = model(self.x.to(self.device))
            model.to("cpu")
        else:
            self.model.to(self.device)
            _, outputs = self.model(self.x.to(self.device))
            self.model.to('cpu')
        
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == self.y.to(self.device)).sum().item()
        self.correct_preds_after_adapt.append(correct)

    def setup_optimizer(self):
        """Set up optimizer for tent adaptation."""
        if self.cfg.OPTIM.METHOD == "Adam":
            return optim.Adam(
                self.params,
                lr=self.cfg.OPTIM.LR,
                betas=(self.cfg.OPTIM.BETA, 0.999),
                weight_decay=self.cfg.OPTIM.WD,
            )

        elif self.cfg.OPTIM.METHOD == "SGD":
            return optim.SGD(
                self.params,
                lr=self.cfg.OPTIM.LR,
                momentum=self.cfg.OPTIM.MOMENTUM,
                dampening=self.cfg.OPTIM.DAMPENING,
                weight_decay=self.cfg.OPTIM.WD,
                nesterov=self.cfg.OPTIM.NESTEROV,
            )
        else:
            raise NotImplementedError(f"Unknown optimizer: {self.cfg.optim_method}")

    def configure_model(self):
        """
        Configures the model by setting the model to train mode and enabling
        all trainable parameters.

        Returns:
            None
        """
        # self.model.train()
        self.model.train()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
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
                m.train()  # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)

    def collect_params(self):
        """
        Collects the affine scale + shift parameters from normalization layers.

        Returns:
            list: A list containing the parameters of the normalization layers.
            list: A list containing the names of the normalization layers.
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(
                m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
            ):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def extract_bn_weights_and_biases(self):
        """
        Extracts the weights and biases of all normalization layers in the model.

        Returns:
            dict: A dictionary containing the weights and biases of all
            normalization layers in the model.
        """
        bn_params = {}
<<<<<<< HEAD
        for name, layer in self.model_ema.named_modules():
            if isinstance(
                layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
            ):
=======
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
>>>>>>> upstream/main
                gamma = layer.weight.data.cpu()  # Scale (weight)
                beta = layer.bias.data.cpu()  # Offset (bias)
                weights = torch.cat((gamma, beta), dim=0)
                bn_params[name] = weights
        return bn_params

    def get_state_dict(self):
        """
        Retrieves the state dictionary of the model.

        Returns:
            dict: A dictionary containing the state of the model.
        """
        return self.model.state_dict()

    def set_state_dict(self, state_dict):
        """
        Load the given state dictionary into the model.

        Args:
            state_dict (dict): A dictionary containing the model's state.
        """
        self.model.load_state_dict(state_dict)

    def get_model(self):
        """
        Retrieves the current model.

        Returns:
            The model instance.
        """
        return self.model
