from typing import Dict, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math

class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, prior_std: float):
        super().__init__()
        # Mean and log variance of weight distribution
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-3, 0.1))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.ones(out_features) * -3)
        self.prior_std = prior_std

    def forward(self, x, sample: bool = True):
        if sample:
            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps = torch.randn_like(self.bias_mu)
            weight = self.weight_mu + torch.exp(0.5 * self.weight_logvar) * weight_eps
            bias = self.bias_mu + torch.exp(0.5 * self.bias_logvar) * bias_eps
        else:
            weight = self.weight_mu
            bias  = self.bias_mu
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        prior_var = self.prior_std ** 2
        post_var = torch.exp(self.weight_logvar)
        kl = 0.5 * (
            (post_var + self.weight_mu**2) / prior_var
            - 1
            + math.log(prior_var) - self.weight_logvar
        ).sum()
        # same for bias
        post_var_bias = torch.exp(self.bias_logvar)
        kl += 0.5 * (
            (post_var_bias + self.bias_mu**2) / prior_var
            - 1
            + math.log(prior_var) - self.bias_logvar
        ).sum()
        return kl

class BayesianFCN(nn.Module):
    """A simple fully-connected neural net for solving equations.

    In this model, lower and upper bound will be used for normalization of input data
    """
    output_names: List[str]
    
    def __init__(self, n_input: int, n_layer: int, n_out: int, x_bound: list[float], t_bound: list[float], prior_std=0.1) -> None:
        """Initialize a `FCN` module.

        :param layers: The list indicating number of neurons in each layer.
        :param lb: Lower bound for the inputs.
        :param ub: Upper bound for the inputs.
        :param output_names: Names of outputs of net.
        :param discrete: If the problem is discrete or not.
        """
        super().__init__()
        n_hidden = 50
        self.x_lb: torch.Tensor
        self.x_ub: torch.Tensor
        self.t_lb: torch.Tensor
        self.t_ub: torch.Tensor
        # store bounds for scaling
        self.register_buffer("x_lb", torch.tensor(x_bound[0], dtype=torch.float32))
        self.register_buffer("x_ub", torch.tensor(x_bound[1], dtype=torch.float32))
        self.register_buffer("t_lb", torch.tensor(t_bound[0], dtype=torch.float32))
        self.register_buffer("t_ub", torch.tensor(t_bound[1], dtype=torch.float32))
        
        self.first = self.block(n_input, n_hidden, prior_std)
        self.hidden: nn.ModuleList = nn.ModuleList([self.block(n_hidden, n_hidden, prior_std) for _ in range(n_layer)])
        self.last = BayesianLinear(n_hidden, n_out, prior_std=prior_std)

        self.apply(self._init_weights)

    def block(self, n_input, n_hidden, prior_std) -> nn.Sequential:
        return nn.Sequential(*[
            BayesianLinear(n_input, n_hidden, prior_std),
            nn.Tanh()
        ])
    
    def _init_weights(self, module):
        """Apply Xavier initialization to linear layers"""
        if isinstance(module, BayesianLinear):
            # Xavier uniform initialization (also called Glorot uniform)
            nn.init.xavier_uniform_(module.weight_mu)
            if module.bias_mu is not None:
                nn.init.zeros_(module.bias_mu)

    def forward(self, x, t, sample=True) -> torch.Tensor:
        x_scaled = 2.0 * (x - self.x_lb) / (self.x_ub - self.x_lb) - 1.0
        t_scaled = 2.0 * (t - self.t_lb) / (self.t_ub - self.t_lb) - 1.0
        X = torch.stack([x_scaled, t_scaled], dim=1)
        h = self.first[0](X, sample)  # BayesianLinear
        h = self.first[1](h)          # Tanh
        for layer in self.hidden:
            h = layer[0](h, sample)
            h = layer[1](h)
        
        return self.last(h, sample)

    def kl_divergence(self):
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for m in self.modules():
            if isinstance(m, BayesianLinear):
                kl += m.kl_divergence()
        return kl

