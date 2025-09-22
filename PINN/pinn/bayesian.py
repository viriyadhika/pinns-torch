from typing import Dict, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math

class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, prior_std: float):
        super().__init__()
        # Mean of weight distribution
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))

        # Variance parametrized with rho (softplus → std)
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features).uniform_(-5, -4))
        self.bias_rho   = nn.Parameter(torch.empty(out_features).uniform_(-5, -4))

        self.prior_std = prior_std
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier initialization for means
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.zeros_(self.bias_mu)

    def forward(self, x, sample: bool = True):
        weight_sigma = F.softplus(self.weight_rho)   # ensures >0
        bias_sigma   = F.softplus(self.bias_rho)

        if sample:
            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps   = torch.randn_like(self.bias_mu)
            weight = self.weight_mu + weight_sigma * weight_eps
            bias   = self.bias_mu + bias_sigma * bias_eps
        else:
            weight = self.weight_mu
            bias   = self.bias_mu

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        prior_var = self.prior_std ** 2
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma   = F.softplus(self.bias_rho)

        # KL between posterior N(mu, sigma²) and prior N(0, prior_std²)
        kl = 0.5 * (
            (weight_sigma.pow(2) + self.weight_mu.pow(2)) / prior_var
            - 1
            + 2 * (math.log(self.prior_std) - torch.log(weight_sigma))
        ).sum()

        kl += 0.5 * (
            (bias_sigma.pow(2) + self.bias_mu.pow(2)) / prior_var
            - 1
            + 2 * (math.log(self.prior_std) - torch.log(bias_sigma))
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
        n_hidden = 100
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

