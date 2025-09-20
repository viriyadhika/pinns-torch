from typing import Dict, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=0.1):
        super().__init__()
        # Mean and log variance of weight distribution
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-3, 0.1))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.ones(out_features) * -3)
        self.prior_std = prior_std

    def forward(self, x, sample=True):
        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)
        weight = self.weight_mu + torch.exp(0.5 * self.weight_logvar) * weight_eps
        bias = self.bias_mu + torch.exp(0.5 * self.bias_logvar) * bias_eps
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
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
    
    def __init__(self, layers, lb, ub, output_names, discrete: bool = False) -> None:
        """Initialize a `FCN` module.

        :param layers: The list indicating number of neurons in each layer.
        :param lb: Lower bound for the inputs.
        :param ub: Upper bound for the inputs.
        :param output_names: Names of outputs of net.
        :param discrete: If the problem is discrete or not.
        """
        super().__init__()

        self.model = self.initalize_net(layers)
        self.register_buffer("lb", torch.tensor(lb, dtype=torch.float32, requires_grad=False))
        self.register_buffer("ub", torch.tensor(ub, dtype=torch.float32, requires_grad=False))
        self.output_names = output_names
        self.discrete = discrete

    def initalize_net(self, layers: List):
        """Initialize the layers of the neural network.

        :param layers: The list indicating number of neurons in each layer.
        :return: The initialized neural network.
        """

        initializer = nn.init.xavier_uniform_
        net = nn.Sequential()

        input_layer = BayesianLinear(layers[0], layers[1])
        initializer(input_layer.weight_mu)

        net.add_module("input", input_layer)
        net.add_module("activation_1", nn.Tanh())

        for i in range(1, len(layers) - 2):
            hidden_layer = BayesianLinear(layers[i], layers[i + 1])
            initializer(hidden_layer.weight_mu)
            net.add_module(f"hidden_{i+1}", hidden_layer)
            net.add_module(f"activation_{i+1}", nn.Tanh())

        output_layer = BayesianLinear(layers[-2], layers[-1])
        initializer(output_layer.weight_mu)
        net.add_module("output", output_layer)
        return net

    def forward(self, spatial: List[torch.Tensor], time: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a single forward pass through the network.

        :param spatial: List of input spatial tensors.
        :param time: Input tensor representing time.
        :return: A tensor of solutions.
        """

        # Discrete Mode
        if self.discrete:
            if len(spatial) == 2:
                x, y = spatial
                z = torch.cat((x, y), 1)
            elif len(spatial) == 3:
                x, y, z = spatial
                z = torch.cat((x, y, z), 1)
            else:
                z = spatial[0]
            z = 2.0 * (z - self.lb[:-1]) / (self.ub[:-1] - self.lb[:-1]) - 1.0

        # Continuous Mode
        else:
            if len(spatial) == 1:
                x = spatial[0]
                z = torch.cat((x, time), 1)
            elif len(spatial) == 2:
                x, y = spatial
                z = torch.cat((x, y, time), 1)
            else:
                x, y, z = spatial
                z = torch.cat((x, y, z, time), 1)
            z = 2.0 * (z - self.lb) / (self.ub - self.lb) - 1.0

        z = self.model(z)

        # Discrete Mode
        if self.discrete:
            outputs_dict = {name: z for i, name in enumerate(self.output_names)}

        # Continuous Mode
        else:
            outputs_dict = {name: z[:, i : i + 1] for i, name in enumerate(self.output_names)}
        return outputs_dict

    def kl_divergence(self):
        kl = 0.0
        for m in self.model.modules():
            if isinstance(m, BayesianLinear):
                kl += m.kl_divergence()
        return kl

if __name__ == "__main__":
    _ = BayesianFCN()
