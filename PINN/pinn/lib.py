import torch
from torch import nn
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import requests
import os
import logging

from pinn.bayesian import BayesianFCN

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info("Using " + device)

class Util:
    def __init__(self):
        pass

    def lhs(self, n_samples: int, n_dims: int, show_plot: bool = False):
        spaces = torch.linspace(0, 1, n_samples + 1).to(device)

        # We need to unsqueeze so that we can use broadcasting on the n_dims
        lower_bound = spaces[:-1].unsqueeze(1)
        upper_bound = spaces[1:].unsqueeze(1)

        rand_idx = torch.rand(n_samples, n_dims, device=device)
        points = lower_bound + rand_idx * (upper_bound - lower_bound)

        if show_plot:
            self.print_2d_space(points)

        for i in range(n_dims):
            samples_idx = torch.randperm(n_samples, device=device)
            points[:,i] = points[samples_idx,i]

        if show_plot:
            self.print_2d_space(points)

        return points

    def print_2d_space(self, X):
        plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), color='blue', s=50)
        plt.show()


class SchrodingerData:
    def __init__(self, util: Util, Exact, x_np, n_data, n_boundary, n_collocation, t_bound: list[float], x_bound: list[float]):
        # data_points_X = n_data x (t, x)
        # collocation_X = n_data x (t, x)

        # x, t
        self.x_np = torch.tensor(x_np, device=device, dtype=torch.float32)
        self.Exact = Exact
        self.n_data = n_data

        self.n_boundary = n_boundary
        self.util = util
        self.n_collocation = n_collocation
        self.x_bound = x_bound
        self.t_bound = t_bound

    def sample_data(self):
        rand_data_idx = torch.randint(0, self.x_np.shape[0], size=[self.n_data], device=device)

        x_data = self.x_np[rand_data_idx].squeeze()
        t_data = torch.zeros(size=(self.n_data, 1), device=device, dtype=torch.float32).squeeze()

        Exact = torch.tensor(self.Exact, device=device)

        real_part = torch.real(Exact[rand_data_idx,0])
        imag_part = torch.imag(Exact[rand_data_idx,0])
        y_train = torch.stack([real_part, imag_part], dim=1)

        return x_data, t_data, y_train

    def sample_collocation(self):
        # Get collocation points
        collocation_points = self.util.lhs(self.n_collocation, 2)

        x_collocation_points = self.x_bound[0] + collocation_points[:,0] * (self.x_bound[1] - self.x_bound[0])
        t_collocation_points = self.t_bound[0] + collocation_points[:,1] * (self.t_bound[1] - self.t_bound[0])

        return x_collocation_points, t_collocation_points

    def sample_boundary(self):
        t_boundary = torch.rand(size=(self.n_boundary,), device=device) * (self.t_bound[1] - self.t_bound[0]) + self.t_bound[0]

        t_low_boundary = t_boundary.clone()
        x_low_boundary = torch.ones(self.n_boundary, device=device) * self.x_bound[0]


        t_high_boundary = t_boundary.clone()
        x_high_boundary= torch.ones(self.n_boundary, device=device) * self.x_bound[1]

        return t_low_boundary, x_low_boundary, t_high_boundary, x_high_boundary


class SchrodingerHMCModel(nn.Module):
    def __init__(self, n_input: int, n_layer: int, n_out: int, x_bound: list[float], t_bound: list[float]):
        super().__init__()
        n_hidden = 100
        # store bounds for scaling
        self.x_lb = float(x_bound[0])
        self.x_ub = float(x_bound[1])
        self.t_lb = float(t_bound[0])
        self.t_ub = float(t_bound[1])
        
        self.first = self.block(n_input, n_hidden)
        self.hidden = nn.Sequential(*[self.block(n_hidden, n_hidden) for i in range(n_layer) ])
        self.last = nn.Linear(n_hidden, n_out)

        self.apply(self._init_weights)

    def block(self, n_input, n_hidden):
        return nn.Sequential(*[
            nn.Linear(n_input, n_hidden),
            nn.Tanh()
        ])

    def _init_weights(self, module):
        """Apply Xavier initialization to linear layers"""
        if isinstance(module, nn.Linear):
            # Xavier uniform initialization (also called Glorot uniform)
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, t):
        x_scaled = 2.0 * (x - self.x_lb) / (self.x_ub - self.x_lb) - 1.0
        t_scaled = 2.0 * (t - self.t_lb) / (self.t_ub - self.t_lb) - 1.0
        X = torch.cat([x_scaled, t_scaled], dim=1)
        return self.last(self.hidden(self.first(X)))




class SchrodingerModel(nn.Module):
    def __init__(self, n_input: int, n_layer: int, n_out: int, x_bound: list[float], t_bound: list[float]):
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
        
        self.first = self.block(n_input, n_hidden)
        self.hidden = nn.Sequential(*[self.block(n_hidden, n_hidden) for i in range(n_layer) ])
        self.last = nn.Linear(n_hidden, n_out)

        self.apply(self._init_weights)

    def block(self, n_input, n_hidden):
        return nn.Sequential(*[
            nn.Linear(n_input, n_hidden),
            nn.Tanh()
        ])

    def _init_weights(self, module):
        """Apply Xavier initialization to linear layers"""
        if isinstance(module, nn.Linear):
            # Xavier uniform initialization (also called Glorot uniform)
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, t):
        x_scaled = 2.0 * (x - self.x_lb) / (self.x_ub - self.x_lb) - 1.0
        t_scaled = 2.0 * (t - self.t_lb) / (self.t_ub - self.t_lb) - 1.0
        X = torch.stack([x_scaled, t_scaled], dim=1)
        return self.last(self.hidden(self.first(X)))


def gradients(dy: torch.Tensor, dx: torch.Tensor):
    return torch.autograd.grad(
        dy,
        dx,
        grad_outputs=torch.ones_like(dy),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]

def get_boundary_loss(
        upper_bound: torch.Tensor, 
        lower_bound: torch.Tensor, 
        x_low_boundary: torch.Tensor, 
        x_high_boundary: torch.Tensor
    ):

    u_upper = upper_bound[:,0]
    v_upper = upper_bound[:,1]
    u_lower = lower_bound[:,0]
    v_lower = lower_bound[:,1]
    
    u_x_upper = gradients(
        u_upper,
        x_high_boundary
    )
    
    v_x_upper = gradients(
        v_upper,
        x_high_boundary
    )
    
    u_x_lower = gradients(
        u_lower,
        x_low_boundary
    )
    
    v_x_lower = gradients(
        v_lower,
        x_low_boundary
    )
    
    boundary_loss = (v_x_lower - v_x_upper)**2 + (u_x_lower - u_x_upper)**2 + (u_upper - u_lower)**2 + (v_upper - v_lower)**2
    boundary_loss = torch.mean(boundary_loss)

    return boundary_loss

def get_function_loss(u: torch.Tensor, v: torch.Tensor, x_collocation_points: torch.Tensor, t_collocation_points: torch.Tensor):
    u_x = gradients(
        u,
        x_collocation_points,
    )
    u_xx = gradients(
        u_x,
        x_collocation_points,
    )
    u_t = gradients(
        u,
        t_collocation_points,
    )
    v_x = gradients(
        v,
        x_collocation_points,
    )
    v_xx = gradients(
        v_x,
        x_collocation_points,
    )
    v_t = gradients(
        v,
        t_collocation_points,
    )
    f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v
    f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u
    f_loss = torch.mean(f_u**2) + torch.mean(f_v**2)

    return f_loss

def get_loss(schrodinger_model: nn.Module, schrodinger_data: SchrodingerData):
    # Data loss
    x_data, t_data, y_train = schrodinger_data.sample_data()
    data_y = schrodinger_model(x_data, t_data)
    data_loss = torch.mean((data_y - y_train)**2)

    # Boundary loss
    t_low_boundary, x_low_boundary, t_high_boundary, x_high_boundary = schrodinger_data.sample_boundary()
    [i.requires_grad_(True) for i in [t_low_boundary, x_low_boundary, t_high_boundary, x_high_boundary]]
    upper_bound = schrodinger_model(x_high_boundary, t_high_boundary)
    lower_bound = schrodinger_model(x_low_boundary, t_low_boundary)
    boundary_loss = get_boundary_loss(upper_bound, lower_bound, x_low_boundary, x_high_boundary)

    ## Function loss
    x_collocation_points, t_collocation_points = schrodinger_data.sample_collocation()
    [i.requires_grad_(True) for i in [x_collocation_points, t_collocation_points]]
    h_collocation = schrodinger_model(x_collocation_points, t_collocation_points)
    u = h_collocation[:,0]
    v = h_collocation[:,1]
    f_loss = get_function_loss(u, v, x_collocation_points, t_collocation_points)

    return data_loss, boundary_loss, f_loss

