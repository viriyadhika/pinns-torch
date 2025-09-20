import torch
from torch import nn
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import requests
import os
import logging

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info("Using " + device)

class Util:
    def __init__(self):
        pass

    def lhs(self, n_samples: int, n_dims: int, dtype=torch.float64, show_plot: bool = False):
        spaces = torch.linspace(0, 1, n_samples + 1, dtype=dtype).to(device)

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
    def __init__(self, util: Util, Exact, x_np, n_data, n_boundary, n_collocation, t_bound, x_bound):
        # data_points_X = n_data x (t, x)
        # collocation_X = n_data x (t, x)

        # x, t
        rand_data_idx = torch.randint(0, x_np.shape[0], size=[n_data], device=device)

        self.x_data = torch.tensor(x_np, device=device)[rand_data_idx].squeeze()
        self.t_data = torch.zeros(size=(n_data, 1), device=device).squeeze()

        Exact = torch.tensor(Exact, device=device)

        real_part = torch.real(Exact[rand_data_idx,0])
        imag_part = torch.imag(Exact[rand_data_idx,0])
        self.y_train = torch.stack([real_part, imag_part], dim=1)


        # x, t
        self.t_boundary = torch.rand(size=(n_boundary,), dtype=torch.float64, device=device) * (t_bound[1] - t_bound[0]) + t_bound[0]

        self.t_low_boundary = self.t_boundary.clone()
        self.x_low_boundary = torch.ones(n_boundary, device=device) * x_bound[0]


        self.t_high_boundary = self.t_boundary.clone()
        self.x_high_boundary= torch.ones(n_boundary, device=device) * x_bound[1]

        # Get collocation points
        collocation_points = util.lhs(n_collocation, 2, dtype=torch.float64)

        self.x_collocation_points = x_bound[0] + collocation_points[:,0] * (x_bound[1] - x_bound[0])
        self.t_collocation_points = t_bound[0] + collocation_points[:,1] * (t_bound[1] - t_bound[0])


class SchrodingerModel(nn.Module):
    def __init__(self, n_input: int, n_layer: int, n_out: int):
        super().__init__()
        n_hidden = 100
        self.first = self.block(n_input, n_hidden)
        self.hidden = nn.Sequential(*[self.block(n_hidden, n_hidden) for i in range(n_layer) ])
        self.last = nn.Linear(n_hidden, n_out, dtype=torch.float64)

        self.apply(self._init_weights)

    def block(self, n_input, n_hidden):
        return nn.Sequential(*[
            nn.Linear(n_input, n_hidden, dtype=torch.float64),
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
        X = torch.stack([x, t], dim=1)
        return self.last(self.hidden(self.first(X)))


def get_boundary_loss(schrodinger_model: SchrodingerModel, schrodinger_data: SchrodingerData):
    x_high_boundary = schrodinger_data.x_high_boundary.clone().requires_grad_(True)
    x_low_boundary = schrodinger_data.x_low_boundary.clone().requires_grad_(True)

    upper_bound = schrodinger_model(x_high_boundary, schrodinger_data.t_high_boundary)
    lower_bound = schrodinger_model(x_low_boundary, schrodinger_data.t_low_boundary)

    u_x_upper = torch.autograd.grad(
        torch.sum(upper_bound[:,0]),
        x_high_boundary,
        create_graph=True,
    )[0]

    v_x_upper = torch.autograd.grad(
        torch.sum(upper_bound[:,1]),
        x_high_boundary,
        create_graph=True,
    )[0]

    u_x_lower = torch.autograd.grad(
        torch.sum(lower_bound[:,0]),
        x_low_boundary,
        create_graph=True,
    )[0]

    v_x_lower = torch.autograd.grad(
        torch.sum(lower_bound[:,1]),
        x_low_boundary,
        create_graph=True,
    )[0]

    boundary_loss = (v_x_lower - v_x_upper)**2 + (u_x_lower - u_x_upper)**2
    boundary_loss = torch.mean(boundary_loss)

    return boundary_loss

def get_function_loss(schrodinger_model: SchrodingerModel, schrodinger_data: SchrodingerData):
    x_collocation_points = schrodinger_data.x_collocation_points.clone().requires_grad_(True)
    t_collocation_points = schrodinger_data.t_collocation_points.clone().requires_grad_(True)

    h_collocation = schrodinger_model(x_collocation_points, t_collocation_points)

    u = h_collocation[:,0]
    v = h_collocation[:,1]

    u_x = torch.autograd.grad(
        u,
        x_collocation_points,
        torch.ones_like(u),
        create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x,
        x_collocation_points,
        torch.ones_like(u_x),
        create_graph=True
    )[0]

    u_t = torch.autograd.grad(
        u,
        t_collocation_points,
        torch.ones_like(u),
        create_graph=True,
    )[0]

    v_x = torch.autograd.grad(
        v,
        x_collocation_points,
        torch.ones_like(v),
        create_graph=True
    )[0]

    v_xx = torch.autograd.grad(
        v_x,
        x_collocation_points,
        torch.ones_like(v_x),
        create_graph=True
    )[0]


    v_t = torch.autograd.grad(
        v,
        t_collocation_points,
        torch.ones_like(v),
        create_graph=True
    )[0]

    f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v
    f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u

    return torch.mean(f_u**2) + torch.mean(f_v**2)

def get_loss(schrodinger_model: SchrodingerModel, schrodinger_data: SchrodingerData):
    data_y = schrodinger_model(schrodinger_data.x_data, schrodinger_data.t_data)
    data_loss = torch.mean((data_y - schrodinger_data.y_train)**2)

    boundary_loss = get_boundary_loss(schrodinger_model, schrodinger_data)
    f_loss = get_function_loss(schrodinger_model, schrodinger_data)


    return data_loss, boundary_loss, f_loss

