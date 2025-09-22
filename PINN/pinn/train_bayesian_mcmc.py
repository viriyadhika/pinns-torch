import torch
from torch import nn
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import requests
import os
import logging
from pinn.lib import SchrodingerModel, SchrodingerData, Util, get_loss
import wandb
import math

logging.basicConfig(
    filename='log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info("Using " + device)

torch.cuda.manual_seed(42)
torch.manual_seed(42)


def neg_log_prior(model: SchrodingerModel, sigma_w=1.0):
    s2 = sigma_w**2
    reg = torch.tensor(0.0, device=device)
    for p in model.parameters():
        reg = reg + torch.sum(p**2)
    return (1.0 / (2.0 * s2)) * reg


@torch.no_grad()
def sgld_step(model: SchrodingerModel, lr):
    for p in model.parameters():
        if p.grad is None: 
            continue
        noise = torch.randn_like(p)
        p.add_( -0.5 * lr * p.grad + math.sqrt(lr) * noise )


def train(wandb_run, lr=1e-5, sigma_w=1, beta_factor=1e-3, epochs=35000, burn_in=30000):
    os.makedirs("Data", exist_ok=True)
    url = "https://github.com/maziarraissi/PINNs/raw/master/main/Data/NLS.mat"
    r = requests.get(url)
    with open("Data/NLS.mat", "wb") as f:
        f.write(r.content)
    logging.info("Downloaded NLS.mat to Data/NLS.mat")

    data = scipy.io.loadmat('Data/NLS.mat')
    t_np = data['tt'].flatten()[:,None]
    x_np = data['x'].flatten()[:,None]
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    util = Util()
    
    T_np, X_np = np.meshgrid(t_np.flatten(), x_np.flatten())
    x_test, t_test = torch.tensor(X_np.ravel(), dtype=torch.float), torch.tensor(T_np.ravel(), dtype=torch.float)

    schrodinger_data = SchrodingerData(util,
                                   Exact,
                                   x_np,
                                   n_data=50,
                                   n_boundary=50,
                                   n_collocation=20000,
                                   t_bound=[0, torch.pi / 2],
                                   x_bound=[-5, 5]
                                  )
    
    

    sample_every = 10

    schrodinger_model = SchrodingerModel(n_input=2,n_layer=3, n_out=2, t_bound=[0, torch.pi / 2],
                                       x_bound=[-5, 5])

    schrodinger_model.to(device)
    samples = []

    for i in range(epochs):
        # Zero grad
        for p in schrodinger_model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        data_loss, boundary_loss, f_loss = get_loss(schrodinger_model, schrodinger_data)

        L_prior = beta_factor * neg_log_prior(schrodinger_model, sigma_w=sigma_w)

        loss = data_loss + boundary_loss + f_loss + L_prior

        loss.backward()

        sgld_step(schrodinger_model, lr)

        if i > burn_in and (i - burn_in) % sample_every == 0:
            schrodinger_model.eval()
            with torch.no_grad():
                u_pred = schrodinger_model(x_test, t_test).detach().cpu().numpy()
                samples.append(u_pred.squeeze())
            schrodinger_model.train()

        wandb_run.log({
            'loss': loss.item(),
            'boundary_loss': boundary_loss.item(),
            'pde_loss': f_loss.item(),
            'data_loss': data_loss.item(),
            'L_prior': L_prior.item()
        })

        if i % 1000 == 0:
            logging.info(loss)
            os.makedirs("Bayesian", exist_ok=True)
            checkpoint_path = f"Bayesian/schrodinger_model-{i}.pt"
            torch.save({
                'model_state_dict': schrodinger_model.state_dict(),
                'epoch': i,                # optional: last completed epoch
            }, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")

    checkpoint_path = f"Bayesian/samples.pt"
    torch.save({ "samples": samples }, checkpoint_path)

