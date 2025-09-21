
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
from pinn.bayesian import BayesianFCN
import wandb

run = wandb.init(
        reinit="finish_previous",
        entity="viriyadhika1",
        project="pinn-lab1",
        name="Bayesian PINN"
)

# Sweep code
config = wandb.config

logging.basicConfig(
    filename='log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info("Using " + device)

torch.cuda.manual_seed(42)
torch.manual_seed(42)

if __name__ == '__main__':
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
    schrodinger_data = SchrodingerData(util,
                                   Exact,
                                   x_np,
                                   n_data=50,
                                   n_boundary=50,
                                   n_collocation=20000,
                                   t_bound=[0, torch.pi / 2],
                                   x_bound=[-5, 5]
                                  )
    
    
    clip_norm = config.clip_norm
    beta_scaling = config.beta_scaling
    lr = config.lr
    mc = config.mc
    prior_std = config.prior_std

    epochs = 10000

    bayesian_fcn = BayesianFCN(n_input=2,n_layer=3, n_out=2, t_bound=[0, torch.pi / 2],
                                       x_bound=[-5, 5], prior_std=prior_std)
    optimizer = torch.optim.Adam(bayesian_fcn.parameters(), lr=lr)
    bayesian_fcn.to(device)

    for i in range(epochs):
        optimizer.zero_grad()

        
        data_loss = boundary_loss = f_loss = torch.tensor(0.0, device=device)
        for _ in range(mc):
            dl, bl, fl = get_loss(bayesian_fcn, schrodinger_data)
            data_loss += dl / mc; boundary_loss += bl / mc; f_loss += fl / mc

        kl_divergence = beta_scaling * bayesian_fcn.kl_divergence()
        loss = data_loss + boundary_loss + f_loss + kl_divergence

        loss.backward()

        torch.nn.utils.clip_grad_norm_(bayesian_fcn.parameters(), clip_norm)

        optimizer.step()

        run.log({
            'loss': loss.item(),
            'boundary_loss': boundary_loss.item(),
            'pde_loss': f_loss.item(),
            'data_loss': data_loss.item(),
            'kl_div': kl_divergence.item()
        })

        if i % 1000 == 0:
            logging.info(loss)
            os.makedirs("Bayesian", exist_ok=True)
            checkpoint_path = f"Bayesian/schrodinger_model-{i}.pt"
            torch.save({
                'model_state_dict': bayesian_fcn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': i,                # optional: last completed epoch
            }, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")

