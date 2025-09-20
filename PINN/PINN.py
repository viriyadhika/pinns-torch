#!/usr/bin/env python
# coding: utf-8

# # PDE Solution

# In[5]:


import torch
from torch import nn
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import requests
import os
import logging
from lib.lib import SchrodingerModel, SchrodingerData, Util, get_loss
import wandb

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
    
    epochs = 30000
    schrodinger_model = SchrodingerModel(n_input=2, n_layer=3, n_out=2)
    optimizer = torch.optim.Adam(schrodinger_model.parameters())
    schrodinger_model.to(device)
    checkpoint_path = "./schrodinger_model.pt"

    if os.path.exists('./schrodinger_model.pt'):
        logging.info("Path Exist, loading weight")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        schrodinger_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    run = wandb.init(
            reinit="finish_previous",
            entity="viriyadhika1",
            project="pinn-lab1",
            name="Schrodinger PINN",
            config={
                "model": "Schrodinger PINN",
                "epochs": epochs
            },
    )
    for i in range(epochs):
        optimizer.zero_grad()
        data_loss, boundary_loss, f_loss = get_loss(schrodinger_model, schrodinger_data)
        loss = data_loss + boundary_loss + f_loss
        loss.backward()
        optimizer.step()

        run.log({
            'loss': loss.item(),
            'boundary_loss': boundary_loss.item(),
            'pde_loss': f_loss.item(),
            'data_loss': data_loss.item()
        })

        
        if i % 1000 == 0:
            logging.info(loss)
            checkpoint_path = f"4Refactor/schrodinger_model-{i}.pt"
            torch.save({
                'model_state_dict': schrodinger_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': i,                # optional: last completed epoch
            }, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")

    run.finish()
