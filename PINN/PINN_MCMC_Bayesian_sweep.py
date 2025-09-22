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
from pinn.train_bayesian_mcmc import train

run = wandb.init(
        reinit="finish_previous",
        entity="viriyadhika1",
        project="pinn-lab1",
        name="Bayesian PINN"
)

config = wandb.config

if __name__ == '__main__':
    train(wandb_run=run, lr=config.lr, sigma_w=config.sigma_w, beta_factor=config.beta_factor,epochs=10000, burn_in=10000)