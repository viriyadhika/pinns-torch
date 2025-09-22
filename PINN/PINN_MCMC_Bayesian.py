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


if __name__ == '__main__':
    train(wandb_run=run, beta_factor=6e-4, lr=1e-4, sigma_w=2, epochs=35000, burn_in=30000)
