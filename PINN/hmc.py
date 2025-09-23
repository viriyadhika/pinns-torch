# --- BAYESIAN SCHRÖDINGER (NLS) VIA HMC / hamiltorch -------------------------
# Requirements:
#   pip install hamiltorch torch matplotlib numpy scipy requests
# Assumes your repo provides pinn.lib.{SchrodingerModel, SchrodingerData, Util}
# and util.{sample_model_bpinns} (B-PINNs helpers you pasted earlier).

import os, math, logging, requests
import numpy as np
import scipy.io
import torch
from torch import nn

import hamiltorch
from pinn.lib import SchrodingerHMCModel, SchrodingerData, Util
import util as bpinns_util  # <-- your B-PINNs util with sample_model_bpinns
import wandb

# --------------------------
# Setup
# --------------------------
logging.basicConfig(
    filename='hmc_log.log', level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
run = wandb.init(
        reinit="finish_previous",
        entity="viriyadhika1",
        project="pinn-lab1",
        name="HMC"
)
device = "cuda" if torch.cuda.is_available() else "cpu"
hamiltorch.set_random_seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
logging.info(f"Using {device}")

# --------------------------
# Data: download + load NLS
# --------------------------
os.makedirs("Data", exist_ok=True)
mat_path = "Data/NLS.mat"
if not os.path.exists(mat_path):
    url = "https://github.com/maziarraissi/PINNs/raw/master/main/Data/NLS.mat"
    r = requests.get(url)
    with open(mat_path, "wb") as f:
        f.write(r.content)
    logging.info("Downloaded NLS.mat to Data/NLS.mat")

data = scipy.io.loadmat(mat_path)
# Raw arrays from .mat
t_np = data['tt'].flatten()[:, None]   # shape [Nt, 1]
x_np = data['x'].flatten()[:, None]    # shape [Nx, 1]
Exact = data['uu']                     # complex array [Nx, Nt]

# --------------------------
# Validation set (from Exact)
# --------------------------
# Build full spacetime grid (validation against the exact solution)
X_val_np, T_val_np = np.meshgrid(x_np.flatten(), t_np.flatten(), indexing="ij")
x_val = torch.tensor(X_val_np.ravel(), dtype=torch.float32, device=device).unsqueeze(1)
t_val = torch.tensor(T_val_np.ravel(), dtype=torch.float32, device=device).unsqueeze(1)
y_val = torch.tensor(
    np.stack([Exact.real.ravel(), Exact.imag.ravel()], axis=1),  # [N_val, 2]
    dtype=torch.float32, device=device
)

# --------------------------
# Training subsets for PINN terms
# --------------------------
util_helper = Util()
# Choose bounds consistent with the data (x ≈ [-5,5], t in [0, pi/2])
x_bound = [-5.0, 5.0]
t_bound = [0.0, float(np.pi/2)]

# These are moderate; tune as you like
n_data = 50
n_boundary = 50
n_collocation = 20000

sch_data = SchrodingerData(
    util=util_helper,
    Exact=Exact,         # complex grid [Nx, Nt]
    x_np=x_np,           # spatial nodes (column)
    n_data=n_data,
    n_boundary=n_boundary,
    n_collocation=n_collocation,
    t_bound=t_bound,
    x_bound=x_bound
)

# Draw *fixed* samples once (HMC likelihood must be stationary)
# Data term (t=0), y is [Re, Im]
x_data, t_data, y_train = sch_data.sample_data()
x_data = x_data.view(-1, 1).to(device)
t_data = t_data.view(-1, 1).to(device)
y_train = y_train.to(device)

# Boundary term (periodic)
t_low_b, x_low_b, t_high_b, x_high_b = sch_data.sample_boundary()
x_low_b = x_low_b.view(-1, 1).to(device).requires_grad_(True)
x_high_b = x_high_b.view(-1, 1).to(device).requires_grad_(True)
t_low_b = t_low_b.view(-1, 1).to(device).requires_grad_(True)
t_high_b = t_high_b.view(-1, 1).to(device).requires_grad_(True)

# PDE collocation
x_f, t_f = sch_data.sample_collocation()
x_f = x_f.view(-1, 1).to(device).requires_grad_(True)
t_f = t_f.view(-1, 1).to(device).requires_grad_(True)

# Bundle all the static tensors
bayes_data = {
    'data': (x_data, t_data, y_train),
    'boundary': (x_low_b, t_low_b, x_high_b, t_high_b),
    'collocation': (x_f, t_f),
    'val': (x_val, t_val, y_val)  # include exact val targets for convenience
}

# --------------------------
# Model
# --------------------------
net = SchrodingerHMCModel(
    n_input=2, n_layer=3, n_out=2,
    t_bound=t_bound, x_bound=x_bound
).to(device)

# --------------------------
# Likelihood precisions (taus)
# --------------------------
# Std of each loss component (higher std = weaker penalty)
like_std_data = 1.0
like_std_bnd  = 1.0
like_std_pde  = 1.0
tau_likes = [1.0/(like_std_data**2), 1.0/(like_std_bnd**2), 1.0/(like_std_pde**2)]

# Prior precision on weights
prior_std = 1.0
tau_prior = 1.0 / (prior_std**2)

# --------------------------
# Schrödinger B-PINN log-likelihood
# --------------------------
# Signature must match your BPINNs util: model_loss(data, fmodel, params_unflattened, tau_likes, gradients_fn, params_single=None)
def schrodinger_model_loss(data, fmodel, params_unflattened, tau_likes, gradients_fn, params_single=None):
    # QUICK NAN/INF GUARD — safer HMC (rejection instead of poisoning the chain)
    def _guard(*tensors):
        for t in tensors:
            if torch.isnan(t).any() or torch.isinf(t).any():
                raise bpinns_util.LogProbError()

    (x_d, t_d, y_d) = data['data']
    (x_lb, t_lb, x_hb, t_hb) = data['boundary']
    (x_c, t_c) = data['collocation']

    tau_data, tau_bnd, tau_pde = tau_likes  # flat tensor of length 3

    # ---- Data term (measurements at t=0): MSE on [u,v]
    pred_data = fmodel[0](x_d, t_d, params=params_unflattened[0])  # [N,2]
    _guard(pred_data, y_d)
    pred_err = (pred_data - y_d)**2
    ll = -0.5 * tau_data * torch.mean(pred_err)

    # ---- Periodic boundary: match function & first derivative across x endpoints
    pred_low  = fmodel[0](x_lb, t_lb, params=params_unflattened[0])  # [Nb,2]
    pred_high = fmodel[0](x_hb, t_hb, params=params_unflattened[0])  # [Nb,2]
    u_low, v_low   = pred_low[:, 0],  pred_low[:, 1]
    u_high, v_high = pred_high[:, 0], pred_high[:, 1]

    u_x_low  = gradients_fn(u_low,  x_lb)
    v_x_low  = gradients_fn(v_low,  x_lb)
    u_x_high = gradients_fn(u_high, x_high_b)
    v_x_high = gradients_fn(v_high, x_high_b)

    bnd_err = (u_low - u_high)**2 + (v_low - v_high)**2 \
            + (u_x_low - u_x_high)**2 + (v_x_low - v_x_high)**2
    _guard(bnd_err)
    ll = ll - 0.5 * tau_bnd * torch.mean(bnd_err)

    # ---- PDE residual (f_u, f_v) at collocation
    pred_f = fmodel[0](x_c, t_c, params=params_unflattened[0])  # [Nf,2]
    u, v = pred_f[:, 0], pred_f[:, 1]

    u_x  = gradients_fn(u, x_c)
    u_xx = gradients_fn(u_x, x_c)
    v_x  = gradients_fn(v, x_c)
    v_xx = gradients_fn(v_x, x_c)
    u_t  = gradients_fn(u, t_c)
    v_t  = gradients_fn(v, t_c)

    # NLS/Schrödinger system (f_u, f_v) = 0
    f_u = u_t + 0.5 * v_xx + (u**2 + v**2) * v
    f_v = v_t - 0.5 * u_xx - (u**2 + v**2) * u
    pde_res = f_u**2 + f_v**2
    _guard(pde_res)
    ll = ll - 0.5 * tau_pde * torch.mean(pde_res)

    run.log({
        'loss': ll.item(),
        'boundary_loss': torch.mean(bnd_err),
        'pde_loss': torch.mean(pde_res),
        'data_loss': torch.mean(pred_err)
    })

    # Return outputs for optional inspection
    return ll, [pred_data, pred_low, pred_high, pred_f]

# --------------------------
# HMC hyperparameters
# --------------------------
# Start conservatively; you can increase step_size or L later.
step_size = 5e-6
L = 10
burn = 10000
num_samples = 15000

# --------------------------
# Run HMC
# --------------------------
nets = [net]
params_hmc = bpinns_util.sample_model_bpinns(
    nets,
    bayes_data,
    model_loss=schrodinger_model_loss,
    num_samples=num_samples,
    num_steps_per_sample=L,
    step_size=step_size,
    burn=burn,
    tau_priors=tau_prior,   # scalar prior precision
    tau_likes=tau_likes,    # tensor([tau_data, tau_bnd, tau_pde])
    device=device,
    pde=True,
    pinns=False,
    epochs=0
)

# --------------------------
# Posterior predictive on full validation grid
# --------------------------
with torch.no_grad():
    # Unflatten each sample into net weights, run forward on (x_val, t_val)
    preds = []
    flat_len = hamiltorch.util.flatten(net).numel()

    for s in params_hmc:
        # Only one net; entire vector belongs to it
        params_unflat = [hamiltorch.util.unflatten(net, s[:flat_len])]
        y_pred = nets[0](x_val, t_val)
        preds.append(y_pred.detach().cpu())

    preds = torch.stack(preds, dim=0)   # [S, N_val, 2]
    pred_mean = preds.mean(dim=0)       # [N_val, 2]
    pred_std  = preds.std(dim=0)        # [N_val, 2]

    # Validation MSE on mean prediction
    mse_val = torch.mean((pred_mean.to(device) - y_val)**2).item()

print(f"Validation MSE (mean prediction): {mse_val:.6e}")
logging.info(f"Validation MSE (mean prediction): {mse_val:.6e}")

# Optional: save posterior predictive summary
os.makedirs("Bayesian", exist_ok=True)
torch.save(
    {
        "samples": params_hmc,
        "x_val": x_val.detach().cpu(),
        "t_val": t_val.detach().cpu(),
        "y_val": y_val.detach().cpu(),
        "pred_mean": pred_mean.detach().cpu(),
        "pred_std": pred_std.detach().cpu(),
        "val_mse_mean_pred": mse_val
    },
    "Bayesian/hmc_posterior_predictive.pt"
)

print("HMC run complete. Posterior predictive saved to Bayesian/hmc_posterior_predictive.pt")
