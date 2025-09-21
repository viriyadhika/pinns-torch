# nls_pinn_torch.py
# PyTorch port of Raissi's NLS PINN (TF -> Torch)
# Author: you ✨

import time
import numpy as np
import scipy.io
from scipy.interpolate import griddata

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
import wandb
import os
import logging
import requests

# ---------- Utilities ----------

def lhs(n_dim: int, n_samples: int, dtype=np.float32):
    """Latin Hypercube Sampling in [0,1]^n_dim."""
    rng = np.random.default_rng()
    cut = np.linspace(0, 1, n_samples + 1)
    u = rng.random((n_samples, n_dim), dtype=dtype)
    a = cut[:n_samples]
    b = cut[1:n_samples + 1]
    rdpoints = u * (b - a)[:, None] + a[:, None]
    H = np.zeros_like(rdpoints)
    for j in range(n_dim):
        order = rng.permutation(n_samples)
        H[:, j] = rdpoints[order, j]
    return H.astype(dtype)

def to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

# ---------- Model ----------

class MLP(nn.Module):
    def __init__(self, layers, lb, ub, device):
        """
        layers: e.g. [2, 100, 100, 100, 100, 2]
        lb, ub: numpy arrays of shape (2,) for [x_min, t_min], [x_max, t_max]
        """
        super().__init__()
        self.lb = torch.as_tensor(lb, dtype=torch.float32, device=device)
        self.ub = torch.as_tensor(ub, dtype=torch.float32, device=device)

        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                net.append(nn.Tanh())
        self.net = nn.Sequential(*net)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                # Xavier/Glorot (same gain as tanh default)
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        # scale to [-1, 1] like the TF code
        X = torch.cat([x, t], dim=1)  # (N,2)
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(H)  # (N,2) -> [u, v]

class PhysicsInformedNN:
    def __init__(self, x0, u0, v0, tb, X_f, layers, lb, ub, device=None):
        """
        x0,u0,v0: initial condition points/values at t=0
        tb: times for boundary points (x=lb[0] and x=ub[0])
        X_f: collocation points in the interior
        layers: network sizes
        lb, ub: domain bounds (numpy arrays)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lb = to_tensor(lb, self.device)
        self.ub = to_tensor(ub, self.device)

        # Build training sets like TF code
        X0 = np.concatenate([x0, 0.0 * x0], axis=1)       # (x0, 0)
        X_lb = np.concatenate([0.0 * tb + lb[0], tb], 1)  # (lb[0], tb)
        X_ub = np.concatenate([0.0 * tb + ub[0], tb], 1)  # (ub[0], tb)

        self.x0 = to_tensor(X0[:, 0:1], self.device).requires_grad_(True)
        self.t0 = to_tensor(X0[:, 1:2], self.device).requires_grad_(True)

        self.x_lb = to_tensor(X_lb[:, 0:1], self.device).requires_grad_(True)
        self.t_lb = to_tensor(X_lb[:, 1:2], self.device).requires_grad_(True)

        self.x_ub = to_tensor(X_ub[:, 0:1], self.device).requires_grad_(True)
        self.t_ub = to_tensor(X_ub[:, 1:2], self.device).requires_grad_(True)

        self.x_f = to_tensor(X_f[:, 0:1], self.device).requires_grad_(True)
        self.t_f = to_tensor(X_f[:, 1:2], self.device).requires_grad_(True)

        self.u0 = to_tensor(u0, self.device)
        self.v0 = to_tensor(v0, self.device)

        # Model
        self.model = MLP(layers, lb, ub, self.device).to(self.device)

        # Optimizers (Adam then LBFGS)
        self.opt_adam = optim.Adam(self.model.parameters(), lr=1e-3)
        # Note: PyTorch LBFGS is unconstrained (no bounds). That's fine here.
        self.opt_lbfgs = optim.LBFGS(
            self.model.parameters(),
            max_iter=50000,
            max_eval=50000,
            tolerance_grad=1e-11,
            tolerance_change=1e-11,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        self.mse = nn.MSELoss()

    # ---- autograd helpers ----

    def _uv(self, x, t):
        """Returns u, v, u_x, v_x with autograd."""
        uv = self.model(x, t)  # (N,2)
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        v_x = grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        return u, v, u_x, v_x

    def _f_uv(self, x, t):
        """Physics residuals f_u, f_v for NLS system used in the TF code."""
        u, v, u_x, v_x = self._uv(x, t)

        u_t = grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        v_t = grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]

        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        v_xx = grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True, retain_graph=True)[0]

        # Raissi TF code:
        # f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v
        # f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u
        mag2 = u**2 + v**2
        f_u = u_t + 0.5 * v_xx + mag2 * v
        f_v = v_t - 0.5 * u_xx - mag2 * u
        return f_u, f_v

    # ---- loss ----

    def loss(self):
        # IC at t=0
        u0_pred, v0_pred, _, _ = self._uv(self.x0, self.t0)

        # periodic BC at x=lb[0] and x=ub[0]
        u_lb, v_lb, u_x_lb, v_x_lb = self._uv(self.x_lb, self.t_lb)
        u_ub, v_ub, u_x_ub, v_x_ub = self._uv(self.x_ub, self.t_ub)

        # physics residual at collocation points
        f_u, f_v = self._f_uv(self.x_f, self.t_f)

        loss_val = (
            self.mse(u0_pred, self.u0)
            + self.mse(v0_pred, self.v0)
            + self.mse(u_lb, u_ub)
            + self.mse(v_lb, v_ub)
            + self.mse(u_x_lb, u_x_ub)
            + self.mse(v_x_lb, v_x_ub)
            + self.mse(f_u, torch.zeros_like(f_u))
            + self.mse(f_v, torch.zeros_like(f_v))
        )
        return loss_val

    # ---- training ----

    def train_adam(self, n_iter=5000, log_every=10):
        self.model.train()
        t0 = time.time()
        run = wandb.init(
            reinit="finish_previous",
            entity="viriyadhika1",
            project="pinn-lab1",
            name="Schrodinger PINN Code",
            config={
                "model": "Schrodinger PINN Code",
            },
        )
        for it in range(1, n_iter + 1):
            self.opt_adam.zero_grad()
            L = self.loss()
            run.log({
                "loss": L.item()
            })
            L.backward()
            self.opt_adam.step()

            if it % log_every == 0:
                dt = time.time() - t0
                print(f"It {it:6d} | Loss {L.item():.3e} | {dt:.2f}s")
                t0 = time.time()

    def train_lbfgs(self):
        self.model.train()

        def closure():
            self.opt_lbfgs.zero_grad()
            L = self.loss()
            L.backward()
            print(f"LBFGS step | Loss {L.item():.3e}")
            return L

        self.opt_lbfgs.step(closure)

    # ---- inference ----

    @torch.no_grad()
    def predict(self, X_star):
        # expects X_star[:,0]=x, X_star[:,1]=t
        x_ = to_tensor(X_star[:, 0:1], self.device).requires_grad_(True)
        t_ = to_tensor(X_star[:, 1:2], self.device).requires_grad_(True)

        self.model.eval()
        uv = self.model(x_, t_)
        u = uv[:, 0:1]
        v = uv[:, 1:2]

        # physics residuals (need grads, so not using no_grad for this part)
        # redo with grad:
        x_.requires_grad_(True)
        t_.requires_grad_(True)
        f_u, f_v = self._f_uv(x_, t_)

        return (
            u.detach().cpu().numpy(),
            v.detach().cpu().numpy(),
            f_u.detach().cpu().numpy(),
            f_v.detach().cpu().numpy(),
        )

# ---------- main ----------

if __name__ == "__main__":
    np.random.seed(1234)
    torch.manual_seed(1234)

    # Domain bounds
    lb = np.array([-5.0, 0.0], dtype=np.float32)
    ub = np.array([5.0, np.pi / 2], dtype=np.float32)

    N0 = 50
    N_b = 50
    N_f = 20000
    layers = [2, 100, 100, 100, 100, 2]
    os.makedirs("Data", exist_ok=True)
    url = "https://github.com/maziarraissi/PINNs/raw/master/main/Data/NLS.mat"
    r = requests.get(url)
    with open("Data/NLS.mat", "wb") as f:
        f.write(r.content)

    # Load data (Raissi NLS dataset)
    data = scipy.io.loadmat("Data/NLS.mat")
    t = data["tt"].flatten()[:, None]
    x = data["x"].flatten()[:, None]
    Exact = data["uu"]
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u ** 2 + Exact_v ** 2)

    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact_u.T.flatten()[:, None]
    v_star = Exact_v.T.flatten()[:, None]
    h_star = Exact_h.T.flatten()[:, None]

    # Build training subsets like TF script
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x, :]
    u0 = Exact_u[idx_x, 0:1]
    v0 = Exact_v[idx_x, 0:1]

    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t, :]

    # Collocation points with LHS mapped to [lb,ub]
    H = lhs(2, N_f)  # in [0,1]^2
    X_f = lb + (ub - lb) * H

    pinn = PhysicsInformedNN(x0, u0, v0, tb, X_f, layers, lb, ub)

    t_train_start = time.time()
    # Adam stage (shorter than TF’s 50k if you like; you can do 50k too)
    pinn.train_adam(n_iter=50000, log_every=10)

    print(f"Training time: {time.time() - t_train_start:.2f}s")

    # Predict on full grid
    u_pred, v_pred, f_u_pred, f_v_pred = pinn.predict(X_star)
    h_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)

    # Errors
    err_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    err_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    err_h = np.linalg.norm(h_star - h_pred, 2) / np.linalg.norm(h_star, 2)
    print(f"Error u: {err_u:.3e}")
    print(f"Error v: {err_v:.3e}")
    print(f"Error h: {err_h:.3e}")

    # (Optional) quick plots similar to TF script
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method="cubic")
    V_pred = griddata(X_star, v_pred.flatten(), (X, T), method="cubic")
    H_pred = griddata(X_star, h_pred.flatten(), (X, T), method="cubic")

    # Example: plot |h(t,x)|
    plt.figure()
    im = plt.imshow(
        H_pred.T,
        interpolation="nearest",
        extent=[lb[1], ub[1], lb[0], ub[0]],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("|h(t,x)| prediction")
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()
