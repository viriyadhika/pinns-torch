from typing import Dict

import torch
import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl

from lightning.pytorch.loggers import WandbLogger

import pinnstorch


def read_data_fn(root_path):
    """Read and preprocess data from the specified root path.

    :param root_path: The root directory containing the data.
    :return: Processed data will be used in Mesh class.
    """

    data = pinnstorch.utils.load_data(root_path, "NLS.mat")
    exact = data["uu"]
    exact_u = np.real(exact) # N x T
    exact_v = np.imag(exact) # N x T
    exact_h = np.sqrt(exact_u**2 + exact_v**2) # N x T
    return {"u": exact_u, "v": exact_v, "h": exact_h}


time_domain = pinnstorch.data.TimeDomain(t_interval=[0, 1.57079633], t_points = 201)
spatial_domain = pinnstorch.data.Interval(x_interval= [-5, 4.9609375], shape = [256, 1])

mesh = pinnstorch.data.Mesh(root_dir='../data',
                            read_data_fn=read_data_fn,
                            spatial_domain = spatial_domain,
                            time_domain = time_domain)

def read_data_fn(root_path):
    """Read and preprocess data from the specified root path.

    :param root_path: The root directory containing the data.
    :return: Processed data will be used in PointCloud class.
    """

    data = pinnstorch.utils.load_data(root_path, "NLS.mat")

    x = data["x"].T  # N x 1
    t = data["tt"].T  # T x 1
    
    exact = data["uu"]
    exact_u = np.real(exact) # N x T
    exact_v = np.imag(exact) # N x T
    exact_h = np.sqrt(exact_u**2 + exact_v**2) # N x T
    
    return pinnstorch.data.PointCloudData(
            spatial=[x], time=[t], solution={"u": exact_u, "v": exact_v, "h": exact_h}
    )

def initial_fun(x):
    return {'u': 2*1/np.cosh(x), 'v': np.zeros_like(x)}

def train(i: int):
    N0 = 50
    mesh = pinnstorch.data.PointCloud(root_dir='./data',
                                  read_data_fn=read_data_fn)
    
    in_c = pinnstorch.data.InitialCondition(mesh = mesh,
                                        num_sample = N0,
                                        solution = ['u', 'v'])
    
    in_c = pinnstorch.data.InitialCondition(mesh = mesh,
                                        num_sample = N0,
                                        initial_fun = initial_fun,
                                        solution = ['u', 'v'])
    
    N_b = 50
    pe_b = pinnstorch.data.PeriodicBoundaryCondition(mesh = mesh,
                                                 num_sample = 50,
                                                 derivative_order = 1,
                                                 solution = ['u', 'v'])
    
    N_f = 20000
    me_s = pinnstorch.data.MeshSampler(mesh = mesh,
                                   num_sample = N_f,
                                   collection_points = ['f_v', 'f_u'])
    
    val_s = pinnstorch.data.MeshSampler(mesh = mesh,
                                    solution = ['u', 'v', 'h'])
    
    net = pinnstorch.models.FCN(layers = [2, 100, 100, 100, 100, 2],
                            output_names = ['u', 'v'],
                            lb=mesh.lb,
                            ub=mesh.ub)
    
    def output_fn(outputs: Dict[str, torch.Tensor],
              x: torch.Tensor,
              t: torch.Tensor):
        """Define `output_fn` function that will be applied to outputs of net."""

        outputs["h"] = torch.sqrt(outputs["u"] ** 2 + outputs["v"] ** 2)

        return outputs
    

    def pde_fn(outputs: Dict[str, torch.Tensor],
           x: torch.Tensor,
           t: torch.Tensor):   
        """Define the partial differential equations (PDEs)."""
        u_x, u_t = pinnstorch.utils.gradient(outputs["u"], [x, t])
        v_x, v_t = pinnstorch.utils.gradient(outputs["v"], [x, t])

        u_xx = pinnstorch.utils.gradient(u_x, x)[0]
        v_xx = pinnstorch.utils.gradient(v_x, x)[0]

        outputs["f_u"] = u_t + 0.5 * v_xx + (outputs["u"] ** 2 + outputs["v"] ** 2) * outputs["v"]
        outputs["f_v"] = v_t - 0.5 * u_xx - (outputs["u"] ** 2 + outputs["v"] ** 2) * outputs["u"]

        return outputs
    wandb_logger = WandbLogger(
        entity="viriyadhika1",
        project="pinn-lab1",
        name=f"Open source Schrodinger PINN - {i}"
    )
    train_datasets = [me_s, in_c, pe_b]
    val_dataset = val_s
    datamodule = pinnstorch.data.PINNDataModule(train_datasets = [me_s, in_c, pe_b],
                                            val_dataset = val_dataset,
                                            pred_dataset = val_s)
    
    model = pinnstorch.models.PINNModule(net = net,
                                     pde_fn = pde_fn,
                                     output_fn = output_fn,
                                     loss_fn = 'mse')
    

    checkpoint_cb = ModelCheckpoint(
        dirpath=f"checkpoints/{i}/",
        filename="pinn-{epoch:04d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,       # best model
        save_last=True,     # <-- always save last.ckpt
        every_n_epochs=1000
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=60000,
        logger=wandb_logger,
        callbacks=[checkpoint_cb]
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.validate(model=model, datamodule=datamodule)

    return model

if __name__ == '__main__':
    n_ensamble = 5
    for i in range(5):
        o = train(i)