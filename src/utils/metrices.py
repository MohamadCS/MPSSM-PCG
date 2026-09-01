from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_undirected



def dirichlet_energy(L : torch.Tensor , X : torch.Tensor) -> torch.Tensor: 
    return torch.trace(X.T @ L @ X)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

