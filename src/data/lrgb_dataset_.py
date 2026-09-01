from typing import Literal

import torch
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_dense_adj




SignalType = Literal["gaussian"]


def normalized_laplacian(data : Data, dtype : torch.dtype = torch.float32) -> torch.Tensor:

    num_nodes = data.num_nodes

    L_edge_index, L_edge_weight = get_laplacian(
            edge_index = data.edge_index,
            edge_weight = None,
            normalization = "sym",
            dtype = dtype,
            num_nodes=num_nodes
    )

    L = to_dense_adj(
            edge_index = L_edge_index,
            edge_attr = L_edge_weight,
            max_num_nodes=num_nodes,
    ).squeeze(dim = 0) # since it returns [1,n,n]

    return L




def sample_gaussian_signal(
        num_nodes : int,
        signal_dim : int,
        generator: torch.Generator,
        dtype : torch.dtype = torch.float32,
        device : torch.device = None
        ) -> torch.Tensor:

    signal = torch.randn(
            num_nodes,
            signal_dim,
            generator=generator,
            dtype = dtype,
            device= device
    )

    column_norms = signal.norm(dim = 0, keepdim=True).clamp_min(1e-12)

    return signal/column_norms



OperatorType = Literal["inverse", "heat", "pseudoinverse"]

# NOTE: Think about a way we can use torch-sla that the prof suggested. 
@torch.no_grad()
def build_operator_target(
        data : Data,
        signal : torch.Tensor,
        operator_type : OperatorType,
        alpha : float = 1.0,
        t : float = 1.0,
        ) -> torch.Tensor:


    device = signal.device

    L_norm = normalized_laplacian(data = data, dtype = signal.dtype).to(device)

    if operator_type == "inverse":
        I = torch.eye(data.num_nodes, dtype = signal.dtype, device = device)
        res_mat = I + alpha * L_norm
        target = torch.linalg.solve(res_mat, signal)
    elif operator_type == "heat":
        target = torch.matrix_exp(-t * L_norm) @ signal

    elif operator_type == "pseudoinverse":
        L_pinv = torch.linalg.pinv(L_norm, hermitian = True)
        target = L_pinv @ signal

    else:
        raise ValueError(f"Unsupported Operator: {operator_type}")

    return target



def prepare_operator_sample(data, operator_type : OperatorType ,signal_dim : int, seed : int = 0,alpha : float = 1.0, t : float = 1.0, device : torch.device | str = "cpu"):


    device = torch.device(device)
    data = data.to(device)

    gen = torch.Generator(device = device)
    gen.manual_seed(seed)

    signal = sample_gaussian_signal(
            num_nodes=data.num_nodes,
            signal_dim =signal_dim,
            generator=gen,
            device = device
    )

    target = build_operator_target(
            data=data, 
            signal=signal,
            operator_type=operator_type,
            alpha = alpha,
            t = t,
    )

    signal = signal.to(device)
    target = target.to(device)

    return data, signal,target




# class TransformedLRGBDataset(Dataset[Data]):
#     def __init__(self,
#                  root : str,
#                  split : str,
#                  operator_type : OperatorType,
#                  name : str = "Peptides-struct",
#                  alpha : float = 1.0,
#                  t : float = 1.0,
#                  signal_dim : int = 8,
#                  signal_type : SignalType = "gaussian",
#                  seed : int = 0,
#                  max_graphs : int = None 
#                  ):
#
#         self.graphs = LRGBDataset(
#                 root = root,
#                 name = name,
#                 split = split
#         )
#
#         self.alpha = alpha
#         self.t = t
#         self.signal_dim = signal_dim
#         self.signal_type = signal_type
#         self.seed = seed
#         self.epoch = 0
#
#         self.num_graphs = min(max_graphs, len(self.graphs))
#
#     def __len__(self) -> int:
#         return self.num_graphs
#
#
#     def set_epoch(self, epoch : int):
#         self.epoch = epoch
#
#     def _make_generator(self, index : int) -> torch.Generator:
#         generator = torch.Generator
#
#         sample_seed = self.seed + index + self.epoch*self.num_graphs
#
#         generator.manual_seed(sample_seed)
#         return generator
#
#     def __getitem__(self, index : int) -> Data:



























