



import torch
from torch import nn
from torch_geometric.data import Data

from external.neuralif.neuralif.models import NeuralIF

from src.utils.preconditioner import Preconditioner


def K_to_graph(K):
    row, col = torch.nonzero(
        K != 0,
        as_tuple=True,
    )

    edge_index = torch.stack(
        [row, col],
        dim=0,
    )

    edge_attr = K[
        row,
        col,
    ].unsqueeze(-1)

    # NeuralIF with augment_nodes=True replaces this
    # with its matrix/graph structural features.
    x = torch.zeros(
        (K.shape[0], 1),
        dtype=K.dtype,
        device=K.device,
    )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


class NeuralIFAdapter(nn.Module):

    def __init__(self, num_probes : int = 4):
        super().__init__()

        self.num_probes = num_probes

        self.model = NeuralIF(
            global_features=0,
            latent_size=8,

            augment_nodes=True,

            message_passing_steps=4,

            skip_connections=True,
            activation="relu",
            aggregate=None,
            decode_nodes=False,

            normalize_diag=False,
            graph_norm=True,
            two_hop=False,
            edge_features=1,
        )


    def forward(self, K):
        data = K_to_graph(K)

        L, _, _ = self.model(data)

        return L


    def loss(self, K : torch.Tensor ,X : torch.Tensor):
        L = self.forward(K)

        target = K @ X

        prediction = L @ (L.T @ X)

        numerator = (prediction - target).square().mean()
        denominator = target.square().mean().clamp_min(1e-12)



        return numerator / denominator 


class NeuralIFPreconditioner(Preconditioner):

    def __init__(self, model, K):
        super().__init__()

        with torch.no_grad():
            L_sparse = model(K)

            # Same dense triangular-solve implementation
            # that we use for SSMPCG.
            self.L = L_sparse.to_dense()


    def apply(self, r):
        is_vec = r.dim() == 1

        if is_vec:
            r = r.unsqueeze(-1)

        y = torch.linalg.solve_triangular(
            self.L,
            r,
            upper=False,
        )

        z = torch.linalg.solve_triangular(
            self.L.T,
            y,
            upper=True,
        )

        if is_vec:
            z = z.squeeze(-1)

        return z
