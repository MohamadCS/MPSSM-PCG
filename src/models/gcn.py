import enum

from sklearn.metrics import average_precision_score
import torch
from torch import nn, sigmoid
import torch.nn.functional as F

from torch_geometric.nn import GraphNorm
from src.utils.preconditioner import Preconditioner



ATOM_DIMS = (119, 5 , 12 , 12, 10, 6 , 6 , 2 ,2)


class AtomEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.embeddings = nn.ModuleList(
                nn.Embedding(size, hidden_dim)
                for size in ATOM_DIMS
        )

    def forward(self, X):
        H = 0

        for i, embedding in enumerate(self.embeddings):
            H = H + embedding(X[:, i].long())

        return H



class GCN(nn.Module):
    def __init__(self, out_dim : int, hidden_dim : int = 16, num_layers : int = 2, dropout : float = 0.1):
        super().__init__()

        self.encoder = AtomEncoder(hidden_dim)


        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim,hidden_dim,bias = False)
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            GraphNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        self.dropout  = dropout

        self.out = nn.Sequential(
                nn.Linear(hidden_dim,hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(hidden_dim,out_dim)
        )




    def forward(self, X : torch.Tensor, A_norm : torch.Tensor = None, preconditioner : Preconditioner = None):
        if A_norm is None and preconditioner is None:
            raise ValueError(f"You need a preconditioner or a shift operator")

        H = self.encoder(X) 


        for layer,norm in zip(self.layers, self.norms):

            residual = H

            if A_norm is not None:
                H = A_norm @ H
            else:
                H = preconditioner.apply(H)  

            H = layer(H)
            H = norm(H)
            H = F.relu(H)

            H = F.dropout(H, p = self.dropout,training=self.training)

            H = H + residual


        H_graph = H.mean(dim = 0, keepdim=True)

        return self.out(H_graph)




class EchoGCN(nn.Module):
    def __init__(self, in_dim : int,hidden_dim : int = 16, num_layers : int = 2, dropout : float = 0.1):
        super().__init__()

        self.encoder = nn.Linear(in_dim,hidden_dim)


        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim,hidden_dim,bias = False)
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        self.dropout  = dropout

        self.out = nn.Sequential(
                nn.Linear(hidden_dim,hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim,1)
        )




    def forward(self, X : torch.Tensor, A_norm : torch.Tensor = None, preconditioner : Preconditioner = None):
        if A_norm is None and preconditioner is None:
            raise ValueError(f"You need a preconditioner or a shift operator")

        H = self.encoder(X) 


        for layer,norm in zip(self.layers, self.norms):

            residual = H

            if A_norm is not None:
                H = A_norm @ H
            else:
                H = preconditioner.apply(H)  

            H = layer(H)
            H = norm(H)
            H = F.relu(H)

            H = F.dropout(H, p = self.dropout,training=self.training)

            H = H + residual


        return self.out(H)




class EchoEnergyGCN(nn.Module):
    def __init__(self, in_dim : int,hidden_dim : int = 16, num_layers : int = 2, dropout : float = 0.1):
        super().__init__()

        self.encoder = nn.Linear(in_dim,hidden_dim)


        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim,hidden_dim,bias = False)
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        self.dropout  = dropout

        self.out = nn.Sequential(
                nn.Linear(3 * hidden_dim,hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim,1)
        )




    def forward(self, X : torch.Tensor, A_norm : torch.Tensor = None, preconditioner : Preconditioner = None):
        if A_norm is None and preconditioner is None:
            raise ValueError(f"You need a preconditioner or a shift operator")

        H = self.encoder(X) 


        for layer,norm in zip(self.layers, self.norms):

            residual = H

            if A_norm is not None:
                H = A_norm @ H
            else:
                H = preconditioner.apply(H)  

            H = layer(H)
            H = norm(H)
            H = F.relu(H)

            H = F.dropout(H, p = self.dropout,training=self.training)

            H = H + residual

        H_graph = torch.cat(
                [
                    H.sum(dim = 0),
                    H.max(dim = 0).values,
                    H.mean(dim = 0),
                ],
                dim = -1
        ).unsqueeze(0)


        return self.out(H_graph)


