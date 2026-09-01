import torch
from torch import nn


def sc_normalize_adj(A : torch.Tensor) -> torch.Tensor:
    A_abs = A.abs()

    max_row = A_abs.sum(dim =1).max()
    max_col = A_abs.sum(dim = 0).max()

    gamma = torch.minimum(max_row, max_col).clamp(1e-12)

    return A / gamma 

class SGCNLayer(nn.Module):
    def __init__(self, in_dim : int , out_dim : int):
        super().__init__()
        self.local_conv = nn.Linear(in_dim, out_dim, bias=False)
        self.state_mix = nn.Linear(in_dim, out_dim, bias=False)
        self.relu = nn.ReLU()


    def forward(self,X : torch.Tensor , A : torch.Tensor) -> torch.Tensor:
        local = self.local_conv(A @ X) 
        glob = self.state_mix(X) 

        X = self.relu(local + glob)

        return X


class GNP(nn.Module):
    def __init__(self, in_dim : int, gcn_dim : int, out_dim : int,  hidden_dim : int, num_layers : int, A : torch.Tensor):
        super().__init__()

        self.register_buffer("A", sc_normalize_adj(A))

        self.mlp_in = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, gcn_dim),
        )

        self.relu = nn.ReLU()

        self.gcn_blocks = nn.ModuleList([
            SGCNLayer(in_dim=gcn_dim, out_dim=gcn_dim)
            for _ in range(num_layers)
        ])


        self.mlp_out = nn.Sequential(
                nn.Linear(gcn_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, X : torch.Tensor) -> torch.Tensor:
        tau = torch.norm(X).clamp(1e-12)
        s =  (X.shape[0] ** 0.5) / tau

        X = s * X
        X = self.relu(self.mlp_in(X))

        for block in self.gcn_blocks:
            X = block(X, self.A)

        X = self.mlp_out(X)

        X = X / s

        return X
