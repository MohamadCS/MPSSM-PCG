import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from tqdm import tqdm

from src.utils.preconditioner import Preconditioner

def matrix_to_graph(K : torch.Tensor, b : torch.Tensor) -> torch.Tensor :
    src, dst = torch.nonzero(K, as_tuple = True)

    edge_index = torch.stack([src,dst], dim = 0)
    edge_attr = K[src,dst].unsqueeze(dim = -1)
    x = b.unsqueeze(dim = -1)

    return x, edge_index, edge_attr 




def make_mlp_block( in_dim : int, out_dim : int,hidden_dim : int = 16 ,num_layers : int  = 2):
    layers = []

    if num_layers == 1:
        return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
        )

    layers.append(nn.Linear(in_dim, hidden_dim))
    layers.append(nn.ReLU())

    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())


    layers.append(nn.Linear(hidden_dim, out_dim))
    layers.append(nn.ReLU())

    return nn.Sequential(*layers)



class Encoder(nn.Module):
    def __init__(self, hidden_dim : int, num_layers : int):
        super().__init__()

        self.node_encoder = make_mlp_block(in_dim = 1, out_dim= hidden_dim, hidden_dim = hidden_dim, num_layers = num_layers)
        self.edge_encoder = make_mlp_block(in_dim = 1, out_dim= hidden_dim, hidden_dim = hidden_dim, num_layers = num_layers)

    def forward(self, node_features, edge_features):
        return self.node_encoder(node_features), self.edge_encoder(edge_features)


class MPNNBlock(MessagePassing):
    def __init__(self, hidden_dim : int = 16, num_layers : int = 2):
        super().__init__(aggr = "add")

        self.node_mlp = make_mlp_block(in_dim = 2* hidden_dim,
                                       out_dim=hidden_dim,
                                       hidden_dim = hidden_dim,
                                       num_layers= num_layers,
        ) # since we pass [node_features, message] 


        self.edge_mlp = make_mlp_block(in_dim = 3* hidden_dim,
                                       out_dim=hidden_dim,
                                       hidden_dim = hidden_dim,
                                       num_layers= num_layers,
        ) # since we pass [edge_features ij, node_features i, node_features j] 



    # When x are the node features
    def forward(self, x : torch.Tensor, edge_index : torch.Tensor, edge_attr : torch.Tensor):

        messages = self.propagate(edge_index, x = x, edge_attr = edge_attr)
        x = self.node_mlp(torch.cat([x, messages], dim = -1))

        src, dst = edge_index
        edge_attr = self.edge_mlp(torch.cat([edge_attr, x[src], x[dst]], dim = -1))

        return x, edge_attr

    def message(self, x_j, edge_attr):
        return x_j * edge_attr



class EdgeDecoder(nn.Module):

    def __init__(self, hidden_dim : int = 16):
        super().__init__()

        self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_attr : torch.Tensor):
        return self.decoder(edge_attr).squeeze(dim = -1)



class NeuralPCG(nn.Module):
    def __init__(self, hidden_dim : int = 16, num_mp_layers : int  = 5, num_mlp_layers : int = 2):
        super().__init__()

        self.encoder = Encoder(hidden_dim = hidden_dim, num_layers= num_mlp_layers)

        self.mp_blocks = nn.ModuleList([
            MPNNBlock(hidden_dim = hidden_dim, num_layers = num_mlp_layers)
            for _ in range(num_mp_layers)
        ])

        self.decoder = EdgeDecoder(hidden_dim = hidden_dim)



    def forward(self, K : torch.Tensor, b : torch.Tensor):
        n = K.shape[0]

        x, edge_index, edge_attr = matrix_to_graph(K , b)
        x, edge_attr = self.encoder(x, edge_attr)


        for block in self.mp_blocks:
            x, edge_attr = block(x, edge_index, edge_attr)

        edge_vals = self.decoder(edge_attr)

        src, dst = edge_index

        M = torch.zeros((n,n), dtype = K.dtype, device = K.device)

        M[src, dst] = edge_vals

        M = 0.5 * (M + M.T)

        L = torch.tril(M, diagonal = -1)
        L = L + torch.diag(torch.sqrt(torch.diag(K)))

        return L

    def loss(self, K : torch.Tensor ,x : torch.Tensor):
        b = K @ x
        L = self.forward(K,b)

        y = L.T @ x
        Px = L @ y

        return F.mse_loss(Px, b)



class NeuralPCGPreconditioner(Preconditioner):
    def __init__(self, model, K : torch.Tensor, b : torch.Tensor):
        self.model = model

        with torch.no_grad():
            self.L = model(K,b)


    def apply(self, r : torch.Tensor):

        is_vec = r.dim() == 1

        if is_vec:
            r = r.unsqueeze(-1)


        y = torch.linalg.solve_triangular(self.L , r, upper = False)
        y = torch.linalg.solve_triangular(self.L.T , y, upper = True)

        if is_vec:
            y = y.squeeze(-1)


        return y

        # y = torch.linalg.solv(self.L, r.unsqueeze(-1))




def main():
    pass





if __name__ == "__main__":
    main()











