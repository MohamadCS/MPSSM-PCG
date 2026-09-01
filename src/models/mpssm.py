import torch
from torch import nn

import torch.nn.functional as F


from src.utils.preconditioner import Preconditioner

from torch_geometric.nn import GraphNorm




def graph_shift(K):

    A = (K != 0).to(K.dtype)

    A.fill_diagonal_(1.0)

    deg = A.sum(dim = 1)

    D_inv_sq = torch.rsqrt(deg.clamp_min(1e-12))

    S = D_inv_sq[:, None] * A * D_inv_sq[None, :]

    return S

# def weighted_graph_shift(K):
#     n = K.shape[0]
#
#     A = torch.zeros_like(K)
#
#     off_diag = ~torch.eye(n, dtype = torch.bool, device = K.device)
#
#     A[off_diag] = (-K[off_diag]).clamp_min(0.0)
#
#     A.fill_diagonal_(1.0)
#
#     deg = A.sum(dim = 1)
#
#     d_inv_sqrt = torch.rsqrt(deg.clamp_min(1e-12))
#
#     return d_inv_sqrt[:,None] * A * d_inv_sqrt[None, :]




# def graph_shift(K):
#     n = K.shape[0]
#
#     A = torch.zeros_like(K)
#
#     off_diag = ~torch.eye(n, dtype = torch.bool, device = K.device)
#
#     A[off_diag] = (-K[off_diag]).clamp_min(0.0)
#
#     A.fill_diagonal_(1.0)
#
#     deg = A.sum(dim = 1)
#
#     D_inv_sq = torch.rsqrt(deg.clamp_min(1e-12))
#
#     S = D_inv_sq[:, None] * A * D_inv_sq[None, :]
#
#     return S
#


def lower_edge_pattern(K):
    mask = torch.tril(K != 0, diagonal = -1)

    src, dst = torch.nonzero(mask, as_tuple = True)

    return src, dst



class MPSSMBlock(nn.Module):

    def __init__(self, in_dim : int ,hidden_dim : int, num_steps : int):
        super().__init__()

        self.num_steps = num_steps

        self.B = nn.Linear(in_dim, hidden_dim, bias = False)

        # self.W_supp = nn.Linear(hidden_dim,hidden_dim, bias = False)
        # self.W_val = nn.Linear(hidden_dim,hidden_dim, bias = False)

        self.W = nn.Linear(hidden_dim,hidden_dim, bias = False)

        self.mlp = nn.Sequential(
                nn.Linear(hidden_dim,hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim,hidden_dim),
        )

    def forward(self, U,S):
        # X = torch.zeros(U.shape[0], self.W_supp.out_features, dtype= U.dtype, device = U.device)
        X = torch.zeros(U.shape[0], self.W.out_features, dtype= U.dtype, device = U.device)

        UB = self.B(U)


        for _ in range(self.num_steps):
            # X = self.W_supp(S_supp @ X) + self.W_val(S_val @ X) + UB
             X =  self.W(S @ X) + UB

        return self.mlp(X)



class MPSSM(nn.Module):
    def __init__(self, in_dim : int, hidden_dim : int, num_steps : int, num_blocks : int, dropout : float = 0.0):
        super().__init__()

        self.num_blocks = num_blocks
        self.num_steps = num_steps
        self.dropout = dropout


        self.in_block = MPSSMBlock(in_dim = in_dim, hidden_dim= hidden_dim, num_steps = num_steps)
        self.blocks = nn.ModuleList(
                [
                    MPSSMBlock(in_dim = hidden_dim, hidden_dim = hidden_dim, num_steps = num_steps)
                    for _ in range(num_blocks - 1)
                ]
        )

        self.norms = nn.ModuleList(
                [
                    GraphNorm(hidden_dim)
                    for _ in range(num_blocks)
                ]
        )

    def forward(self, S : torch.Tensor, U : torch.Tensor):
        H = self.norms[0](self.in_block(U,S))

        

        for i, block in enumerate(self.blocks):
            H_t = block(H,S) 
            H_t = F.dropout(H_t, p = self.dropout, training=self.training)
            H = self.norms[i + 1](H + H_t)

        return H






def matrix_node_features(K : torch.Tensor):
    eps = 1e-12
    n = K.shape[0]

    diag = torch.diag(K)
    scale = diag.abs().mean().clamp_min(eps)

    off_diag = K.clone()
    off_diag.fill_diagonal_(0.0)

    row_strength = off_diag.abs().sum(dim = 1)

    deg = (off_diag.abs() > eps).sum(dim = 1).to(K.dtype)
    deg_scale = deg.mean().clamp_min(1.0)

    if n > 1:
        pos = torch.linspace(-1.0, 1.0, n, dtype = K.dtype, device = K.device)
    else:
        pos = torch.zeros(1, dtype = K.dtype, device = K.device)

    features = torch.stack(
            [
                diag / scale,
                row_strength / scale,
                deg / deg_scale,
                pos
            ],
            dim = -1
    )

    return features, scale





class MPSSMPCG(nn.Module):
    def __init__(self,hidden_dim : int = 16, ssm_steps : int = 4, num_blocks : int = 2, dropout : float = 0.0, num_probes : int = 4) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.ssm_steps = ssm_steps
        self.num_blocks = num_blocks
        self.num_probes = num_probes

        self.mpssm = MPSSM(
                in_dim = 4,
                hidden_dim = hidden_dim,
                num_steps = ssm_steps,
                num_blocks = num_blocks,
                dropout = dropout
            )

        self.factor_decoder = nn.Sequential(
                nn.Linear(2 * hidden_dim + 1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
        )

        self.diag_decoder = nn.Linear(hidden_dim, 1)
        self.init_params()




    def init_params(self):
        nn.init.zeros_(self.diag_decoder.weight)
        nn.init.zeros_(self.diag_decoder.bias)



    def forward(self, K : torch.Tensor):

        c = 0.1

        S= graph_shift(K)
        # S_supp = graph_shift(K)
        # S_val = weighted_graph_shift(K)

        features, scale = matrix_node_features(K)

        H = self.mpssm(S, features)

        src, dst = lower_edge_pattern(K)
        L = torch.zeros_like(K)

        if src.numel() > 0:
            edge_val = (K[src, dst] / scale).unsqueeze(-1)

            edge_features = torch.cat(
                    [
                        H[src],
                        H[dst],
                        edge_val
                    ],
                    dim = -1
            )

            values = self.factor_decoder(edge_features).squeeze(-1)

            L[src,dst] = values

        diag_delta = self.diag_decoder(H).squeeze(-1)
        L_diag =  torch.exp(c * diag_delta)

        L = L + torch.diag(L_diag)


        return L

    def loss(self, K : torch.Tensor ,X : torch.Tensor):
        L = self.forward(K)

        target = K @ X

        prediction = L @ (L.T @ X)

        numerator = (prediction - target).square().mean()
        denominator = target.square().mean().clamp_min(1e-12)



        return numerator / denominator 






class SSMPCGPreconditioner(Preconditioner):
    def __init__(self, model : nn.Module ,K : torch.Tensor):
        super().__init__()

        self.model = model

        with torch.no_grad():
            self.L = model(K)

    def apply(self, r):
        is_vec = r.dim() == 1

        if is_vec:
            r = r.unsqueeze(-1)


        y = torch.linalg.solve_triangular(self.L , r, upper = False)
        y = torch.linalg.solve_triangular(self.L.T , y, upper = True)

        if is_vec:
            y = y.squeeze(-1)


        return y





#
# ### OLD MODEL
# class SSMPCG(nn.Module):
#
#     def __init__(self,hidden_dim = 32, ssm_steps = 5, num_probes = 1) -> None:
#         super().__init__()
#
#         self.hidden_dim = hidden_dim
#         self.ssm_steps = ssm_steps
#         self.num_probes = num_probes
#
#         self.diag_decoder = nn.Linear(hidden_dim, 1)
#
#         nn.init.zeros_(self.diag_decoder.bias)   
#         nn.init.zeros_(self.diag_decoder.weight)   
#
#
#         self.input_proj = nn.Linear(1, hidden_dim)
#
#         self.W = nn.Linear(hidden_dim,hidden_dim, bias = False)
#         self.B = nn.Linear(hidden_dim,hidden_dim, bias = False)
#
#
#         self.readout = nn.Sequential(
#                 nn.Linear(hidden_dim,hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim,hidden_dim)
#         )
#
#         self.factor_decoder = nn.Sequential( # Z_i,Z_j,K_ij
#                 nn.Linear(2 * hidden_dim + 1, hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, 1),
#         )
#
#
#     def forward(self, K):
#
#         S = graph_shift(K)
#
#         diag = torch.diag(K).clamp_min(1e-12)
#         scale = diag.mean().clamp_min(1e-12)
#
#
#         # row_strength = K.abs().mean(dim = -1)
#
#
#         features = torch.stack(
#                 [
#                     diag / scale,
#                     # row_strength / scale, 
#                 ],
#                 dim = -1
#         ) # (N,2)
#
#         U = self.input_proj(features)
#         UB = self.B(U)
#
#
#         H = torch.zeros_like(U)
#         H_hist = [] 
#
#         for _ in range(self.ssm_steps):
#             H = self.W(S @ H) + UB
#             H_hist.append(H)
#
#         # H = torch.stack(H_hist, dim = 0).mean(dim = 0) # mean(T * N * d, dim = 0) => N * d
#
#         H = self.readout(H) 
#
#         src, dst = lower_edge_pattern(K)
#
#         L = torch.zeros_like(K)
#
#         if src.numel() > 0:
#             edge_features = (K[src, dst] / scale).unsqueeze(dim = -1)
#             pair = torch.cat([H[src], H[dst], edge_features], dim = -1)
#
#             values = self.factor_decoder(pair).squeeze(dim = -1) 
#             values = torch.sqrt(scale) * torch.tanh(values)
#
#             L[src, dst] = values
#
#
#         # L = L + torch.diag(torch.sqrt(diag))
#
#         diag_delta = self.diag_decoder(H).squeeze(-1)
#         L_diag = torch.sqrt(diag) * torch.exp(0.1 * diag_delta)
#         L = L + torch.diag(L_diag)
#
#         return L
#
#     def loss(self, K : torch.Tensor ,X : torch.Tensor):
#         B = K @ X
#         L = self.forward(K)
#
#         PX = L @ (L.T @ X)
#
#         return F.mse_loss(PX, B) 
#
#
#
#

