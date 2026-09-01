import torch


def adjacaency_matrix_from_edge_index(edge_index : torch.Tensor, edge_weights : torch.Tensor, num_nodes : int) -> torch.Tensor:

    A = torch.zeros((num_nodes, num_nodes), dtype = edge_weights.dtype, device = edge_weights.dtype)

    edge_index = edge_index.to(edge_weights.device)
    edge_weights = edge_weights.to(edge_weights.device)

    src, dest = edge_index

    A[src, dest] = edge_weights

    return A



def normalize_adj(A : torch.Tensor, self_loops : bool = True) -> torch.Tensor:
    if self_loops:
        A = A + torch.eye(A.shape[0], dtype = A.dtype, device = A.device)

    degrees = A.sum(dim = 1) 
    non_zero_mask = degrees != 0
    degrees[non_zero_mask] = torch.rsqrt(degrees[non_zero_mask]) # better that 1/sqrt(x)
    D_tilde = torch.diag(degrees)

    return D_tilde @ A @ D_tilde


def laplacian(A : torch.Tensor):
    D = torch.diag(A.sum(dim = 1))
    return D - A




def heat_operator(L : torch.Tensor, t : float) -> torch.Tensor:
    return torch.matrix_exp(-t * L)

def heat_target(L : torch.Tensor, t : float, X : torch.Tensor) -> torch.Tensor:
    return heat_operator(L, t) @ X




def psuedo_inverse_operator(L : torch.Tensor) -> torch.Tensor:
    return torch.linalg.pinv(L)

def pseudo_inv_target(L : torch.Tensor, X : torch.Tensor) -> torch.Tensor:
    return psuedo_inverse_operator(L) @ X




def diffusion_system(L : torch.Tensor, alpha : float) -> torch.Tensor:
    I = torch.eye(L.shape[0], dtype = L.dtype, device = L.device)
    return I + alpha * L

def diffusion_operator(L : torch.Tensor, alpha : float) -> torch.Tensor:
    M = diffusion_system(L, alpha)
    return torch.inverse(M)

def diffusion_target(L : torch.Tensor, alpha : float, X : torch.Tensor) -> torch.Tensor:
    M = diffusion_system(L, alpha)
    return torch.linalg.solve(M,X)




