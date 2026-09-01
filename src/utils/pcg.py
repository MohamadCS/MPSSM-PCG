import numpy as np
import torch
import tqdm as tqdm

def pcg(A : torch.Tensor, b : torch.Tensor, preconditioner, x0 : float = None , tol : float = 1e-8, max_iter = 100):

    x = torch.zeros_like(b) if x0 is None else x0

    r = b - A @ x

    z = preconditioner.apply(r)

    p = z


    residual_hist = [torch.norm(r) / torch.norm(b)]

    for k in range(max_iter):
        q = A @ p


        rz = torch.dot(r,z) # Will be reused later 
        alpha = rz / torch.dot(p,q)

        x = x + alpha * p
        r = r - alpha * q


        rel_residual =  torch.norm(r) / torch.norm(b) 
        residual_hist.append(rel_residual.item())

        if rel_residual < tol:

            return {
                    "x"  : x,
                    "converged" : True,
                    "num_iter" : k + 1,
                    "residual_hist" : residual_hist
            }


        z = preconditioner.apply(r)

        beta = torch.dot(r, z) / rz


        p = z + beta * p

    return {
            "x"  : x,
            "converged" : False,
            "num_iter" : max_iter,
            "residual_hist" : residual_hist
    }


