import torch 
import numpy as np





def fgmres(A : torch.Tensor, b : torch.Tensor, preconditioner ,x0 : float = None, tol : float = 1e-8, max_iter : int = 10000):
    n = b.shape[0]

    max_iter = min(max_iter, n)
    x0 = torch.zeros_like(b) if x0 is None else x0

    r = b - A @ x0 

    V = torch.zeros((n, max_iter + 1), dtype = A.dtype, device = A.device)
    Z = torch.zeros((n, max_iter), dtype = A.dtype, device = A.device)
    H = torch.zeros((max_iter + 1, max_iter), dtype = A.dtype, device = A.device)

    b_norm = torch.norm(b)
    beta = torch.norm(r)
    rel_residual = beta / b_norm

    residual_hist = [rel_residual.item()]


    if rel_residual < tol: 
        return {
                "x" : x0,
                "converged" : True,
                "num_iter" : 0, 
                "residual_hist" : residual_hist
        }


    V[:, 0] = r / beta

    g = torch.zeros(
            max_iter + 1,
            dtype = A.dtype,
            device = A.device,
    )

    g[0] = beta


    for k in range(max_iter):
        z = preconditioner.apply(V[:, k]) 
        Z[:, k] = z

        w = A @ z


        for j in range(k + 1):
            H[j,k] = torch.dot(V[:, j], w)
            w = w - H[j,k] * V[:,j]

        H[k + 1,k] = torch.norm(w)

        if H[k + 1, k] > 1e-15: # avoid dividing by 0
            V[:, k + 1] = w / H[k + 1,k]

        Hk = H[:k + 2, : k + 1]
        gk = g[:k + 2]

        y = torch.linalg.lstsq(Hk,gk).solution

        x = x0 + Z[:, :k +1] @ y

        r = b - A@x

        rel_residual = torch.norm(r) / b_norm

        residual_hist.append(rel_residual.item())


        if rel_residual < tol:
            return {
                    "x" : x,
                    "converged" : True,
                    "num_iter" : k + 1, 
                    "residual_hist" : residual_hist
            }

        if H[k + 1, k] <= 1e-15:
            return {
                    "x" : x,
                    "converged" : False,
                    "num_iter" : k + 1, 
                    "residual_hist" : residual_hist
            }

    return {
            "x" : x,
            "converged" : False,
            "num_iter" : max_iter, 
            "residual_hist" : residual_hist
    }













