import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn

 
class Preconditioner:
    def apply(self, r : torch.Tensor):
        raise NotImplementedError


class IdenitityPreconditioner(Preconditioner):
    def apply(self, r: torch.Tensor):
        return r


class JacobiPreconditioner(Preconditioner):
    def __init__(self, A : torch.Tensor):

       diag = torch.diag(A)
       self.inv_diag = 1 / diag

    def apply(self, r: torch.Tensor):
        return self.inv_diag * r # no need for full matrix mul.


from src.utils.pcg import pcg 
from src.utils.fgmres import fgmres 


def main():
    n = 100
    R = torch.rand((n,n)) 
    A = R.T @ R + 0.01 * torch.eye(n) # to ensure diag is not 0
    b = torch.rand(n)


    jacobi = JacobiPreconditioner(A)
    pcg_jacobi = pcg(A = A, b = b, preconditioner=jacobi, max_iter=10000)
    fgmres_jacobi = fgmres(A = A, b = b, preconditioner=jacobi)


    res = []

    for r in pcg_jacobi['residual_hist']:
        res.append(torch.norm(r))


    plt.figure()
    plt.plot(range(pcg_jacobi['num_iter'] + 1), res)
    plt.show()


    res = []

    for r in fgmres_jacobi['residual_hist']:
        res.append(r)


    plt.figure()
    plt.plot(range(fgmres_jacobi['num_iter'] + 1), res)
    plt.show()





if __name__ == "__main__":
    main()

