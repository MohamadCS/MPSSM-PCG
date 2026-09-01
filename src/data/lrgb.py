from numpy import append
import torch.nn.functional as F
from torch_geometric.datasets import LRGBDataset
import torch
import time

from tqdm import tqdm

from src.models.neuralpcg import (NeuralPCG, NeuralPCGPreconditioner)
from src.utils import preconditioner
from src.utils.preconditioner import *
from src.utils.pcg import pcg

from src.models.gcn import (GCN, evaluate_gcn, train_gcn)

def load_datasets():
    train_dataset = LRGBDataset(root = "data/LRGB", name = "Peptides-struct", split = "train")
    val_dataset = LRGBDataset(root = "data/LRGB", name = "Peptides-struct", split = "val")
    test_dataset = LRGBDataset(root = "data/LRGB", name = "Peptides-struct", split = "test")

    return train_dataset,val_dataset, test_dataset




def graph_to_sys(data, alpha : float = 1.0, dtype = torch.float32, device = 'cuda'):

    n = data.num_nodes

    edge_index = data.edge_index.to(device)

    src,dst = edge_index

    A = torch.zeros((n,n), dtype = dtype, device = device)

    A[src, dst] = 1.0
    A[dst, src] = 1.0

    A.fill_diagonal_(0)

    deg = A.sum(dim = 1)

    L = torch.diag(deg) - A

    I = torch.eye(n, dtype = dtype, device = device)

    return I + alpha * L 


def neural_pcg_loss(model, K : torch.Tensor, x : torch.Tensor, b : torch.Tensor):
    L = model(K,b)

    y = L.T @ x
    Px = L @ y

    return F.mse_loss(Px, b)



@torch.no_grad()
def validate(model, val_dataset, device, alpha = 1.0, max_graphs = None):
    model.eval()
    total_loss = 0.0
    count = 0


    for i, data in enumerate(val_dataset):

        if max_graphs is not None and i >= max_graphs:
            break

        K = graph_to_sys(data, alpha = alpha, device = device)
        n = K.shape[0]

        gen = torch.Generator(device = device)

        gen.manual_seed(i)

        x = torch.randn(n, dtype = K.dtype, device = K.device, generator = gen)

        b = K @ x

        loss = neural_pcg_loss(model, K , x, b)

        total_loss += loss.item()
        count += 1

    model.train()

    return total_loss  / count


def train_neuralpcg_lrgb(model, train_dataset, val_dataset, device,max_train_graphs = None, max_val_graphs = None, epochs = 100, lr = 1e-3, alpha = 1.0):

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    best_val = float('inf')
    best_model = None


    for epoch in range(epochs):
        model.train()
        train_total_loss = 0
        train_counter = 0

        for i,data in enumerate(train_dataset):
            if max_train_graphs is not None and i >= max_train_graphs:
                break

            K = graph_to_sys(data, alpha = alpha, device = device)
            n = K.shape[0]

            x = torch.randn(n, dtype = K.dtype, device = device)
            b = K @ x

            optimizer.zero_grad()

            loss = neural_pcg_loss(model, K, x , b)

            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()
            train_counter += 1

        train_loss = train_total_loss / train_counter
        val_loss = validate(model, val_dataset,device, alpha, max_graphs = max_val_graphs)

        print( f"Epoch {epoch} | Train {train_loss} | Val {val_loss}" )

        if val_loss < best_val:
            best_val = val_loss
            best_model = {k : v.detach().cpu().clone() for k,v in model.state_dict().items()}



    model.load_state_dict(best_model)
    return model



def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


# we should use that instead of hard coding
def build_methods(model):
    return {
            "Identity" : lambda K,b : IdenitityPreconditioner(),
            "Jacobi" : lambda K,b : IdenitityPreconditioner(K),
            "NeuralPCG" : lambda K,b : NeuralPCGPreconditioner(model,K,b),
    }

def evaluate_method(name, model, K, b , x_true, device, tol = 1e-6, max_iter = 1000):
    sync(device)
    start = time.perf_counter()

    if name == "Jacobi":
        precondtioner = JacobiPreconditioner(K)
    elif name == "NeuralPCG":
        precondtioner = NeuralPCGPreconditioner(model, K , b)
    elif name == "Identity":
        precondtioner = IdenitityPreconditioner()

    sync(device)
    setup_time = time.perf_counter() - start


    sync(device)
    start = time.perf_counter()

    result = pcg(K, b, precondtioner, tol = tol, max_iter = max_iter)

    sync(device)
    solve_time = time.perf_counter() - start

    x_pred = result['x']
    relative_error = (torch.linalg.norm(x_pred - x_true) / torch.linalg.norm(x_true)).item()  

    relative_residual = (torch.linalg.norm(b - K @ x_pred) / torch.linalg.norm(b)).item()

    return {
            "method" : name,
            "converged" : result['converged'],
            "iterations" : result['num_iter'],
            "relative_residual" :relative_residual,
            "relative_error" :relative_error,
            "setup_time" :setup_time,
            "solve_time" :solve_time,
            "total_time" :solve_time + setup_time,
    }

@torch.no_grad()
def benchmark(model, test_dataset, device, alpha = 1.0, tol = 1e-6, max_iter = 1000, num_rhs = 1, max_graphs = None):
    methods = ["Jacobi", "NeuralPCG", "Identity"]

    results = {method : [] for method in methods}

    model.eval()

    for graph_idx, data in enumerate(test_dataset):
        if max_graphs is not None and graph_idx >= max_graphs:
            break

        K= graph_to_sys(data, alpha = alpha, device = device)

        n = K.shape[0]


        for rhs_idx in range(num_rhs):
            generator = torch.Generator(device = device)
            generator.manual_seed(graph_idx * 100 + rhs_idx)
            x_true = torch.randn(n, dtype = K.dtype, device = device, generator = generator)
            b = K @ x_true


            for name in methods:
                result = evaluate_method(name = name, model = model, K = K, b = b , x_true=x_true, device=device, tol = tol, max_iter = max_iter)
                results[name].append(result)

    return results



def make_neuralpcg_factory(model, device, alpha = 1.0):
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    def factory(data, graph_idx):
        K = graph_to_sys(data, alpha = alpha, device = device)

        n = K.shape[0]

        generator = torch.Generator(device = device)
        generator.manual_seed(graph_idx)

        x = torch.randn(n, dtype = K.dtype, device = device, generator=generator)

        b = K @ x

        return NeuralPCGPreconditioner(model, K, b)

    return factory

    

## I generated this, since I'm bad at printing pretty results. 
def summarize_results(results):

    print(
        f"{'Method':<15}"
        f"{'Conv%':>10}"
        f"{'Iter':>10}"
        f"{'Residual':>14}"
        f"{'RelError':>14}"
        f"{'Setup(ms)':>14}"
        f"{'Solve(ms)':>14}"
        f"{'Total(ms)':>14}"
    )

    print("-" * 105)

    for name, runs in results.items():

        n = len(runs)

        conv = sum(
            r["converged"]
            for r in runs
        ) / n

        iterations = sum(
            r["iterations"]
            for r in runs
        ) / n

        residual = sum(
            r["relative_residual"]
            for r in runs
        ) / n

        error = sum(
            r["relative_error"]
            for r in runs
        ) / n

        setup = sum(
            r["setup_time"]
            for r in runs
        ) / n

        solve = sum(
            r["solve_time"]
            for r in runs
        ) / n

        total = sum(
            r["total_time"]
            for r in runs
        ) / n

        print(
            f"{name:<15}"
            f"{100 * conv:>9.1f}%"
            f"{iterations:>10.2f}"
            f"{residual:>14.3e}"
            f"{error:>14.3e}"
            f"{1000 * setup:>14.3f}"
            f"{1000 * solve:>14.3f}"
            f"{1000 * total:>14.3f}"
        )












def main():
    device = "cuda"
    torch.manual_seed(0)


    train_dataset, val_dataset, test_dataset = load_datasets()






    model = NeuralPCG().to(device)
    model = train_neuralpcg_lrgb(model,train_dataset,val_dataset,device, epochs = 100, max_train_graphs = None, max_val_graphs=None)

    results = benchmark(model = model, test_dataset=test_dataset, device = device, alpha = 1.0, tol = 1e-6, max_iter = 1000, max_graphs= 100, num_rhs = 3)

    summarize_results(results)

    pgcn = GCN(hidden_dim=128, num_layers= 2).to(device)

    pgcn,hist = train_gcn (
            model = pgcn,
            train_dataset = train_dataset,
            val_dataset= val_dataset,
            device= device,
            epochs= 100,
            # class_weights = class_weights,
            lr = 1e-3,
            max_train_graphs=1000,
            max_val_graphs=1000,
            preconditioner_factory = make_neuralpcg_factory(model,device)
    )



if __name__ == "__main__":
    main()



