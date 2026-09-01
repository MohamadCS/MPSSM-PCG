import argparse
from typing import Callable, Optional

import numpy as np

from pathlib import Path
from torch import nn
import torch
import time

import json
from datetime import datetime


from dataclasses import dataclass

RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")


from src.models.mpssm import (MPSSMPCG, SSMPCGPreconditioner)
from src.models.neuralif_adapter import (NeuralIFAdapter, NeuralIFPreconditioner)
from src.utils.preconditioner import *
from src.utils.pcg import pcg
from src.utils.metrices import count_params



from torch_geometric.datasets import LRGBDataset


from torch_geometric.data import InMemoryDataset


PEPTIDES_STRUCT = "Peptides-struct"
PEPTIDES_FUNC = "Peptides-func"
ECHO_SSSP = "ECHO-SSSP"
ECHO_CHARGE= "ECHO-Charge"
ECHO_ENERGY= "ECHO-ENERGY"

def print_saved_results(path):
    path = Path(path)

    with open(path, "r") as f:
        records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    aggregate = next(
        (
            record
            for record in reversed(records)
            if record["type"] == "aggregate"
        ),
        None,
    )

    if aggregate is None:
        print("No aggregate result found.")
        return

    results = aggregate["results"]

    print("\n" + "=" * 150)
    print(
        f"Dataset: {aggregate['dataset']} | "
        f"alpha={aggregate['alpha']} | "
        f"seeds={aggregate['seeds']}"
    )
    print("=" * 150)

    print(
        f"{'Method':<15}"
        f"{'Conv%':>18}"
        f"{'Iter':>18}"
        f"{'Residual':>24}"
        f"{'RelError':>24}"
        f"{'Setup/RHS(ms)':>24}"
        f"{'Solve(ms)':>24}"
        f"{'Total/RHS(ms)':>24}"
    )

    print("-" * 165)

    for method, stats in results.items():
        conv = stats["converged"]
        iters = stats["iterations"]
        residual = stats["relative_residual"]
        error = stats["relative_error"]
        setup = stats["setup_time"]
        solve = stats["solve_time"]
        total = stats["total_time"]

        print(
            f"{method:<15}"
            f"{100 * conv['mean']:>8.1f}±{100 * conv['std']:<7.1f}"
            f"{iters['mean']:>8.2f}±{iters['std']:<7.2f}"
            f"{residual['mean']:>12.3e}±{residual['std']:<11.3e}"
            f"{error['mean']:>12.3e}±{error['std']:<11.3e}"
            f"{1000 * setup['mean']:>10.3f}±{1000 * setup['std']:<9.3f}"
            f"{1000 * solve['mean']:>10.3f}±{1000 * solve['std']:<9.3f}"
            f"{1000 * total['mean']:>10.3f}±{1000 * total['std']:<9.3f}"
        )

class EchoSplit(InMemoryDataset):
    def __init__(self, path):
        super().__init__(root = None)

        payload = torch.load(path, map_location="cpu", weights_only=False)

        self._data = payload[0]
        self.slices = payload[1]
        self._data_list = None


def load_echo_dataset(task):
    task_to_dir = {
            ECHO_SSSP : "sssp",
            ECHO_CHARGE : "charge",
            ECHO_ENERGY: "energy",
    }

    directory = task_to_dir[task]



    base_path = f"data/ECHO/{directory}/processed/"
    train_dataset = EchoSplit(base_path + f"train_{directory}.pt")
    val_dataset = EchoSplit(base_path + f"val_{directory}.pt")
    test_dataset = EchoSplit(base_path + f"test_{directory}.pt")

    return train_dataset,val_dataset, test_dataset


def load_lrgb_datasets(name):
    root = "data/LRGB"

    train_dataset = LRGBDataset(
        root = root,
        name= name,
        split="train",
    )
    val_dataset = LRGBDataset(
        root=root,
        name=name,
        split="val",
    )

    test_dataset = LRGBDataset(
        root=root,
        name=name,
        split="test",
    )

    return train_dataset, val_dataset, test_dataset


def aggregate_seed_results(seed_results):
    final = {}

    keys = [
        "converged",
        "iterations",
        "relative_residual",
        "relative_error",
        "setup_time",
        "solve_time",
        "total_time",
    ]

    for method in seed_results[0].keys():
        final[method] = {}

        for key in keys:
            values = torch.tensor(
                [result[method][key] for result in seed_results],
                dtype=torch.float64,
            )

            final[method][key] = {
                "mean": values.mean().item(),
                "std": values.std(unbiased=True).item(),
            }

    return final


def save_result_record(record, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

@dataclass 
class Method:
    name : str
    model : Optional[nn.Module]
    build_preconditioner : Callable
    trainable : bool
    rhs_dependant : bool


def build_methods(device):

    num_probes = 10

    # ssmpcg = SSMPCG().to(device)
    neuralif = NeuralIFAdapter(num_probes=num_probes).to(device)
    mpssmpcg = MPSSMPCG(num_probes=num_probes).to(device)
    # neuralpcg = NeuralPCG().to(device)

    methods = [
            Method(
                name = "MPSSMPCG",
                model = mpssmpcg,
                build_preconditioner= lambda model , K, b : SSMPCGPreconditioner(model, K),
                trainable = True,
                rhs_dependant=False,
                ),
            # Method(
            #     name = "SSMPCG",
            #     model = ssmpcg,
            #     build_preconditioner= lambda model , K, b : SSMPCGPreconditioner(model, K),
            #     trainable = True,
            #     rhs_dependant=False,
            #     ),
            Method(
                name = "NeuralIF",
                model = neuralif,
                build_preconditioner= lambda model , K, b : NeuralIFPreconditioner(model, K),
                trainable = True,
                rhs_dependant=False,
            ),
            Method(
                name = "Identity",
                model = None,
                build_preconditioner= lambda model , K, b : IdenitityPreconditioner(),
                trainable = False,
                rhs_dependant=False,
            ),

            Method(
                name = "Jacobi",
                model = None,
                build_preconditioner= lambda model , K, b : JacobiPreconditioner(K),
                trainable = False,
                rhs_dependant=False,
            ),
            # Method(
            #     name = "NeuralPCG",
            #     model = neuralpcg,
            #     build_preconditioner= lambda model , K, b : NeuralPCGPreconditioner(model, K, b),
            #     trainable = True,
            #     rhs_dependant=True,
            # ),
    ]

    for method in methods:
        if method.trainable:
            print(f"[{method.name}] Parameter count {count_params(method.model)}")


    return methods




def graph_to_sys(data, alpha : float = 1.0, dtype = torch.float32, device = 'cuda'):

    n = data.num_nodes

    edge_index = data.edge_index.to(device)

    src,dst = edge_index

    A = torch.zeros((n,n), dtype = dtype, device = device)

    A[src, dst] = 1.0
    A[dst, src] = 1.0

    A.fill_diagonal_(1.0)

    deg = A.sum(dim = 1)

    d_inv_sqrt = torch.rsqrt(deg.clamp_min(1.0))

    A_norm = d_inv_sqrt[:,None] * A  * d_inv_sqrt[None, :] # faster than matrix multiplication

    I = torch.eye(n, dtype = dtype, device = device)

    L = I - A_norm


    return I + alpha * L 





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

        num_probes = getattr(model, "num_probes", 1)

        if num_probes == 1:
            x = torch.randn(n, dtype = K.dtype, device = device, generator = gen)
        else:
            x = torch.randn(n, num_probes, dtype = K.dtype, device = device, generator = gen)


        loss = model.loss(K, x)

        total_loss += loss.item()
        count += 1

    model.train()

    return total_loss  / count


def train(model, name, train_dataset, val_dataset, device, seed,max_train_graphs = None, max_val_graphs = None, epochs = 100, lr = 1e-3, alpha = 1.0):

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

            num_probes = getattr(model, "num_probes", 1)

            if num_probes == 1:
                x = torch.randn(n, dtype = K.dtype, device = device)
            else:
                x = torch.randn(n, num_probes, dtype = K.dtype, device = device)

            optimizer.zero_grad()

            loss = model.loss(K, x)

            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()
            train_counter += 1

        train_loss = train_total_loss / train_counter
        val_loss = validate(model, val_dataset,device, alpha, max_graphs = max_val_graphs)

        print( f" [{name}]  Epoch {epoch} | Train {train_loss} | Val {val_loss}" )

        if val_loss < best_val:
            best_val = val_loss
            best_model = {k : v.detach().cpu().clone() for k,v in model.state_dict().items()}


    model.load_state_dict(best_model)
    return model


def train_all(methods, train_dataset, val_dataset, device, seed,max_train_graphs = None, max_val_graphs = None, epochs = 100, lr = 1e-3, alpha = 1.0):

    for method in methods:
        if not method.trainable:
            continue

        method.model = train(
                model = method.model,
                name = method.name,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                device=device,
                seed=seed,
                epochs=epochs,
                lr = lr,
                alpha = alpha,
                max_train_graphs=max_train_graphs,
                max_val_graphs=max_val_graphs
        )



def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def contruct_preconditioner(method, K, b , device):
    sync(device)
    start = time.perf_counter()

    preconditioner = method.build_preconditioner(method.model, K,b)
    sync(device)
    setup_time = time.perf_counter() - start

    return preconditioner, setup_time

def evalutate_solve(preconditioner, K, b, x_true, device, setup_time, tol = 1e-6, max_iter = 1000): 
    sync(device)
    start = time.perf_counter()

    result = pcg(K,b,preconditioner, tol = tol, max_iter= max_iter)

    sync(device)
    solve_time = time.perf_counter() - start 

    x_pred = result['x']
    relative_error = (torch.linalg.norm(x_pred - x_true) / torch.linalg.norm(x_true)).item()  

    relative_residual = (torch.linalg.norm(b - K @ x_pred) / torch.linalg.norm(b)).item()

    return {
            "converged" : result['converged'],
            "iterations" : result['num_iter'],
            "relative_residual" :relative_residual,
            "relative_error" :relative_error,
            "setup_time" :setup_time,
            "solve_time" :solve_time,
            "total_time" :solve_time + setup_time,
    }




@torch.no_grad()
def benchmark(methods, test_dataset, device, seed, alpha = 1.0, tol = 1e-6, max_iter = 1000, num_rhs = 1, max_graphs = None):
    results = {method.name : [] for method in methods}



    for method in methods:
        if method.model is not None:
            method.model.eval()


    for graph_idx, data in enumerate(test_dataset):
        if max_graphs is not None and graph_idx >= max_graphs:
            break


        K = graph_to_sys(data, alpha = alpha ,device = device)

        n = K.shape[0]

        reusable = {}

        for method in methods:
            if method.rhs_dependant:
                continue

            preconditioner, setup_time = contruct_preconditioner(
                    method = method,
                    K = K,
                    b = None,
                    device = device
            )

            reusable[method.name] = (preconditioner, setup_time / num_rhs)



        for rhs_idx in range(num_rhs):

            gen = torch.Generator(device = device)
            gen.manual_seed(seed * 1000000 + graph_idx * 100 + rhs_idx)
            x_true = torch.randn(n, dtype = K.dtype, device = device, generator = gen)

            b = K @ x_true

            for method in methods:
                if method.rhs_dependant:
                    preconditioner, setup_time = contruct_preconditioner(method = method, K = K, b= b, device = device)
                else:
                    preconditioner,setup_time = reusable[method.name]

                result = evalutate_solve(
                        preconditioner = preconditioner,
                        K = K,
                        b = b,
                        x_true= x_true,
                        device = device,
                        setup_time = setup_time,
                        tol = tol,
                        max_iter=max_iter
                )

                results[method.name].append(result)

    return results





#### I Generated the summary printing functions

def reduce_results(results):
    summary = {}

    for name, runs in results.items():
        n = len(runs)

        summary[name] = {
            "converged": sum(
                r["converged"]
                for r in runs
            ) / n,

            "iterations": sum(
                r["iterations"]
                for r in runs
            ) / n,

            "relative_residual": sum(
                r["relative_residual"]
                for r in runs
            ) / n,

            "relative_error": sum(
                r["relative_error"]
                for r in runs
            ) / n,

            "setup_time": sum(
                r["setup_time"]
                for r in runs
            ) / n,

            "solve_time": sum(
                r["solve_time"]
                for r in runs
            ) / n,

            "total_time": sum(
                r["total_time"]
                for r in runs
            ) / n,
        }

    return summary


def summarize_seed_results(seed_results):
    print("\n" + "=" * 120)
    print("FINAL RESULTS: mean ± std across seeds")
    print("=" * 120)

    print(
        f"{'Method':<15}"
        f"{'Conv%':>18}"
        f"{'Iter':>18}"
        f"{'Residual':>24}"
        f"{'RelError':>24}"
        f"{'Setup/Rhs(ms)':>24}"
        f"{'Solve(ms)':>24}"
        f"{'Total/Rhs(ms)':>24}"
    )

    print("-" * 165)

    methods = seed_results[0].keys()

    keys = [
        "converged",
        "iterations",
        "relative_residual",
        "relative_error",
        "setup_time",
        "solve_time",
        "total_time",
    ]

    for method in methods:
        stats = {}

        for key in keys:
            values = torch.tensor(
                [
                    result[method][key]
                    for result in seed_results
                ],
                dtype=torch.float64,
            )

            stats[key] = (
                values.mean().item(),
                values.std(unbiased=True).item(),
            )

        conv_m, conv_s = stats["converged"]
        iter_m, iter_s = stats["iterations"]
        res_m, res_s = stats["relative_residual"]
        err_m, err_s = stats["relative_error"]
        setup_m, setup_s = stats["setup_time"]
        solve_m, solve_s = stats["solve_time"]
        total_m, total_s = stats["total_time"]

        print(
            f"{method:<15}"
            f"{100*conv_m:>8.1f}±{100*conv_s:<7.1f}"
            f"{iter_m:>8.2f}±{iter_s:<7.2f}"
            f"{res_m:>12.3e}±{res_s:<11.3e}"
            f"{err_m:>12.3e}±{err_s:<11.3e}"
            f"{1000*setup_m:>10.3f}±{1000*setup_s:<9.3f}"
            f"{1000*solve_m:>10.3f}±{1000*solve_s:<9.3f}"
            f"{1000*total_m:>10.3f}±{1000*total_s:<9.3f}"
        )



def summarize_results(results):

    print(
        f"{'Method':<15}"
        f"{'Conv%':>10}"
        f"{'Iter':>10}"
        f"{'Residual':>14}"
        f"{'RelError':>14}"
        f"{'Setup/Rhs(ms)':>14}"
        f"{'Solve(ms)':>14}"
        f"{'Total/Rhs(ms)':>14}"
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




def set_seed(seed : int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.use_deterministic_algorithms(True)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def save_methods(methods, dataset_name):

    for method in methods:
        if method.trainable:
            torch.save(method.model.state_dict(),f"{dataset_name}_{method.name}.pt")




def run_full_benchmark(
        name,
        train_dataset,
        val_dataset,
        test_dataset,
        device,
        seeds,
        epochs,
        lr ,
        alpha,
        tol,
        max_iter,
        num_rhs,
        max_train_graphs,
        max_val_graphs,
        max_test_graphs,
 ):

    print(f"Benchmarking on {name}")

    result_path = (
            f"results/"
            f"{RUN_ID}_{name.replace('-', '_')}_preconditioner.jsonl"
            )

    seed_results = []

    for seed in seeds:
        print("\n" + "=" * 100)
        print(f"SEED {seed}")
        print("=" * 100)

        set_seed(seed)
        methods = build_methods(device)

        train_all(
            methods=methods,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            seed=seed,
            epochs=epochs,
            lr=lr,
            alpha=alpha,
            max_train_graphs=max_train_graphs,
            max_val_graphs=max_val_graphs,
        )

        results = benchmark(
            methods=methods,
            test_dataset=test_dataset,
            device=device,
            seed=seed,
            alpha=alpha,
            tol=tol,
            max_iter=max_iter,
            num_rhs=num_rhs,
            max_graphs=max_test_graphs,
        )

        print(f"\nResults for seed {seed}")
        summarize_results(results)

        # Convert all graph/RHS results into ONE summary for this seed

        seed_summary = reduce_results(results)
        seed_results.append(seed_summary)

        save_result_record(
            {
                "type": "seed",
                "dataset": name,
                "seed": seed,
                "alpha": alpha,
                "epochs": epochs,
                "lr": lr,
                "tol": tol,
                "max_iter": max_iter,
                "num_rhs": num_rhs,
                "max_train_graphs": max_train_graphs,
                "max_val_graphs": max_val_graphs,
                "max_test_graphs": max_test_graphs,
                "results": seed_summary,
            },
            result_path,
        )

    final_summary = aggregate_seed_results(seed_results)

    save_result_record(
        {
            "type": "aggregate",
            "dataset": name,
            "seeds": seeds,
            "alpha": alpha,
            "epochs": epochs,
            "lr": lr,
            "tol": tol,
            "max_iter": max_iter,
            "num_rhs": num_rhs,
            "max_train_graphs": max_train_graphs,
            "max_val_graphs": max_val_graphs,
            "max_test_graphs": max_test_graphs,
            "results": final_summary,
        },
        result_path,
    )

    print(f"\nSaved results to: {result_path}")



def load_dataset(name):
    if name in (PEPTIDES_STRUCT, PEPTIDES_FUNC):
        return load_lrgb_datasets(name)
    elif name in (ECHO_SSSP, ECHO_CHARGE, ECHO_ENERGY):
        return load_echo_dataset(name)



def run_paper_benchmark(args):
    args.epochs = 50
    args.lr = 1e-3
    args.num_rhs = 5
    args.alpha = 8.0
    args.seeds = [0,1,2]
    args.max_iter = 1000
    args.tol = 1e-6



    task = ECHO_SSSP
    train,val,test = load_dataset(name = task)
    run_full_benchmark(
        name = task,
        train_dataset = train,
        val_dataset = val,
        test_dataset = test,
        device = args.device,
        seeds = args.seeds,
        epochs = args.epochs,
        lr = args.lr ,
        alpha = args.alpha,
        tol = args.tol,
        max_iter = args.max_iter,
        num_rhs = args.num_rhs,
        max_train_graphs = None,
        max_val_graphs = None,
        max_test_graphs = None,
    )


    task = PEPTIDES_STRUCT
    train,val,test = load_dataset(name = task)
    run_full_benchmark(
        name = task,
        train_dataset = train,
        val_dataset = val,
        test_dataset = test,
        device = args.device,
        seeds = args.seeds,
        epochs = args.epochs,
        lr = args.lr ,
        alpha = args.alpha,
        tol = args.tol,
        max_iter = args.max_iter,
        num_rhs = args.num_rhs,
        max_train_graphs = 2000,
        max_val_graphs = 500,
        max_test_graphs = 500,
    )






def main():
    parser = argparse.ArgumentParser()

    # Peptides-struct is loaded from data/LRGB via torch_geometric.datasets.LRGBDataset
    parser.add_argument("--device", type = str,default = "cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type = int,nargs="+",default = [0,1,2])
    parser.add_argument("--alpha", type = float, default = 8.0)
    parser.add_argument("--epochs", type = int,default = 40)
    parser.add_argument("--lr", type = float,default = 1e-3)
    parser.add_argument("--max-train", type = int, default = None)
    parser.add_argument("--max-val", type = int, default = None)
    parser.add_argument("--max-test", type = int, default = None)
    parser.add_argument("--num-rhs", type = int, default = 5)
    parser.add_argument("--tol", type = float, default = 1e-6)
    parser.add_argument("--dataset", type = str, default = PEPTIDES_STRUCT)
    parser.add_argument("--max-iter", type = int, default = 1000)




    args = parser.parse_args()
    task = args.dataset

    if args.dataset == "paper":
        run_paper_benchmark(args)
        return

    train,val,test = load_dataset(name = task)

    run_full_benchmark(
        name = task,
        train_dataset = train,
        val_dataset = val,
        test_dataset = test,
        device = args.device,
        seeds = args.seeds,
        epochs = args.epochs,
        lr = args.lr ,
        alpha = args.alpha,
        tol = args.tol,
        max_iter = args.max_iter,
        num_rhs = args.num_rhs,
        max_train_graphs = args.max_train,
        max_val_graphs = args.max_val,
        max_test_graphs = args.max_test,
    )








if __name__ == "__main__":
    main()





