import argparse

import torch

import json
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from tqdm import tqdm


from pathlib import Path 
from datetime import datetime

from src.benchmarks.preconditioner_benchmark import (build_methods, train_all, graph_to_sys, set_seed, load_dataset)
from src.models.gcn import GCN, EchoGCN
from src.models.mpssm import SSMPCGPreconditioner
from src.utils.preconditioner import Preconditioner
from src.utils.metrices import count_params


PEPTIDES_STRUCT = "Peptides-struct"
PEPTIDES_FUNC = "Peptides-func"
ECHO_SSSP = "ECHO-SSSP"
ECHO_CHARGE = "ECHO-Charge"

SAVE_RESULTS = False



METRIC = {
        PEPTIDES_STRUCT : "mae",
        PEPTIDES_FUNC : "ap",
        ECHO_SSSP : "mae",
        ECHO_CHARGE: "mae",
}


ATOM_DIMS = (119, 5 , 12 , 12, 10, 6 , 6 , 2 ,2)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

def graph_to_norm_adj(data, dtype = torch.float32, device = "cuda"):
    n = data.num_nodes

    src, dst = data.edge_index.to(device)


    A = torch.zeros((n,n), dtype = dtype, device = device)

    A[src,dst] = 1.0
    A[dst,src] = 1.0

    A.fill_diagonal_(1.0)

    degree = A.sum(dim = 1)

    D_inv_sqrt= torch.diag(torch.rsqrt(degree))

    A_norm = D_inv_sqrt @ A @ D_inv_sqrt 

    return A_norm


class PreconditionerFactory:
    def __init__(self,model,device, alpha = 1.0):
        self.model = model
        self.device = device
        self.alpha = alpha
        self.cache = {}
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

    def key(self, data):
        edge_index = data.edge_index.detach().cpu()
        return int(data.num_nodes), edge_index.numpy().tobytes()

    @torch.no_grad()
    def build(self, data):
        K = graph_to_sys(data, alpha = self.alpha, device = self.device)

        preconditioner = SSMPCGPreconditioner(self.model, K)

        I = torch.eye(K.shape[0], dtype = K.dtype, device = K.device)

        return preconditioner.apply(I)

    def __call__(self, data, graph_idx = None):
        key = self.key(data)

        if key not in self.cache:
            self.cache[key] = self.build(data).detach().cpu()

        return MatrixPreconditioner(self.cache[key].to(self.device))



class MatrixPreconditioner(Preconditioner):
    def __init__(self, P):
        super().__init__()
        self.P = P

    def apply(self, H):
        return self.P @ H


def average_precision(predictions, targets):
    predictions = torch.sigmoid(predictions)
    return average_precision_score(targets.detach().cpu().numpy(), predictions.detach().cpu().numpy(),average="macro")



from torch_geometric.data import InMemoryDataset

class EchoSplit(InMemoryDataset):
    def __init__(self, path):
        super().__init__(root = None)

        payload = torch.load(path, map_location="cpu", weights_only=False)

        self._data = payload[0]
        self.slices = payload[1]
        self._data_list = None


@torch.no_grad()
def evaluate_gcn(model,task, dataloader, device , preconditioner_factory = None, max_graphs = None):
    model.eval()

    if task in (PEPTIDES_STRUCT,ECHO_SSSP, ECHO_CHARGE):
        total_error= 0
        total_targets= 0

    elif task == PEPTIDES_FUNC:
        predicitons = []
        targets = []




    for graphs in dataloader:
        for data in graphs:
            # if max_graphs is not None and graph_idx >= max_graphs:
            #     break

            X = data.x.float().to(device)

            if preconditioner_factory is None:
                A_norm = graph_to_norm_adj(data, device = device)
                prediction = model(X, A_norm = A_norm)
            else: 
                preconditioner = preconditioner_factory(data,)
                prediction = model(X, preconditioner = preconditioner)

            

            target = data.y.to(dtype = prediction.dtype, device = prediction.device).reshape_as(prediction)

            if task == PEPTIDES_STRUCT:
                mask = torch.isfinite(target)
                total_error += (prediction[mask] - target[mask]).abs().sum().item()
                total_targets += mask.sum().item()

            elif task == PEPTIDES_FUNC:
                predicitons.append(prediction.detach().cpu())
                targets.append(target.detach().cpu())
            elif task in (ECHO_SSSP,ECHO_CHARGE):
                total_error += (prediction - target).abs().sum().item()
                total_targets += target.numel()
            


    if task in (PEPTIDES_STRUCT, ECHO_SSSP, ECHO_CHARGE):
        mae = total_error / max(total_targets, 1)


        return {
                "mae" : mae
        }

    elif task == PEPTIDES_FUNC:
        predicitons = torch.cat(predicitons, dim = 0) 
        targets = torch.cat(targets, dim = 0) 

        ap = average_precision(predicitons, targets)

        return {
                "ap" : ap
        }






def masked_mae(prediction, target):

    target = target.to(device = prediction.device, dtype = prediction.dtype)

    target = target.reshape_as(prediction)

    mask = torch.isfinite(target)

    if not mask.any():
        return prediction.sum() * 0.0

    return (prediction[mask] - target[mask]).abs().mean()


def train_gcn(model, task, train_loader,val_loader, device, preconditioner_factory = None, epochs = 100,lr = 1e-4, max_train_graphs = None, max_val_graphs = None):


    metric_name = METRIC[task]

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)


    if task in (PEPTIDES_STRUCT,ECHO_SSSP, ECHO_CHARGE):
        best_metric = float('inf') 
    elif task == PEPTIDES_FUNC:
        best_metric = -float('inf') 



    best_model = None
    best_epoch = -1


    history = []



    for epoch in range(epochs):
        total_loss= 0.0
        count= 0




        model.train()
        for graphs in tqdm(train_loader, desc = f'Epoch {epoch}', leave = False):

            eff_batch_size = len(graphs)
            batch_loss = 0.0

            optimizer.zero_grad()
            for data in graphs:



                X = data.x.float().to(device)
                # y = data.y.float().to(device)


                if preconditioner_factory is None:
                    A_norm = graph_to_norm_adj(data, device = device)
                    prediction = model(X, A_norm = A_norm)
                else: 
                    preconditioner = preconditioner_factory(data)
                    prediction = model(X, preconditioner = preconditioner)

                target = data.y.to(dtype = prediction.dtype, device = prediction.device).reshape_as(prediction)

                if task == PEPTIDES_STRUCT:
                    loss = masked_mae(prediction,target)
                elif task == PEPTIDES_FUNC: 
                    loss = F.binary_cross_entropy_with_logits(prediction,target)
                elif task in (ECHO_SSSP,ECHO_CHARGE):
                    loss = F.mse_loss(prediction,target)

                (loss / eff_batch_size).backward()
                batch_loss += loss.item()

            optimizer.step()


            total_loss += batch_loss 
            count += eff_batch_size


        train_loss = total_loss /  max(count, 1)

        val_results = evaluate_gcn(
                model = model,
                task = task ,
                dataloader = val_loader,
                device = device,
                preconditioner_factory=preconditioner_factory,
                max_graphs = max_val_graphs
        )

        if task in (PEPTIDES_STRUCT,ECHO_SSSP, ECHO_CHARGE):
            val_metric = val_results[metric_name] 
            improved = val_metric <= best_metric

        elif task == PEPTIDES_FUNC: 
            val_metric = val_results[metric_name] 
            improved = val_metric >= best_metric


        if improved:
            best_metric = val_metric
            best_epoch = epoch
            best_model =  {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


        history.append({
            "epochs" : epoch,
            "train_loss": train_loss,
            f"val_{metric_name}": val_metric
        })

        print(f"{"*" if improved else " "} Epoch {epoch} | Train Loss {train_loss} | Validation {metric_name}  {val_metric}")




    model.load_state_dict(best_model)

    print(f"Best Epoch {best_epoch} | Best Validation {metric_name} {best_metric}")

    return model, history



def train_preconditioner(device,train_dataset,val_dataset, seed, alpha, epochs, lr, max_train_graphs, max_val_graphs):

    set_seed(seed)

    methods = build_methods(device)

    ssmpcg = next(method for method in methods if method.name == "MPSSMPCG")

    train_all(
            methods = [ssmpcg],
            alpha = alpha,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            seed=seed,
            epochs = epochs,
            lr = lr,
            max_train_graphs=max_train_graphs,
            max_val_graphs=max_val_graphs
    )

    return ssmpcg.model



### GENERATED
def save_history(
    task,
    alpha,
    gcn_layers,
    pgcn_layers,
    seed,
    local_history,
    pgcn_history,
    path=None,
):
    if not SAVE_RESULTS:
        return
    if path is None:
        path = f"results/{RUN_ID}_training_histories.jsonl"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "task": task,
        "alpha": alpha,
        "gcn_layers": gcn_layers,
        "pgcn_layers": pgcn_layers,
        "seed": seed,
        "local": local_history,
        "preconditioned": pgcn_history,
    }

    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

def benchmark(
        task, 
        local_train_loader,
        pgcn_train_loader,
        val_loader,
        test_loader,
        preconditioner_model,
        device,
        alpha,
        hidden_dim,
        gcn_num_layers,
        pgcn_num_layers,
        epochs,
        lr,
        seed, 
        max_train_graphs,
        max_val_graphs,
        max_test_graphs
  ):

    if task == PEPTIDES_STRUCT:
        def build_model(num_layers):
            return GCN(out_dim = 11, hidden_dim= hidden_dim, num_layers=num_layers).to(device)
    elif task == PEPTIDES_FUNC:
        def build_model(num_layers):
            return GCN(out_dim = 10, hidden_dim= hidden_dim, num_layers=num_layers).to(device)
    elif task in (ECHO_SSSP,ECHO_CHARGE):
        def build_model(num_layers):
            in_dim = pgcn_train_loader.dataset[0].x.shape[-1]
            return EchoGCN(in_dim = in_dim, hidden_dim= hidden_dim, num_layers=num_layers).to(device)

    preconditioner_factory = PreconditionerFactory(model=preconditioner_model, device = device, alpha = alpha)


    set_seed(seed)


    preconditioned_gcn = build_model(pgcn_num_layers)
    print(f"Benchmarking {task} | PreconditionedGCN | Params {count_params(preconditioned_gcn)}")
    preconditioned_gcn, pgcn_history = train_gcn(
            model = preconditioned_gcn, 
            train_loader=pgcn_train_loader,
            val_loader=val_loader,
            device = device,
            task = task,
            preconditioner_factory=preconditioner_factory,
            epochs = epochs,
            lr = lr,
            max_train_graphs=max_train_graphs,
            max_val_graphs=max_val_graphs
    )

    preconditioned_results = evaluate_gcn(
            model = preconditioned_gcn,
            dataloader=test_loader,
            device = device,
            task = task,
            max_graphs=max_test_graphs,
            preconditioner_factory = preconditioner_factory
    )

    set_seed(seed)



    local_gcn = build_model(gcn_num_layers) 
    print(f"Benchmarking {task} | LocalGCN | Params {count_params(local_gcn)}")
    local_gcn, local_history = train_gcn(
            model = local_gcn, 
            train_loader=local_train_loader,
            val_loader=val_loader,
            device = device,
            task = task,
            preconditioner_factory=None,
            epochs = epochs,
            lr = lr,
            max_train_graphs=max_train_graphs,
            max_val_graphs=max_val_graphs
    )

    local_results = evaluate_gcn(
            model = local_gcn,
            dataloader=test_loader,
            device = device,
            task = task,
            preconditioner_factory = None
    )

    save_history(
            task = task,
            alpha = alpha,
            gcn_layers = gcn_num_layers,
            pgcn_layers = pgcn_num_layers,
            seed = seed,
            local_history=local_history,
            pgcn_history=pgcn_history,
    )


    return local_results[METRIC[task]], preconditioned_results[METRIC[task]]


def make_loader(dataset, batch_size = 32, max_graphs = None, shuffle = False, seed = 0):

    if max_graphs is not None:
        dataset = Subset(dataset,range(min(max_graphs, len(dataset))))

    gen = torch.Generator()
    gen.manual_seed(seed)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda graphs : graphs, generator=gen if shuffle else None)





def benchmark_dataset(task,args,train_dataset, val_dataset, test_dataset):

    print("\n" + "=" * 100)
    print("Settings")
    print(f"alpha {args.alpha}")
    print(f"lr {args.lr}")
    print(f"local gcn layers {args.gcn_num_layers}")
    print(f"pgcn layers {args.pgcn_num_layers}")
    print(f"hidden dim {args.hidden_dim}")
    print("\n" + "=" * 100)

    device = torch.device(args.device)
    batch_size = args.batch_size

    seed_results = []

    for seed in args.seeds:
        print("\n" + "=" * 100)
        print(f"{task} | SEED {seed}")
        print("=" * 100)

        preconditioner_model = train_preconditioner(
                device=device,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                seed = seed,
                alpha = args.alpha,
                epochs = args.pre_epochs,
                lr = args.pre_lr,
                max_train_graphs=args.max_train_graphs,
                max_val_graphs=args.max_val_graphs,
        )



        local_train_loader = make_loader(dataset=train_dataset, batch_size=batch_size, max_graphs=args.max_train_graphs, shuffle = True, seed = seed)
        pgcn_train_loader = make_loader(dataset=train_dataset, batch_size=batch_size, max_graphs=args.max_train_graphs, shuffle = True, seed = seed)
        val_loader = make_loader(dataset=val_dataset, batch_size=batch_size, max_graphs=args.max_val_graphs, shuffle = False, seed = seed)
        test_loader = make_loader(dataset=test_dataset, batch_size=batch_size, max_graphs=args.max_test_graphs, shuffle = False, seed = seed)

        local_res, pre_res = benchmark(
                task = task,
                local_train_loader=local_train_loader,
                pgcn_train_loader=pgcn_train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                gcn_num_layers=args.gcn_num_layers,
                pgcn_num_layers=args.pgcn_num_layers,
                preconditioner_model=preconditioner_model,
                device = device,
                alpha = args.alpha,
                hidden_dim=args.hidden_dim,
                epochs = args.epochs,
                lr = args.lr,
                seed = seed,
                max_train_graphs = args.max_train_graphs,
                max_val_graphs= args.max_val_graphs,
                max_test_graphs= args.max_test_graphs,
        )

        seed_results.append({
            "seed": seed,
            "local": local_res,
            "preconditioned": pre_res,
            })

        print(
                f"\nSeed {seed} | "
                f"LocalGCN {METRIC[task]}: {local_res:.6f} | "
                f"PreconditionedGCN {METRIC[task]}: {pre_res:.6f}"
        )

    local_values = torch.tensor(
            [r["local"] for r in seed_results],
            dtype=torch.float64,
            )

    pre_values = torch.tensor(
            [r["preconditioned"] for r in seed_results],
            dtype=torch.float64,
            )

    local_mean = local_values.mean().item()
    pre_mean = pre_values.mean().item()

    if len(seed_results) > 1:
        local_std = local_values.std(unbiased=True).item()
        pre_std = pre_values.std(unbiased=True).item()
    else:
        local_std = 0.0
        pre_std = 0.0

    print("\n" + "=" * 100)
    print(f"FINAL RESULTS | {task}")
    print("=" * 100)

    print(
            f"LocalGCN          | {METRIC[task]} "
            f"{local_mean:.6f} ± {local_std:.6f}"
            )

    print(
            f"PreconditionedGCN | {METRIC[task]} "
            f"{pre_mean:.6f} ± {pre_std:.6f}"
            )

    return {
        "task": task,
        "alpha": args.alpha,
        "gcn_layers": args.gcn_num_layers,
        "pgcn_layers": args.pgcn_num_layers,

        "local_mean": local_mean,
        "local_std": local_std,

        "preconditioned_mean": pre_mean,
        "preconditioned_std": pre_std,

        "seeds": seed_results,
    }

### GENERATED
def save_result(
    result,
    path=None,
):
    if not SAVE_RESULTS:
        return

    if path is None:
        path = f"results/{RUN_ID}_downstream.jsonl"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a") as f:
        f.write(json.dumps(result) + "\n")



def print_saved_gcn_results(path, show_seeds=False):
    path = Path(path)

    with open(path, "r") as f:
        records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if not records:
        print("No results found.")
        return

    print("\n" + "=" * 110)
    print("DOWNSTREAM GCN RESULTS")
    print("=" * 110)

    print(
        f"{'Task':<20}"
        f"{'α':>6}"
        f"{'GCN':>8}"
        f"{'PGCN':>8}"
        f"{'Local GCN':>25}"
        f"{'Preconditioned GCN':>28}"
    )

    print("-" * 110)

    for r in records:
        local = f"{r['local_mean']:.6f} ± {r['local_std']:.6f}"
        pre = (
            f"{r['preconditioned_mean']:.6f} "
            f"± {r['preconditioned_std']:.6f}"
        )

        print(
            f"{r['task']:<20}"
            f"{r['alpha']:>6.1f}"
            f"{r['gcn_layers']:>8}"
            f"{r['pgcn_layers']:>8}"
            f"{local:>25}"
            f"{pre:>28}"
        )

        if show_seeds:
            for seed_result in r["seeds"]:
                print(
                    f"{'':<20}"
                    f"{'':>6}"
                    f"{'':>8}"
                    f"{'':>8}"
                    f"seed {seed_result['seed']}: "
                    f"{seed_result['local']:.6f}"
                    f"{'':>7}"
                    f"{seed_result['preconditioned']:.6f}"
                )

    print("=" * 110)



    # parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # parser.add_argument("--alpha", type = float, default = 1.0)
    #
    # parser.add_argument("--pre-hidden-dim", type = int, default = 32)
    # parser.add_argument("--pre-epochs", type = int, default = 20)
    # parser.add_argument("--pre-layers", type = int, default = 5)
    # parser.add_argument("--pre-lr", type = float, default = 1e-4)
    #
    # parser.add_argument("--hidden-dim", type = int, default = 384)
    # parser.add_argument("--gcn-num-layers", type = int, default = 4)
    # parser.add_argument("--pgcn-num-layers", type = int, default = 1)
    # parser.add_argument("--epochs", type = int, default = 100)
    #
    # parser.add_argument("--batch-size", type = int, default = 16)
    #
    # parser.add_argument("--seeds", type = int, nargs = "+", default = [0,1,2])
    # parser.add_argument("--lr", type = float, default = 1e-5)
    # parser.add_argument("--max-train-graphs", type = int, default = 5000)
    # parser.add_argument("--max-val-graphs", type = int, default = 2000)
    # parser.add_argument("--max-test-graphs", type = int, default = 2000)
    # parser.add_argument("--dataset", default = PEPTIDES_STRUCT)

def run_paper_benchmark(task, args):
    args.pre_epochs = 30
    args.pre_lr = 1e-4
    args.hidden_dim = 384
    args.epochs = 300
    args.batch_size = 32
    args.seeds = [0,1,2]



    train,val,test = load_dataset(name = task)

    args.max_train_graphs = None
    args.max_val_graphs = None
    args.max_test_graphs = None

    args.epochs = 300
    args.pre_epochs = 30




    # ## GCN 1 vs PGCN 1, alpha diff
    # args.gcn_num_layers=1
    # args.pgcn_num_layers=1
    #
    # for alpha in [1.0, 2.0, 4.0, 8.0]:
    #     train,val,test = load_dataset(name = task)
    #     args.alpha = alpha
    #     results = benchmark_dataset(
    #             task = task,
    #             args = args,
    #             train_dataset=train,
    #             val_dataset=val,
    #             test_dataset=test
    #     )
    #
    #     save_result(results)
    #
    #
    # # # GCN 4 vs PGCN 2, alpha 8
    # train,val,test = load_dataset(name = task)
    # args.alpha = 8.0
    # args.gcn_num_layers=4
    # args.pgcn_num_layers=2
    # results = benchmark_dataset(
    #         task = task,
    #         args = args,
    #         train_dataset=train,
    #         val_dataset=val,
    #         test_dataset=test
    # )
    # save_result(results)
    #
    # ## GCN 4 vs PGCN 4, alpha 8
    # train,val,test = load_dataset(name = task)
    # args.alpha = 8.0
    # args.gcn_num_layers=8
    # args.pgcn_num_layers=4
    # results = benchmark_dataset(
    #         task = task,
    #         args = args,
    #         train_dataset=train,
    #         val_dataset=val,
    #         test_dataset=test
    # )
    # save_result(results)


    # GCN 2 vs PGCN 8, alpha 8
    train,val,test = load_dataset(name = task)
    args.alpha = 8.0
    args.gcn_num_layers=2
    args.pgcn_num_layers=8
    results = benchmark_dataset(
            task = task,
            args = args,
            train_dataset=train,
            val_dataset=val,
            test_dataset=test
    )

    save_result(results)




def main():
    parser = argparse.ArgumentParser()


    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--alpha", type = float, default = 1.0)

    parser.add_argument("--pre-hidden-dim", type = int, default = 32)
    parser.add_argument("--pre-epochs", type = int, default = 20)
    parser.add_argument("--pre-layers", type = int, default = 5)
    parser.add_argument("--pre-lr", type = float, default = 1e-4)

    parser.add_argument("--hidden-dim", type = int, default = 384)
    parser.add_argument("--gcn-num-layers", type = int, default = 4)
    parser.add_argument("--pgcn-num-layers", type = int, default = 1)
    parser.add_argument("--epochs", type = int, default = 100)

    parser.add_argument("--batch-size", type = int, default = 16)

    parser.add_argument("--seeds", type = int, nargs = "+", default = [0,1,2])
    parser.add_argument("--lr", type = float, default = 1e-5)
    parser.add_argument("--max-train-graphs", type = int, default = 5000)
    parser.add_argument("--max-val-graphs", type = int, default = 2000)
    parser.add_argument("--max-test-graphs", type = int, default = 2000)
    parser.add_argument("--dataset", default = PEPTIDES_STRUCT)

    args = parser.parse_args()
    task = args.dataset

    if args.dataset == "paper":
        run_paper_benchmark(task = "ECHO-SSSP", args=args)
        return

    train,val,test = load_dataset(name = task)

    benchmark_dataset(
            task = task,
            args = args,
            train_dataset=train,
            val_dataset=val,
            test_dataset=test
    )

    # print_saved_gcn_results("results/20260830_000300_downstream.jsonl")





if __name__ == "__main__":
    main()


















