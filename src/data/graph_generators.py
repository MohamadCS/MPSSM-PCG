import torch


def path_graph(num_nodes : int) -> tuple[torch.Tensor, torch.Tensor]:
    edges = []


    for i in range(num_nodes - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))

    edge_index = torch.tensor(edges, dtype = torch.long).t()
    weights = torch.ones(edge_index.shape[1], dtype = torch.float32)

    return edge_index, weights




