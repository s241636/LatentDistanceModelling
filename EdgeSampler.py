import torch

# AI-GENERATED
# PROMPT (DeepSeek V4 Pro):
# "Create a standalone helper function, for negative sampling. 
# It should take an adjacency matrix, and return the senders recievers and targets. 
# It should also be able to take a batch size, such that it doesn't train on the entire set of positive edges, but can also take a subsample of this.""

def sample_edges(
    adjacency_matrix: torch.Tensor,
    batch_size: int | None = None,
    neg_ratio: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample positive and negative edges for training.

    Args:
        adjacency_matrix: Binary adjacency matrix of shape (N, N).
        batch_size: Number of positive edges to use. If None, all positive
                    edges are used.
        neg_ratio: Number of negative edges sampled per positive edge.
                   Default 1.0 gives a balanced 1:1 batch.

    Returns:
        senders: Indices of sender nodes, shape ((1 + neg_ratio) * num_pos,).
        receivers: Indices of receiver nodes, shape ((1 + neg_ratio) * num_pos,).
        targets: Binary labels (1=positive, 0=negative).
    """
    num_nodes = adjacency_matrix.shape[0]
    device = adjacency_matrix.device

    edge_coords = torch.nonzero(adjacency_matrix == 1)
    edge_coords = edge_coords[edge_coords[:, 0] != edge_coords[:, 1]]

    pos_senders = edge_coords[:, 0]
    pos_receivers = edge_coords[:, 1]

    if batch_size is not None:
        idx = torch.randperm(len(pos_senders), device=device)[:batch_size]
        pos_senders = pos_senders[idx]
        pos_receivers = pos_receivers[idx]

    num_pos = len(pos_senders)
    num_neg = int(num_pos * neg_ratio)

    neg_senders = torch.randint(0, num_nodes, (num_neg,), device=device)
    neg_receivers = torch.randint(0, num_nodes, (num_neg,), device=device)

    senders = torch.cat([pos_senders, neg_senders])
    receivers = torch.cat([pos_receivers, neg_receivers])
    targets = torch.cat([
        torch.ones(num_pos, device=device),
        torch.zeros(num_neg, device=device),
    ])

    return senders, receivers, targets
