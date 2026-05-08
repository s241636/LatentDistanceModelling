# %%
from functools import cache

import numpy as np
import torch
import torch.nn as nn

from DataLoader import DataLoader

# Data: N x D
# - 70.000 x 784 (MNIST)
# Distance: N x N
# - Pairwise distance for alle vektorer
# Sampling:
# - K tætteste distances for hvert vektor (Konstant for alle vektorer, baseret på originalt latent space)
# - K * negative_ratio for hvert vektor (Samples nye hver epoch, et sæt tilfældige neighbors til at )
# - Alle vægtes med p_ij = exp(-(d_ij**2) / sigma**2)
# Training:
# - BCEWithLogits på alle de samplede vektorer, for alle 7000 vektorer.
#   - p_ij = p_ji

# %%
loader = DataLoader()

@cache
def get_distances(X: torch.Tensor) -> torch.Tensor:
    return torch.cdist(X, X)

# %%
def sample_edges(distances: torch.Tensor, k: int, negative_ratio: float) -> tuple[torch.Tensor]:
    negatives_count = round(k * negative_ratio)
    pos_edges = distances.topk(k = k+1, largest=False).indices[:, 1:]
    N = len(distances)
    negative_edges = torch.empty(size=(N, negatives_count))
    for idx, pos_edge in enumerate(pos_edges):
        random_edges = torch.randint(low = 0, high = N, size = (1, negatives_count))
        non_negative_edges = torch.cat((torch.tensor([idx]), pos_edge))

        # Continues drawing random numbers until random edges are not apart of the neighborhood, or the vector itself.
        while torch.isin(random_edges, non_negative_edges).any():
            random_edges = random_edges[~torch.isin(random_edges, non_negative_edges)]
            new_random_edges = torch.randint(low=0, high=N, size=(1, negatives_count - len(random_edges)))
            random_edges = torch.cat((random_edges, new_random_edges.flatten()))
        negative_edges[idx] = random_edges

    edges = torch.cat((pos_edges, negative_edges), dim=1).to(int)
    edge_distances = torch.gather(distances, dim=1, index=edges)
    
    sigma = np.percentile(distances, q=10)
    probs = torch.exp(-(edge_distances**2) / (sigma**2))
    targets = torch.clip(probs, 1e-4, 0.95)
    return edges, edge_distances, targets

