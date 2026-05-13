# %%
from functools import cache

import numpy as np
import torch

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



# %%
def sample_edges(
    X: torch.Tensor,
    knn_indices: np.ndarray,
    negative_ratio: float,
    sigma: float,
) -> tuple[torch.Tensor]:
    
    # Finder index for de k positive edges, for alle nodes.
    k = len(knn_indices[0])
    N = len(knn_indices)
    
    # Mængden af negative edges
    negatives_count = round(k * negative_ratio)
    pos_edges = knn_indices

    # Finder tilsvarende tilfældige "negative" edges.
    # Loop fortsætter til der er præcist k*negative ratio og at de fundne edges ikke er en del af pos_edges, eller node'n selv.
    negative_edges = torch.empty(size=(N, negatives_count))
    for idx, pos_edge in enumerate(pos_edges):
        # Tager tilfældige edges
        random_edges = torch.randint(low=0, high=N, size=(1, negatives_count))
        # Uønskede tilfældige edges (knn og noden selv)
        non_negative_edges = torch.cat((torch.tensor([idx]), pos_edge))
        # Udtager indtil de findes
        while torch.isin(random_edges, non_negative_edges).any():
            random_edges = random_edges[~torch.isin(random_edges, non_negative_edges)]
            new_random_edges = torch.randint(
                low=0, high=N, size=(1, negatives_count - len(random_edges))
            )
            random_edges = torch.cat((random_edges, new_random_edges.flatten()))
        negative_edges[idx] = random_edges

    # Samler edge indices til en tensor. Første k er da de positive edge indices, og de næste er de negative.
    edge_indices = torch.cat((pos_edges, negative_edges), dim=1).to(int)
    
    # Finder afstande, til de givne edges.
    base_points = X.unsqueeze(1)
    target_points = X[edge_indices]
    edge_distances = torch.linalg.norm(base_points - target_points, dim=-1)
    
    # Udregner til sandsynligheder
    probs = torch.exp(-(edge_distances**2) / (sigma**2))
    target_probabilities = torch.clip(probs, 1e-4, 0.95)
    return edge_indices, edge_distances, target_probabilities