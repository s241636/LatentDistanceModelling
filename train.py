# %%
import time

import torch
import torch.nn as nn

from LatentDistanceModel import LatentDistanceModel
from utils import (
    get_initial_parameters,
    load_MNIST,
    load_s_hole,
    load_swissroll,
    sample_edges,
)

# %% HYPERPARAMETER
k = 20  # K-nearest-neighbors
lr = 0.05
weight_decay = 1e-4
negative_ratio = 20

# %% MODEL
X,y = load_swissroll(n_samples=20000)
# X,y = load_MNIST(subset_percent=0.1)

knn_indices, sigma = get_initial_parameters(X, k_neighbors=k, sigma_q=5)

ldm = LatentDistanceModel(
    data = X,
    output_dimension=2,
    data_labels=y,
)

optimizer = torch.optim.Adam(ldm.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = nn.BCEWithLogitsLoss()


# %% TRAINING
t0 = time.time()
epochs = 200
for epoch in range(epochs):
    optimizer.zero_grad()
    edges, edge_distances, targets = sample_edges(X = X, knn_indices=knn_indices, negative_ratio=negative_ratio, sigma=sigma)
    
    logits = ldm(edges)
    loss = loss_fn(logits, targets)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
            # f"Alpha: {ldm.alpha.item():.4f} | ",
        )
        ldm.visualize(datatype="swissroll", show=False, save_path=f"timelines/shole/{epoch}")
print(f"Time: {time.time() - t0} seconds")

ldm.visualize(datatype="swissroll")