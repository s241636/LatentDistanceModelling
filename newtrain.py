# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from DataLoader import DataLoader
from EdgeSampler import get_distances, sample_edges
from GraphConstructor import GraphConstructor
from LatentDistanceModel import LatentDistanceModel

# %% HYPERPARAMETER
subset_percent = 0.01
k = 20  # K-nearest-neighbors
lr = 0.05
weight_decay = 1e-4
negative_ratio = 10


# %% MODEL
loader = DataLoader()
constructor = GraphConstructor()
# X, y = loader.load_MNIST(subset_percent=subset_percent)
X = loader.load_mammoth(subset_percent=subset_percent)
print(len(X))

distances = get_distances(X)
adjacency_matrix = constructor.construct_knn(X, k_neighbors=k, undirected=False)
ldm = LatentDistanceModel(
    adjacency_matrix=adjacency_matrix, output_dimension=2
)
optimizer = torch.optim.Adam(ldm.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = nn.BCEWithLogitsLoss()


# %% TRAINING
epochs = 200
for epoch in range(epochs):

    optimizer.zero_grad()
    edges, edge_distances, targets = sample_edges(distances=distances, k=k, negative_ratio=negative_ratio)

    # logits = ldm(batch_senders, batch_receivers)
    logits = ldm(edges)
    loss = loss_fn(logits, targets)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
            f"Alpha: {ldm.alpha.item():.4f} | ",
        )


ldm.visualize()


# %%
