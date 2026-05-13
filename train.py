# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from DataLoader import load_MNIST, load_s_hole, load_swissroll
from EdgeSampler import get_distances, sample_edges
from GraphConstructor import construct_knn
from LatentDistanceModel import LatentDistanceModel

# %% HYPERPARAMETER
subset_percent = 0.01
k = 20  # K-nearest-neighbors
lr = 0.05
weight_decay = 1e-4
negative_ratio = 10


# %% MODEL


X,y = load_swissroll(n_samples=5000)

distances = get_distances(X)
adjacency_matrix = construct_knn(X, k_neighbors=k, undirected=False)
ldm = LatentDistanceModel(
    data = X,
    output_dimension=2,
    data_labels=y,
    
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


ldm.visualize(datatype="swissroll", save_path="evals/model2/swissroll|n=5000|k=20|epoch=200")



