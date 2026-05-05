# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from DataLoader import DataLoader
from EdgeSampler import sample_edges
from GraphConstructor import GraphConstructor
from LatentDistanceModel import LatentDistanceModel

# %% HYPERPARAMETERs
subset_percent = 0.1
k = 5  # K-nearest-neighbors
lr = 0.05
weight_decay = 1e-4
batch_size = None  # Virker ekstremt dårligt ud fra tests. Skal nok bare undgås at bruge.
patience = 50
min_delta = 1e-4
pca_n = 50

loader = DataLoader()
constructor = GraphConstructor()
X, y = loader.load_MNIST(subset_percent=subset_percent)
pca = PCA(n_components=pca_n)
X = torch.tensor(pca.fit_transform(X), dtype=torch.float32)

adjacency_matrix = constructor.construct_knn(X, k_neighbors=k, undirected=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
adjacency_matrix = adjacency_matrix.to(device)
ldm = LatentDistanceModel(adjacency_matrix=adjacency_matrix, data_labels=y, output_dimension=2,data=X).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(ldm.parameters(), lr=lr, weight_decay=weight_decay)


# %%


# %%
epochs = 500
best_loss = float("inf")
stale_epochs = 0

t0 = time.time()

for epoch in range(epochs):
    optimizer.zero_grad()

    batch_senders, batch_receivers, targets = sample_edges(adjacency_matrix, batch_size=batch_size, neg_ratio=5.0)

    logits = ldm(batch_senders, batch_receivers)

    loss = loss_fn(logits, targets)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | Alpha: {ldm.alpha.item():.4f}")
        # print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")

    if loss.item() < best_loss - min_delta:
        best_loss = loss.item()
        stale_epochs = 0
    else:
        stale_epochs += 1

    if stale_epochs >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

elapsed = time.time() - t0
print(f"Training took {elapsed:.2f}s")

ldm.visualize()
# %%
ldm.beta
