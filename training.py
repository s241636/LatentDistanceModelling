# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from DataLoader import DataLoader
from EdgeSampler import sample_edges
from GraphConstructor import GraphConstructor
from LatentDistanceModel import LatentDistanceModel
from testloss import heavy_tail_ldm_loss

# %% HYPERPARAMETERs
subset_percent = 0.1
k = 5  # K-nearest-neighbors
lr = 0.05
weight_decay = 1e-4
batch_size = (
    None  # Virker ekstremt dårligt ud fra tests. Skal nok bare undgås at bruge.
)
patience = 500
min_delta = 1e-4
pca_n = 50
knn_eval_k = 10
exaggeration_factor = 2
burn_in_epochs = 100


def latent_knn_accuracy(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10,
) -> float:
    positions = embeddings.detach().cpu().numpy()
    label_array = labels.detach().cpu().numpy()

    neighbors = NearestNeighbors(
        n_neighbors=min(len(positions), k + 1),
        metric="euclidean",
    )
    neighbors.fit(positions)
    neighbor_indices = neighbors.kneighbors(return_distance=False)[:, 1:]

    return float((label_array[:, None] == label_array[neighbor_indices]).mean())


loader = DataLoader()
constructor = GraphConstructor()
X, y = loader.load_MNIST(subset_percent=subset_percent)
adjacency_matrix = constructor.construct_knn(X, k_neighbors=k, undirected=False)


device = torch.device("cuda" if torch.cuda.is_available() else "mps")
# %%
adjacency_matrix = adjacency_matrix.to(device)


ldm = LatentDistanceModel(
    adjacency_matrix=adjacency_matrix, data_labels=y, output_dimension=2
).to(device)
optimizer = torch.optim.Adam(ldm.parameters(), lr=lr, weight_decay=weight_decay)


# %%
epochs = 500
best_knn_accuracy = float("-inf")
stale_epochs = 0
t0 = time.time()
# current_weight = exaggeration_factor if epoch < burn_in_epochs else 1.0

# # PyTorch requires pos_weight to be a tensor
# weight_tensor = torch.tensor([current_weight], device=device)

for epoch in range(epochs):
    # if epoch == 0:
    #     loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0))
    # elif epoch == burn_in_epochs:
    #     loss_fn = nn.BCEWithLogitsLoss()

    optimizer.zero_grad()

    batch_senders, batch_receivers, targets = sample_edges(
        adjacency_matrix, batch_size=batch_size
    )

    logits = ldm(batch_senders, batch_receivers)

    loss = loss_fn(logits, targets)

    loss.backward()
    optimizer.step()
    knn_accuracy = latent_knn_accuracy(ldm.Z.weight, y, k=knn_eval_k)

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
            f"Alpha: {ldm.alpha.item():.4f} | "
            f"Latent KNN@{knn_eval_k}: {knn_accuracy:.4f}"
        )

    if knn_accuracy > best_knn_accuracy + min_delta:
        best_knn_accuracy = knn_accuracy
        stale_epochs = 0
    else:
        stale_epochs += 1

    if stale_epochs >= patience:
        print(f"Early stopping at epoch {epoch} based on Latent KNN@{knn_eval_k}")
        break

elapsed = time.time() - t0
print(f"Training took {elapsed:.2f}s")
final_knn_accuracy = latent_knn_accuracy(ldm.Z.weight, y, k=knn_eval_k)
print(f"Final Latent KNN@{knn_eval_k}: {final_knn_accuracy:.4f}")

ldm.visualize()

# %%
