# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from DataLoader import DataLoader
from GraphConstructor import GraphConstructor
from LatentDistanceModel import LatentDistanceModel, LatentDistanceModelAI

# %%
# --- 1. Your Existing Setup ---
loader = DataLoader()
constructor = GraphConstructor()
X, y = loader.load_MNIST(subset_percent=0.1)
adjacency_matrix = constructor.construct_knn(X, k_neighbors=5)

# (For the sake of making this script runnable, assuming adjacency_matrix and y exist)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
adjacency_matrix = adjacency_matrix.to(device)

num_nodes = adjacency_matrix.shape[0]
ldm = LatentDistanceModelAI(num_nodes=num_nodes, data_labels=y, output_dimension=2).to(device)
    
# --- 2. Training Preparation ---
# BCEWithLogitsLoss handles the Sigmoid conversion internally
loss_fn = nn.BCEWithLogitsLoss()

# We add weight_decay (L2 Regularization) to stop the space from inflating infinitely
optimizer = torch.optim.Adam(ldm.parameters(), lr=0.05, weight_decay=1e-4)

# Extract the fixed indices of all true edges (the 1s)
# We ignore the diagonal (self-loops) by ensuring sender != receiver
edge_coords = torch.nonzero(adjacency_matrix == 1)
edge_coords = edge_coords[edge_coords[:, 0] != edge_coords[:, 1]]

pos_senders = edge_coords[:, 0]
pos_receivers = edge_coords[:, 1]
num_edges = len(pos_senders)

# We create a 1D tensor of targets: half 1s (for edges) and half 0s (for non-edges)
targets = torch.cat([
    torch.ones(num_edges, device=device), 
    torch.zeros(num_edges, device=device)
])

# --- 3. The Training Loop ---
epochs = 500

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # 1. Negative Sampling
    # Randomly pick 'num_edges' amount of senders and receivers
    random_senders = torch.randint(0, num_nodes, (num_edges,), device=device)
    random_receivers = torch.randint(0, num_nodes, (num_edges,), device=device)
    
    # 2. Combine true edges with the negative samples
    batch_senders = torch.cat([pos_senders, random_senders])
    batch_receivers = torch.cat([pos_receivers, random_receivers])
    
    # 3. Forward Pass (only on the selected pairs)
    logits = ldm(batch_senders, batch_receivers)
    
    # 4. Calculate Loss
    loss = loss_fn(logits, targets)
    
    # 5. Backpropagation
    loss.backward()
    optimizer.step()
    
    # 6. Logging
    # print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | Alpha: {ldm.alpha.item():.4f}")
    if epoch % 10 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | Alpha: {ldm.alpha.item():.4f}")

# --- 4. Visualize Results ---
ldm.visualize()
# %%
