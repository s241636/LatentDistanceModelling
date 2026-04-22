
# %%
import random
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import nn
from torchvision import datasets, transforms

from DataLoader import DataLoader
from GraphConstructor import GraphConstructor

# Tanker
# - Først prøver jeg bare at alpha = 0, og beta = 0, for at lave den simpleste model som muligt.
# - Send ting til device

# %%
#https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
class LatentDistanceModel(nn.Module):
    def __init__(self, adjacency_matrix: torch.Tensor, data_labels: np.ndarray, output_dimension: int = 2) -> None:
        super().__init__()
        # Hvordan skal man lave PCA når man teknisk set ikke har det originale data?
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.adjacency_matrix = adjacency_matrix.to(torch.float32)
        self.N = adjacency_matrix.shape[0]
        self.data_labels = data_labels

        # LDM Specifikke parametre
        self.Z = nn.Parameter((0.1 * torch.randn(self.N, output_dimension, device=device)).requires_grad_()) # Initialiserer positionerne tilfældigt
        # self.Z = nn.Embedding(num_embeddings=self.N, embedding_dim=output_dimension)
        self.alpha = nn.Parameter(torch.tensor([0.0])) # Prøver først med alpha som 0
        self.beta = 0 # Prøver også først med beta som 0
        # self.covariates = torch.zeros(size=adjacency_matrix.shape) # Covariate matrice også bare sat til 0 fra start.
        self.covariates = 0 # Sætter til 0 først, skal egentligt være en matrice men skal lige have fundet ud af det basale først.
    
    def forward(self):
        logits = self.alpha + self.beta * self.covariates - torch.cdist(self.Z, self.Z) # Logodds for each edge
        # logits = self.alpha + self.beta * self.covariates - torch.cdist(self.Z[i], self.Z[j]) # Logodds for each edge
        return logits
    
    def visualize(self):
        vis_data = self.Z.detach().numpy()
        for i in range(10):
            grouped = vis_data[self.data_labels == i]
            x,y = grouped[:,0], grouped[:, 1]
            plt.scatter(x,y)


loader = DataLoader()
constructor = GraphConstructor()
X, y = loader.load_MNIST(subset_percent=0.01)
adjacency_matrix = constructor.construct_knn(X, k_neighbors=100)
ldm = LatentDistanceModel(adjacency_matrix=adjacency_matrix, data_labels=y)

eye_mask = torch.eye(ldm.N, dtype=torch.bool, device=ldm.Z.device)
m = nn.Sigmoid()
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ldm.N / 100))
optimizer = torch.optim.Adam(ldm.parameters(), lr=1e-3)
for i in range(3000):
    optimizer.zero_grad()
    
    # Get raw log-odds (no sigmoid!)
    logits = ldm.forward()
    
    # 6. Apply mask: We only calculate loss for non-diagonal elements
    logits_masked = logits[~eye_mask]
    targets_masked = ldm.adjacency_matrix[~eye_mask]
    
    loss = loss_fn(logits_masked, targets_masked)
    
    if i % 100 == 0:
        print(f"Step {i}, Loss: {loss.item():.4f}, Alpha: {ldm.alpha.item():.4f}")
    
    loss.backward()
    optimizer.step()

ldm.visualize()





