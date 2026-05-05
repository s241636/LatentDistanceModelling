
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
# - Grafen vi laver skal vel netop være symmetrisk? Hele ideen med reciprocity er jo at der i den originale graf
# - vil en edge imellem begge punkter, hvis det er tæt på hinanden.

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
        return logits
    
    def visualize(self):
        vis_data = self.Z.detach().numpy()
        for i in range(10):
            grouped = vis_data[self.data_labels == i]
            x,y = grouped[:,0], grouped[:, 1]
            plt.scatter(x,y)


class LatentDistanceModelAI(nn.Module):
    def __init__(
        self, num_nodes: int, data_labels: np.ndarray, output_dimension: int = 2
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.data_labels = data_labels

        # 1. The Embedding Layer for Latent Positions
        # This automatically registers the coordinates as learnable parameters.
        self.Z = nn.Embedding(num_embeddings=num_nodes, embedding_dim=output_dimension)

        # We initialize the embeddings with a small spread around the origin
        nn.init.normal_(self.Z.weight, mean=0.0, std=0.1)

        # 2. Learnable Baseline Density (Alpha)
        # Defined as a pure scalar (0-dimensional tensor)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, sender_indices, receiver_indices):
        # Retrieve the specific latent vectors for the batch
        z_i = self.Z(sender_indices)
        z_j = self.Z(receiver_indices)

        # Calculate Euclidean distance between the pairs
        distances = torch.norm(z_i - z_j, p=2, dim=1)

        # Calculate log-odds
        logits = self.alpha - distances
        return logits

    def visualize(self):
        # Extract the coordinates from the embedding layer
        vis_data = self.Z.weight.detach().cpu().numpy()

        plt.figure(figsize=(8, 8))
        # Assuming MNIST labels (0-9)
        for i in range(10):
            grouped = vis_data[self.data_labels == i]
            x, y = grouped[:, 0], grouped[:, 1]
            plt.scatter(x, y, label=str(i), alpha=0.6, edgecolors="w")

        plt.title("Latent Space Visualization")
        plt.legend()
        plt.show()