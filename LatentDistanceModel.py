
# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


# %%
#https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
class LatentDistanceModel(nn.Module):
    def __init__(
        self,
        adjacency_matrix: torch.Tensor,
        data_labels: np.ndarray,
        output_dimension: int = 2,
        data: torch.Tensor = None,
        # data: torch.Tensor | None = None,
    ) -> None:
        
        super().__init__()
        self.adjacency_matrix = adjacency_matrix.to(torch.float32)
        self.num_nodes = adjacency_matrix.shape[0]
        self.data_labels = data_labels
        self.device = adjacency_matrix.device


        # Embedding lag, har alle N datapunker, hvert med en position i output_dimension space.
        self.Z = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=output_dimension)

        # Til at initialiserer Z med PCA. Gør ikke rigtigt forskel, og kan skygge over andre
        # reelle forbedringer.
        # if data is not None:
        #     from sklearn.decomposition import PCA
        #     pca = PCA(n_components=output_dimension)
        #     Z_init = pca.fit_transform(data.cpu().numpy())
        #     self.Z.weight.data = torch.tensor(Z_init, dtype=torch.float32)
        # else:
        nn.init.normal_(self.Z.weight, mean=0.0, std=0.1)

        # Alpha som også skal læres
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.covariates = torch.tensor(data, device=self.device)

        # dist = torch.cdist(data, data, p=2)
        # dist = (dist - dist.min()) / (dist.max() - dist.min())
        # self.covariates = torch.tensor(dist, device=self.device, requires_grad=False)
        # self.beta = nn.Parameter(torch.zeros(len(self.covariates[0])))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, sender_indices, receiver_indices) -> torch.Tensor:
        
        z_i = self.Z(sender_indices)
        z_j = self.Z(receiver_indices)

        # Udregner afstanden
        distances = torch.norm(z_i - z_j, p=2, dim=1)

        # Udregner log odds
        # logits = self.alpha - distances
        # logits = (
        #     self.alpha
        #     + (self.covariates[sender_indices] - self.covariates[receiver_indices]) @ self.beta.T - distances
        # )
        # logits = (
        #     self.alpha + self.beta * self.covariates[sender_indices, receiver_indices] - distances
        # )
        

        logits = self.alpha - distances
        return logits

    def visualize(self, save_path: str | None = None, show: bool = True) -> None:
        # Funktion til blot at visualiserer og gemme visualisering, til kvalitativ analyse.
        vis_data = self.Z.weight.detach().cpu().numpy()

        plt.figure(figsize=(8, 8))
        for i in range(10):
            grouped = vis_data[self.data_labels == i]
            x, y = grouped[:, 0], grouped[:, 1]
            plt.scatter(x, y, label=str(i), alpha=0.6, edgecolors="w")

        plt.title("Latent Space Visualization")
        plt.legend()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        if show:
            plt.show()