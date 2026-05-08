
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
        data_labels: np.ndarray = None,
        output_dimension: int = 2,
    ) -> None:
        
        super().__init__()
        self.adjacency_matrix = adjacency_matrix.to(torch.float32)
        self.num_nodes = adjacency_matrix.shape[0]
        self.data_labels = data_labels

        # Embedding lag, har alle N datapunker, hvert med en position i output_dimension space.
        self.Z = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=output_dimension)
        
        # Initialiserer dem tilfældigt
        nn.init.normal_(self.Z.weight, mean=0.0, std=0.1)

        # Alpha som også skal læres
        self.alpha = nn.Parameter(torch.tensor(0.0))

    # def forward(self, sender_indices, receiver_indices) -> torch.Tensor:
        
    #     z_i = self.Z(sender_indices)
    #     z_j = self.Z(receiver_indices)

    #     # Udregner afstanden
    #     distances = torch.norm(z_i - z_j, p=2, dim=1) + 1e-6

    #     # Udregner log odds
    #     logits = self.alpha - distances
    #     return logits
    
    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        embedded_dist = torch.cdist(self.Z.weight, self.Z.weight)
        sampled_embedded_dist = torch.gather(embedded_dist, dim=1, index=edges)
        logits = self.alpha - sampled_embedded_dist
        return logits

    def visualize(self, save_path: str | None = None, show: bool = True) -> None:
        # Funktion til blot at visualiserer og gemme visualisering, til kvalitativ analyse.
        vis_data = self.Z.weight.detach().cpu().numpy()

        plt.figure(figsize=(8, 8))
        if self.data_labels is not None:
            for i in range(10):
                grouped = vis_data[self.data_labels == i]
                x, y = grouped[:, 0], grouped[:, 1]
                plt.scatter(x, y, label=str(i), alpha=0.6, edgecolors="w")
        else:
            plt.scatter(vis_data[:, 0], vis_data[:, 1], alpha=0.6, edgecolors="w")


        plt.title("Latent Space Visualization")
        plt.legend()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        if show:
            plt.show()