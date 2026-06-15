
# %%
import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA


# %%
#https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
class LatentDistanceModel(nn.Module):
    def __init__(
        self,
        data: torch.Tensor,
        data_labels: np.ndarray = None,
        output_dimension: int = 2,
        initialization: str = "random",
    ) -> None:
        super().__init__()

        # Alle datapunkter har en 2 dimensional position, og en node på grafen.
        self.num_nodes = data.shape[0]
        self.data_labels = data_labels

        # Embedding lag, har alle N datapunker, hvert med en position z i output_dimension space.
        self.Z = nn.Embedding(
            num_embeddings=self.num_nodes, embedding_dim = output_dimension
        )

        # Initialiserer dem tilfældigt
        nn.init.normal_(self.Z.weight, mean=0.0, std=0.1)

        # Optionally initialize embeddings using sklearn PCA (simple, minimal checks)
        if initialization is not None and initialization.lower() == "pca":
            X = data.cpu().numpy()
            pca = PCA(n_components=output_dimension)
            Z_init = pca.fit_transform(X)
            coords_t = torch.tensor(Z_init, dtype=torch.float32, device=self.Z.weight.device)
            with torch.no_grad():
                self.Z.weight.copy_(coords_t)

        # Alpha som også skal læres
        # self.alpha = nn.Parameter(torch.tensor(0.0))
        self.alpha = nn.Parameter(torch.zeros(self.num_nodes))

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        # Embedded Space Distance Matricen, c_ij = |z_i2 - z_j|

        # Finder afstande fra alle nodes, til de udvalgte edges fra @sample_edges
        # Udregner afstande. Mere effektivt end N x N
        weights = self.Z.weight
        base_points = weights.unsqueeze(1)
        target_points = weights[edges]
        sampled_embedded_dist = torch.linalg.norm(base_points - target_points, dim=-1) # N x (k + k*neg_ratio)

        # Logits er da bare alpha - |z_i - z_j| for alle z, og alle deres edges.
        logits = (self.alpha.unsqueeze(1) + self.alpha[edges]) - sampled_embedded_dist
        
        # Ideelt set er denne tilsvarende til target_probs fra @sample_edges?
        return logits

    def visualize(self, datatype: Literal["default", "mnist", "swissroll"] = "default", save_path: str | None = None, show: bool = True) -> None:
        # Funktion til blot at visualiserer og gemme visualisering, til kvalitativ analyse.
        vis_data = self.Z.weight.detach().cpu().numpy()
        plt.figure(figsize=(8, 8))
        plt.title("Latent Space Visualization")
        match datatype:
            case "default":
                plt.scatter(vis_data[:, 0], vis_data[:, 1], alpha=0.6, edgecolors="w")
                
            case "mnist":
                for i in range(10):
                    grouped = vis_data[self.data_labels == i]
                    x, y = grouped[:, 0], grouped[:, 1]
                    plt.scatter(x, y, label=str(i), alpha=0.6, edgecolors="w")
                    plt.legend()
    
            case "swissroll" | "s_hole":
                assert self.data_labels is not None
                plt.scatter(vis_data[:, 0], vis_data[:, 1], c=self.data_labels, edgecolors="w")


        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()
