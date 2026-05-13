
# %%
import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


# %%
#https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html


class LatentDistanceModel(nn.Module):
    def __init__(
        self,
        data: torch.Tensor,
        data_labels: np.ndarray = None,
        output_dimension: int = 2,
    ) -> None:
        super().__init__()

        # Alle datapunkter har en 2 dimensional position, og en node på grafen.
        self.num_nodes = data.shape[0]
        self.data_labels = data_labels

        # Embedding lag, har alle N datapunker, hvert med en position z i output_dimension space.
        self.Z = nn.Embedding(
            num_embeddings=self.num_nodes, embedding_dim=output_dimension
        )

        # Initialiserer dem tilfældigt
        nn.init.normal_(self.Z.weight, mean=0.0, std=0.1)

        # Alpha som også skal læres
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        # Embedded Space Distance Matricen, c_ij = |z_i - z_j|

        embedded_dist = torch.cdist(self.Z.weight, self.Z.weight)

        # Finder afstande fra alle nodes, til de udvalgte edges fra @sample_edges
        sampled_embedded_dist = torch.gather(embedded_dist, dim=1, index=edges)

        # Logits er da bare alpha - |z_i - z_j| for alle z, og alle deres edges.
        logits = self.alpha - sampled_embedded_dist
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
    
            case "swissroll":
                assert self.data_labels is not None
                plt.scatter(vis_data[:, 0], vis_data[:, 1], c=self.data_labels, edgecolors="w")


        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()
