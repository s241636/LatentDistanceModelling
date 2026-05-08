import os
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision import datasets

# Tankegang
# Data: N x D
# - 70.000 x 784 (MNIST)
# Latent Distance Matrix: N x N
# - Pairwise distance for alle vektorer i latent space.

# Sampling:
# - KNN: K tætteste distances for hvert vektor (Konstant for alle vektorer, baseret på originalt latent space)
# - Tilfældige: K * negative_ratio for hvert vektor (Samples nye hver epoch)
# - EdgeCount = K + K * negative_ratio
# - Til hvert række i latent distance matricen udtages disse edges.
# - De omdannes til sandsynligheder.
# - Ender med en N x EdgeCount matrice. De udregnede sandsynligheder imellem alle vektorer og deres edges
# - Netop denne matrice vi prøver at approksimerer kun med 2 dimensioner.

# Embedded Space: N x 2
# - 70.000 x 2 (MNIST), alle digits repræsenteret i 2 dimensioner
# - Embedded Distance Matric: N x N
# - Matrice med afstanden imellem hvert node, i embedding space.

# Træning:
# - Sampler (K + K * negative_ratio) edges
# - Udtager disse edges fra embedded distance matricen
# - Denne matrice sammenlignes med den tilsvarende latent space (sandsynligheds) matrice
# - Loss findes, og Z opdateres til nærmere at tilnærme sig latent space matricen.

# Hele ideen er altså at vi både i latent space og i embedding space har en N x N distance matrice
# Der er dog kun 2 dimensioner i embedding space, men vi ønsker den samme afstand imellem alle punkter, som i latent space matricen.

def load_MNIST(subset_percent: float) -> tuple[torch.Tensor]:
    mnist_train = datasets.MNIST(root='./data', train=True, download=True) # 60_000 x 28 x 28
    mnist_test = datasets.MNIST(root='./data', train=False, download=True) # 10_000 x 28 x 28
    X_full = torch.cat([mnist_train.data, mnist_test.data], dim=0).float().view(-1, 28*28) / 255.0 # 70_000 x 728
    y_full = torch.cat([mnist_train.targets, mnist_test.targets], dim=0)
    if subset_percent == 1:
        return X_full, y_full
    X, _, y, _ = train_test_split(
        X_full, y_full,
        train_size=subset_percent,
        stratify=y_full,
        random_state=50,
        )
    return X, y 



class LatentDistanceModel(nn.Module):
    def __init__(self, data: torch.Tensor, data_labels: np.ndarray = None, output_dimension: int = 2) -> None:
        super().__init__()

        # Alle datapunkter har en 2 dimensional position, og en node på grafen.
        self.num_nodes = data.shape[0]
        self.data_labels = data_labels

        # Embedding lag, har alle N datapunker, hvert med en position z i output_dimension space.
        self.Z = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=output_dimension)
        
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
            plt.savefig(save_path)
        if show:
            plt.show()




def sample_edges(distances: torch.Tensor, k: int, negative_ratio: float) -> tuple[torch.Tensor]:
    # Mængden af negative edges
    negatives_count = round(k * negative_ratio)
    
    # Finder index for de k positive edges, for alle nodes.
    pos_edges = distances.topk(k = k+1, largest=False).indices[:, 1:]
    N = len(distances)

    # Finder tilsvarende tilfældige "negative" edges. 
    # Loop fortsætter til der er præcist k*negative ratio og at de fundne edges ikke er en del af pos_edges, eller node'n selv.
    negative_edges = torch.empty(size=(N, negatives_count))
    for idx, pos_edge in enumerate(pos_edges):
        # Tager tilfældige edges
        random_edges = torch.randint(low = 0, high = N, size = (1, negatives_count))
        # Uønskede tilfældige edges (knn og noden selv)
        non_negative_edges = torch.cat((torch.tensor([idx]), pos_edge))
        # Udtager indtil de findes
        while torch.isin(random_edges, non_negative_edges).any():
            random_edges = random_edges[~torch.isin(random_edges, non_negative_edges)]
            new_random_edges = torch.randint(low=0, high=N, size=(1, negatives_count - len(random_edges)))
            random_edges = torch.cat((random_edges, new_random_edges.flatten()))
        negative_edges[idx] = random_edges

    # Samler edge indices til en tensor. Første k er da de positive edge indices, og de næste er de negative.
    edge_indices = torch.cat((pos_edges, negative_edges), dim=1).to(int)
    
    # Latent Space Distance Matricen, omdannes til sandsynligheder for de givne edges.
    edge_distances = torch.gather(distances, dim=1, index=edge_indices)
    
    # Omdanner til sandsynligheder
    sigma = np.percentile(distances, q=10) # Skal sigma findes for alle distancer, eller blot de samplede distancer?
    probs = torch.exp(-(edge_distances**2) / (sigma**2))
    target_probabilities = torch.clip(probs, 1e-4, 0.95)
    return edge_indices, edge_distances, target_probabilities


# Hyperparametre
subset_percent = 0.1
k = 5  # K-nearest-neighbors
lr = 0.05
weight_decay = 1e-4
negative_ratio = 15

# Loader MNIST
X, y = load_MNIST(subset_percent=subset_percent)
distances = torch.cdist(X,X)

# Initialiserer model
ldm = LatentDistanceModel(data = X,output_dimension=2, data_labels=y)
optimizer = torch.optim.Adam(ldm.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = nn.BCEWithLogitsLoss()

# Træner
epochs = 200
for epoch in range(epochs):

    optimizer.zero_grad()
    edge_indices, edge_distances, targets = sample_edges(distances=distances, k=k, negative_ratio=negative_ratio)

    # LDM'en tager jo blot edge indices, og finder resten selv.
    logits = ldm(edge_indices)
    loss = loss_fn(logits, targets)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
            f"Alpha: {ldm.alpha.item():.4f} | ",
        )
ldm.visualize(save_path="oskar.png")
