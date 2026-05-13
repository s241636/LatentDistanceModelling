import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_s_curve, make_swiss_roll
from sklearn.model_selection import train_test_split
from torchvision import datasets


def get_initial_parameters(data: torch.Tensor, k_neighbors: int) -> torch.Tensor:
    """
    Konstruerer en graf hvor hvert node vil have 'k' edges, gående til de tilhørende
    k-nearest-neighbors i latent space.

    Args:
        data (torch.Tensor): Data tensor, forventes at være reshaped til (N, -1)
        k_neighbors (int): Mængden af edges for hvert node, altså mængden af neighbors i latent space.
        undirected (bool): Om grafen er directed eller ej. I tilfælde af undirected=True kan det ikke garanteres
        at hvert node har 'k' edges, eftersom der kan blive tilføjet edges for at gør den symmetrisk. Er dog nok heller aldrig
        optimalt at have grafen undirected. Bliver da til shared nearest neighbors

    Returns:
        np.ndarray: Adjacency matrix over grafen.
    """
    # Finder de k indices med lavest euclidean distance.
    dist = torch.cdist(data, data)
    knn_indices = torch.topk(dist, k=k_neighbors + 1, largest=False).indices[:, 1:]
    sigma = np.percentile(dist, q=10)  # Meget bedre med sigma for hele datasættet

    return knn_indices, sigma

def sample_edges(
    X: torch.Tensor,
    knn_indices: np.ndarray,
    negative_ratio: float,
    sigma: float,
) -> tuple[torch.Tensor]:
    # Finder index for de k positive edges, for alle nodes.
    k = len(knn_indices[0])
    N = len(knn_indices)

    # Mængden af negative edges
    negatives_count = round(k * negative_ratio)
    pos_edges = knn_indices

    # Finder tilsvarende tilfældige "negative" edges.
    # Loop fortsætter til der er præcist k*negative ratio og at de fundne edges ikke er en del af pos_edges, eller node'n selv.
    negative_edges = torch.empty(size=(N, negatives_count))
    for idx, pos_edge in enumerate(pos_edges):
        # Tager tilfældige edges
        random_edges = torch.randint(low=0, high=N, size=(1, negatives_count))
        # Uønskede tilfældige edges (knn og noden selv)
        non_negative_edges = torch.cat((torch.tensor([idx]), pos_edge))
        # Udtager indtil de findes
        while torch.isin(random_edges, non_negative_edges).any():
            random_edges = random_edges[~torch.isin(random_edges, non_negative_edges)]
            new_random_edges = torch.randint(
                low=0, high=N, size=(1, negatives_count - len(random_edges))
            )
            random_edges = torch.cat((random_edges, new_random_edges.flatten()))
        negative_edges[idx] = random_edges

    # Samler edge indices til en tensor. Første k er da de positive edge indices, og de næste er de negative.
    edge_indices = torch.cat((pos_edges, negative_edges), dim=1).to(int)

    # Finder afstande, til de givne edges.
    base_points = X.unsqueeze(1)
    target_points = X[edge_indices]
    edge_distances = torch.linalg.norm(base_points - target_points, dim=-1)
    # Udregner til sandsynligheder
    probs = torch.exp(-(edge_distances) / (sigma))
    # probs = torch.exp(-(edge_distances**2) / (sigma**2))
    target_probabilities = torch.clip(probs, 1e-4, 0.95)
    return edge_indices, edge_distances, target_probabilities





def load_MNIST(subset_percent: float) -> tuple[torch.Tensor]:
    # Load MNIST and turn every image into a node.
    # The 28x28 image is flattened into a 784-dimensional feature vector.
    mnist_train = datasets.MNIST(
        root="./data", train=True, download=True
    )  # 60_000 x 28 x 28
    mnist_test = datasets.MNIST(
        root="./data", train=False, download=True
    )  # 10_000 x 28 x 28

    # X_full: node-feature matrix of shape (n, d).
    # y_full: digit labels, only used later for coloring plots and checking structure.
    X_full = (
        torch.cat([mnist_train.data, mnist_test.data], dim=0).float().view(-1, 28 * 28)
        / 255.0
    )  # 70_000 x 728
    y_full = torch.cat([mnist_train.targets, mnist_test.targets], dim=0)
    if subset_percent == 1:
        return X_full, y_full

    # # Keep only a subset of the data, but stratify by label so the digit proportions stay similar.
    X, _, y, _ = train_test_split(
        X_full,
        y_full,
        train_size=subset_percent,
        stratify=y_full,  # preserves class distribution
        random_state=42,
    )

    return X, y


def load_mammoth(subset_percent: float = 1.0) -> torch.Tensor:
    mammoth = pd.read_csv("data/mammoth.csv")
    mammoth = mammoth.to_numpy()
    if subset_percent == 1.0:
        return torch.tensor(mammoth)
    n = len(mammoth)
    indices = np.random.randint(0, n, size=round(n * subset_percent))
    return torch.tensor(mammoth[indices])


def load_swissroll(n_samples: int = 5000) -> tuple[np.ndarray]:
    X, y = make_swiss_roll(
        n_samples=n_samples,
        noise=0.03,
        random_state=0,
        hole=False,
    )
    return torch.tensor(X), torch.tensor(y)


def load_s_hole(n_samples=10_000, random_state=20200202) -> tuple[torch.Tensor]:
    X, t = make_s_curve(n_samples=n_samples, random_state=random_state)

    # Gør hullet tydeligere?
    anchor = np.array([0, 1, 0])
    keep = np.sum((X - anchor) ** 2, axis=1) > 0.3

    X = X[keep]
    t = t[keep]

    return torch.tensor(X), torch.tensor(t)
