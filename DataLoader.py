#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_s_curve, make_swiss_roll
from sklearn.model_selection import train_test_split
from torchvision import datasets


def load_MNIST(subset_percent: float) -> tuple[torch.Tensor]:
    # Load MNIST and turn every image into a node.
    # The 28x28 image is flattened into a 784-dimensional feature vector.
    mnist_train = datasets.MNIST(root='./data', train=True, download=True) # 60_000 x 28 x 28
    mnist_test = datasets.MNIST(root='./data', train=False, download=True) # 10_000 x 28 x 28
    
    # X_full: node-feature matrix of shape (n, d).
    # y_full: digit labels, only used later for coloring plots and checking structure.
    X_full = torch.cat([mnist_train.data, mnist_test.data], dim=0).float().view(-1, 28*28) / 255.0 # 70_000 x 728
    y_full = torch.cat([mnist_train.targets, mnist_test.targets], dim=0)
    if subset_percent == 1:
        return X_full, y_full

    # # Keep only a subset of the data, but stratify by label so the digit proportions stay similar.
    X, _, y, _ = train_test_split(
        X_full, y_full,
        train_size=subset_percent,
        stratify=y_full,           # preserves class distribution
        random_state=42,
        )

    return X, y 

def load_mammoth(subset_percent: float = 1.0) -> torch.Tensor:
    mammoth = pd.read_csv("data/mammoth.csv")
    mammoth = mammoth.to_numpy()
    if subset_percent == 1.0:
        return torch.tensor(mammoth)
    n = len(mammoth)
    indices = np.random.randint(0, n, size=round(n*subset_percent))
    return torch.tensor(mammoth[indices])


def load_swissroll(n_samples: int = 5000) -> tuple[np.ndarray]:
    X,y = make_swiss_roll(
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



