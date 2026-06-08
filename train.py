# %%
import time
from pathlib import Path

import torch
import torch.nn as nn

from LatentDistanceModel import LatentDistanceModel
from utils import (
    get_initial_parameters,
    load_MNIST,
    load_s_hole,
    load_swissroll,
    sample_edges,
)

# %% HYPERPARAMETER
k = 60  # K-nearest-neighbors
lr = 0.05
weight_decay = 1e-4
negative_ratio = 10
samples = 10_000
epochs = 150
save_timeline = True
sigma_q = 10
dataset = "s_hole"
# %% MODEL
t_edge = t_forward = t_backward = t_step = 0.0
t0_total = time.perf_counter()


match dataset:
    case "swissroll":
        X, y = load_swissroll(n_samples=samples)
    case "s_hole":
        X, y = load_s_hole(n_samples=samples)


t0 = time.perf_counter()
knn_indices, sigma = get_initial_parameters(X, k_neighbors=k, sigma_q=sigma_q)
t_alldistances = time.perf_counter() - t0

ldm = LatentDistanceModel(
    data = X,
    output_dimension=2,
    data_labels=y,
)

optimizer = torch.optim.Adam(ldm.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = nn.BCEWithLogitsLoss()
if save_timeline:
    timeline_dir = Path(f"timelines/{dataset}")
    # timeline_dir.mkdir(parents=True, exist_ok=True)
    model_folder = timeline_dir / f"{k=}|{samples=}|{negative_ratio=}|{sigma_q=}|{lr=}|{weight_decay=}|{epochs=}"
    model_folder.mkdir(parents=True, exist_ok=True)



# %% TRAINING
for epoch in range(epochs+1):
    optimizer.zero_grad()

    t0 = time.perf_counter()
    edges, edge_distances, targets = sample_edges(X = X, knn_indices=knn_indices, negative_ratio=negative_ratio, sigma=sigma)
    t_edge += time.perf_counter() - t0

    t0 = time.perf_counter()
    logits = ldm(edges)
    loss = loss_fn(logits, targets)
    t_forward += time.perf_counter() - t0

    t0 = time.perf_counter()
    loss.backward()
    t_backward += time.perf_counter() - t0

    t0 = time.perf_counter()
    optimizer.step()
    t_step += time.perf_counter() - t0

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
            # f"Alpha: {ldm.alpha.item():.4f} | ",
        )
        if save_timeline:
            save_path = model_folder / f"{epoch=}|t={(time.perf_counter() - t0_total):.3f}s.png"
            ldm.visualize(datatype=dataset, show=False, save_path=save_path)



# %% Information
total_time = time.perf_counter() - t0_total
diff_time = total_time - (t_alldistances + t_edge + t_forward + t_backward + t_step)


lines = []
lines.append(f"--- Time breakdown {dataset} ---")
lines.append(f"Hyperparams: {k=}, {negative_ratio=}, {samples=}, {epochs=}")
lines.append(f"Total Time:  {total_time:.3f}s")
lines.append(f"edge sampling:  {t_edge:.3f}s  ({t_edge / total_time * 100:.1f}%)")
lines.append(f"forward pass:   {t_forward:.3f}s  ({t_forward / total_time * 100:.1f}%)")
lines.append(f"backward pass:  {t_backward:.3f}s  ({t_backward / total_time * 100:.1f}%)")
lines.append(f"optimizer step: {t_step:.3f}s  ({t_step / total_time * 100:.1f}%)")
lines.append(f"Dense Distance Matrix(NxN): {t_alldistances:.3f}s  ({t_alldistances / total_time * 100:.1f}%)")
lines.append(f"Other: {diff_time:.3f}s  ({diff_time / total_time * 100:.1f}%)")
print(*lines, sep='\n')

if save_timeline:
    timing_path = model_folder / "timings.txt"
    with timing_path.open("w") as f:
        f.write("\n".join(lines))

