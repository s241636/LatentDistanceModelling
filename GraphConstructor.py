import numpy as np
import torch


class GraphConstructor:
    # TODO Omdan så alt kører som tensors.
    def construct_random(self, data: torch.Tensor, n_neighbors: int) -> np.ndarray:
        """
        Konstruerer en tilfældig graf, hvor hvert node har 'n_neighbor' edges, til en tilfældig anden node.
        Antages grafen er directed(nemmere fordi man ikke behøver sikre symmetri).
        

        Args:
            data (torch.Tensor): Data tensor, forventes at være reshaped til (N, -1)
            n_neighbors (int): Mængden af naboer per node.

        Returns:
            np.ndarray: Adjacency matrix over den tilfældige graf.
        """
        node_count = data.shape[0]
        adjacency_matrix = np.zeros((node_count, node_count))
        rng = np.random.default_rng()
        for i in range(node_count):
            choices = np.delete(np.arange(node_count), i) # Fjerner noden selv fra muligheder af naboer
            neighbors = rng.choice(choices, size=n_neighbors, replace=False)
            adjacency_matrix[i][neighbors] = 1
            
        return adjacency_matrix
        
    def construct_knn(self, data: torch.Tensor, k_neighbors: int, undirected: bool = False) -> torch.Tensor:
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
        node_count = data.shape[0]
        
        adjacency_matrix = torch.zeros(node_count, node_count, dtype=torch.uint8)

        # Finder de k indices med lavest euclidean distance.
        dist = torch.cdist(data, data)
        knn_indices = torch.topk(dist, k=k_neighbors+1, largest=False).indices[:, 1:]

        for i in range(node_count):
            adjacency_matrix[i, knn_indices[i]] = 1
        if undirected:
            adjacency_matrix = torch.max(adjacency_matrix, adjacency_matrix.T)

        return adjacency_matrix
