import numpy as np
from scipy.spatial import cKDTree

def min_l2_distance(vectors: np.ndarray) -> float:
    """
    Minimum Euclidean distance between distinct rows, ignoring duplicates.
    Uses cKDTree for efficiency.
    """
    X = np.asarray(vectors, dtype=np.float32)
    X = np.unique(X, axis=0)           # drop duplicate rows
    if len(X) < 2:
        raise ValueError("Need at least 2 unique vectors.")

    tree = cKDTree(X)
    # k=2 -> first is the point itself (distance 0), second is nearest neighbor
    dists, _ = tree.query(X, k=2, workers=-1)
    return float(np.min(dists[:, 1]))