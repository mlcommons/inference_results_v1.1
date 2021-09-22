from typing import Optional

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset


def random_subset(dataset: Dataset, subset_size: int, seed: Optional[int] = None) -> Dataset:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), subset_size, replace=False)
    return Subset(dataset, indices)
