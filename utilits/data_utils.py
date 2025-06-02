import numpy as np
from torch.utils.data import DataLoader

def dataset_to_numpy(dataset, batch_size=1):
    """
    Convert a PyTorch Dataset to NumPy arrays.

    Args:
        dataset (torch.utils.data.Dataset): PyTorch dataset object.
        batch_size (int): Batch size for loading data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature array X and target array y.
    """
    X, y = [], []
    loader = DataLoader(dataset, batch_size=batch_size)
    for x_, y_ in loader:
        X.append(x_.squeeze(0).numpy())
        y.append(y_.item())
    return np.array(X), np.array(y)
