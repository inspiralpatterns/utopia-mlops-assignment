import numpy as np


def pad(x: np.ndarray, n: int) -> np.ndarray:
    return np.concatenate((x, np.zeros(n)), axis=0)
