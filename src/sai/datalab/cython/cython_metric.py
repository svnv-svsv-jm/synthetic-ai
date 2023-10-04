from numpy import ndarray
import numpy as np
import cython


@cython.boundscheck(False)
@cython.wraparound(False)
def mixed_distance(
    x: ndarray,
    y: ndarray,
    categoric_slice: int,
) -> float:
    """_summary_
    Args:
        x (ndarray): _description_
        y (ndarray): _description_
        categoric_slice (int): _description_
    Returns:
        float: _description_
    """
    n = x.shape[0]
    i = 0
    res = 0.0
    for i in range(categoric_slice):
        res += abs(x[i] != y[i])
    for i in range(categoric_slice, n):
        res += abs(x[i] - y[i])
    return res
