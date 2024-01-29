# pylint: disable=not-callable
__all__ = ["f_one_hot"]

from torch import Tensor
import torch.nn.functional as F


def f_one_hot(x: Tensor, num_classes: int = None) -> Tensor:
    """Patch."""
    if num_classes is None:
        x = F.one_hot(x)
    else:
        x = F.one_hot(x, num_classes)
    return x
