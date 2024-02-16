from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor
import numpy as np

from gnnshap.utils import get_logger

log = get_logger(__name__)


class BaseSolver(ABC):
    """A base class for solvers
    """

    def __init__(self, mask_matrix: Tensor, kernel_weights: Tensor,
                  yhat: Tensor, fnull: Tensor, ffull: Tensor, **kwargs: dict) -> None:
        """Common initialization for all solvers.

        Args:
            mask_matrix (Tensor): mask matrix
            kernel_weights (Tensor): kernel weights
            yhat (Tensor): model predictions
            fnull (Tensor): null model prediction
            ffull (Tensor): full model prediction
            **kwargs (dict): additional arguments
        """
        self.mask_matrix = mask_matrix
        self.kernel_weights = kernel_weights
        self.yhat = yhat
        self.fnull = fnull
        self.ffull = ffull
        self.kwargs = kwargs

    @abstractmethod
    def solve(self) -> Tuple[np.array, dict]:
        """An abstract method that all solvers should override.

        Returns:
            Tuple[np.array, dict]: shapley values and solver statistics dictionary.
        """
