import itertools
from collections.abc import Iterable
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
import scipy

from gnnshap.utils import get_logger

from ._base import BaseSampler

log = get_logger(__name__)

class SHAPExactSampler(BaseSampler):
    """Brute Force Kernel SHAP sampler from: 
    `SHAP package. <https://shap-lrjball.readthedocs.io/en/latest/example_notebooks/kernel_explainer/Simple%20Kernel%20SHAP.html>`__
    It samples all :math:`2^{N}` possible coalitions. Note that it does not include 
    empty (no players) and full (all players) coalitions. In addition, the weights are normalized 
    to match Shapley formula. It is not suggested to use this sampler since it is not practical.
    """

    def __init__(self, nplayers: int, **kwargs) -> None:
        """Only requires number of players

        Args:
            nplayers (int): number of players

        Raises:
            AssertionError: Raises error if there are more than 30 players since it is 
                not practical.
        """
        if nplayers > 30:
            raise AssertionError("It is not possible to iterate all possible coalitions"
                                 f" when there are more than 30 players: 2^30={2 ** 30}")
        super().__init__(nplayers=nplayers, nsamples=2 ** nplayers)

    def shapley_kernel(self, s):
        """
        Computes coalition weight
        :param M: total number of players
        :param s: number of players in the coalition
        :return: coalition weight
        """
        M = self.nplayers
        # return a large number for empty and full coalition
        if s == 0 or s == M:
            return 10000
        if scipy.special.binom(M, s) == float('+inf'):
            return 0
        return (M - 1) / (scipy.special.binom(M, s) * s * (M - s))

    def _powerset(self, iterable: Iterable) -> itertools.chain:
        """Generates and returns powerset.

        Args:
            iterable (Iterable): an iterable object. Example: range(10)

        Returns:
            itertools.chain: a chain object.
        """
        coal_size = list(iterable)
        return itertools.chain.from_iterable(
            itertools.combinations(coal_size, r) for r in range(len(coal_size) + 1))

    def sample(self) -> Tuple[Tensor, Tensor]:
        """Returns all possible coalitions and weights. Note that it doesn't include empty and full
        coalitions (Solver does not need them).

        Returns:
            Tuple[Tensor, Tensor]: mask_matrix and kernel weights.
        """
        mask_matrix = np.zeros((2 ** self.nplayers, self.nplayers))
        weights = np.zeros(2 ** self.nplayers)

        # exact kernel weights
        p_w = np.array([self.shapley_kernel(s) for s in range(0, self.nplayers + 1)])

        for i, coal in enumerate(self._powerset(range(self.nplayers))):
            coal = list(coal)
            mask_matrix[i, coal] = 1
            weights[i] = p_w[len(coal)]  # shapley_kernel(M, len(s))

        return (torch.tensor(mask_matrix[1:-1], requires_grad=False).bool(),
                torch.tensor(weights[1:-1], requires_grad=False))
