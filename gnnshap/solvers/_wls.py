from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from gnnshap.utils import get_logger

from ._base import BaseSolver

log = get_logger(__name__)


class WLSSolver(BaseSolver):
    """Solver that uses pytorch to solve the weighted least squares problem."""
    def __init__(self, mask_matrix: Tensor, kernel_weights: Tensor, yhat: Tensor,
                 fnull: Tensor, ffull: Tensor, **kwargs: dict) -> None:
        """Initialization for WLSSolver.

        Args:
            mask_matrix (Tensor): mask matrix
            kernel_weights (Tensor): kernel weights
            yhat (Tensor): model predictions
            fnull (float): null model prediction
            ffull (float): full model prediction
            **kwargs (dict): additional arguments
        """
        super().__init__(mask_matrix, kernel_weights, yhat, fnull, ffull, **kwargs)
        self.device = kwargs.get('device', 'cpu')

        self.mask_matrix = self.mask_matrix.to(self.device).double()
        self.kernel_weights = self.kernel_weights.to(self.device)
        self.yhat = self.yhat.to(self.device)
   
    def solve(self) -> Tuple[np.array, dict]:
        r"""Solves weighted least squares problem and learns shapley values

        Args:
            mask_matrix (Tensor): coalition matrix
            kernel_weights (Tensor): coalition weight values
            ey (Tensor): coalition predictions

        Returns:
            np.array: shapley_values
        """


        # no need to add base value as player thanks to this: (base + shap_values) = ffull
        eyAdj = self.yhat - self.fnull

        # eliminate one variable with the constraint that all features sum to the output
        eyAdj2 = eyAdj - self.mask_matrix[:, -1] * (self.ffull - self.fnull)
        etmp = self.mask_matrix[:, :-1] - self.mask_matrix[:, -1].unsqueeze(1)


        # solve a weighted least squares equation to estimate phi
        tmp_transpose = (etmp * self.kernel_weights.unsqueeze(1)).transpose(0, 1)
        
        etmp_dot = torch.mm(tmp_transpose, etmp)
        try:
            tmp2 = torch.linalg.inv(etmp_dot)
        except torch.linalg.LinAlgError:
            tmp2 = torch.linalg.pinv(etmp_dot)
            print("Equation is singular, using pseudo-inverse.",
                  "Consider increasing the number of samples.")
        w = torch.mm(tmp2, torch.mm(tmp_transpose, eyAdj2.unsqueeze(1)))[:,0].cpu()

        phi = torch.zeros(self.mask_matrix.size(1))
        phi[:-1] = w
        phi[-1] = (self.ffull - self.fnull) - torch.sum(w)

        # clean up any rounding errors
        #phi[torch.abs(phi) < 1e-10] = 0


        return phi.numpy()
