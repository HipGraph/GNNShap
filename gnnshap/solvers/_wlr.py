from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from gnnshap.utils import get_logger

from ._base import BaseSolver

log = get_logger(__name__)


class WLRSolver(BaseSolver):
    """Solver that uses pytorch to solve the linear regression."""
    def __init__(self, mask_matrix: Tensor, kernel_weights: Tensor, yhat: Tensor,
                 fnull: Tensor, ffull: Tensor, **kwargs: dict) -> None:
        """Initialization for WLRSolver.

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

        # will convert to double later
        self.mask_matrix = self.mask_matrix.to(self.device).type(torch.int8)
        self.kernel_weights = self.kernel_weights.to(self.device).unsqueeze(1)
        self.yhat = self.yhat.to(self.device)
        self.nplayers = self.mask_matrix.size(1)
   
    def solve(self) -> Tuple[np.array, dict]:
        r"""Solves weighted linear regression problem by training a linear model via PyTorch.

        Args:
            mask_matrix (Tensor): coalition matrix
            kernel_weights (Tensor): coalition weight values
            ey (Tensor): coalition predictions

        Returns:
            np.array: shapley_values
        """

        # no need to add base value as player thanks to this: (base + shap_values) = ffull
        eyAdj = self.yhat - self.fnull
        del self.yhat

        # eliminate one variable with the constraint that all features sum to the output
        eyAdj2 = (eyAdj - self.mask_matrix[:, -1] * (self.ffull - self.fnull)).unsqueeze(1)
        etmp = self.mask_matrix[:, :-1] - self.mask_matrix[:, -1].unsqueeze(1)
        del self.mask_matrix


        torch.set_grad_enabled(True)

        lin_model = torch.nn.Linear(etmp.size(1), 1, dtype=torch.double, bias=False).to(self.device)
        optimizer = torch.optim.Adam(lin_model.parameters(), lr=0.001, weight_decay=0.005)


        etmp = etmp.double()
        # solve a weighted least squares equation to estimate phi
        lin_model.train()
        for i in range(200):
            optimizer.zero_grad()
            pred = lin_model(etmp)
            loss = torch.sum(self.kernel_weights * ((eyAdj2 - pred)**2))
            loss.backward()
            optimizer.step()


        phi = torch.zeros(self.nplayers)
        phi[:-1] = lin_model.weight.squeeze()
        phi[-1] = (self.ffull - self.fnull) - torch.sum(lin_model.weight)

        # clean up any rounding errors
        #phi[torch.abs(phi) < 1e-10] = 0


        return phi.detach().numpy()