from torch import Tensor

from ._base import BaseSolver
from ._wlr import WLRSolver
from ._wls import WLSSolver


def get_solver(solver_name: str, mask_matrix: Tensor, kernel_weights: Tensor, yhat: Tensor,
               fnull: float, ffull: float, **kwargs: dict) -> BaseSolver:
    """Returns the instanciated solver based on the name.

    Args:
        solver_name (str): Solver name
        mask_matrix (Tensor): mask matrix
        kernel_weights (Tensor): kernel weights
        yhat (Tensor): model predictions
        fnull (float): null model prediction
        ffull (float): full model prediction

    Raises:
        KeyError: If solver name is not found

    Returns:
        BaseSolver: Instanciated solver
    """
    solvers = {
        'WLSSolver': WLSSolver,
        'WLRSolver': WLRSolver
    }

    try:
        return solvers[solver_name](mask_matrix, kernel_weights, yhat, fnull, ffull, **kwargs)
    except KeyError as exc:
        raise KeyError(f"Solver '{solver_name}' not found!") from exc
