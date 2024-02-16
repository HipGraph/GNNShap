from typing import Optional

from ._base import BaseSampler
from ._exact import SHAPExactSampler
from ._gnnshap import GNNShapSampler
from ._svx import SVXSampler
from ._shap import SHAPSampler
from ._shap_unique import SHAPUniqueSampler


def get_sampler(nplayers: int, nsamples: int = None, sampler_name: str = "shap_sampler",
                **kwargs: Optional) -> BaseSampler:
    """Returns the instanciated sampler based on the name. 

    Args:
        nplayers (int): number of players
        nsamples (int, optional): number of samples. Defaults to None.
        sampler_name (str, optional): sampler name. Defaults to "shap_sampler".
        kwargs (optional): extra arguments if sampler needs it.

    Raises:
        KeyError: Raises error if sampler doesnot exist.

    Returns:
        BaseSampler: A sampler class instance.
    """
    samplers = {
        'SHAPSampler': SHAPSampler,
        'SHAPExactSampler': SHAPExactSampler,
        'SHAPUniqueSampler': SHAPUniqueSampler,
        'GNNShapSampler': GNNShapSampler,
        'SVXSampler': SVXSampler
    }

    try:
        return samplers[sampler_name](nplayers=nplayers, nsamples=nsamples, **kwargs)
    except KeyError:
        raise KeyError(f"Sampler '{sampler_name}' not found!")
