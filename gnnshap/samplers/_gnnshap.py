import copy
import os
from typing import Tuple

import numpy as np
import torch
from scipy.special import binom
from torch import Tensor

from gnnshap.utils import get_logger

from ._base import BaseSampler

log = get_logger(__name__)

from torch.utils.cpp_extension import load

cppsamp = load(name='cudaGNNShapSampler', sources=['cppextension/cudagnnshap.cu'],
                   extra_cflags=['-O2'], verbose=False)


class GNNShapSampler(BaseSampler):
    r"""This sampling algorithm is implemented in Cuda to speed up the sampling process. It 
        creates samples in parallel. The number of blocks and threads can be adjusted.
        The total weights are scaled to 100 to increase numerical stability.
        """

    def __init__(self, nplayers: int, nsamples: int, **kwargs) -> None:
        """number of players and number of samples are required.

        Args:
            nplayers (int): number of players
            nsamples (int): number of samples
            num_blocks (int, optional): number of blocks for cuda. Defaults to 16.
            num_threads (int, optional): number of threads for cuda. Defaults to 128.
        """
        super().__init__(nplayers=nplayers, nsamples=nsamples)
        self.num_blocks = kwargs.get('num_blocks', 16)
        self.num_threads = kwargs.get('num_threads', 128)

    def sample(self) -> Tuple[Tensor, Tensor]:
        mask_matrix = torch.zeros((self.nsamples, self.nplayers),
                                  dtype=torch.bool, requires_grad=False).cuda()
        kernel_weights = torch.zeros((self.nsamples), dtype=torch.float64,
                                     requires_grad=False).cuda()

        cppsamp.sample(mask_matrix, kernel_weights, self.nplayers, self.nsamples,
                       self.num_blocks, self.num_threads)
        return mask_matrix, kernel_weights
