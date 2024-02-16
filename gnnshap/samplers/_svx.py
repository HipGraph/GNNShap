import itertools
import random
import time
from copy import deepcopy
from itertools import combinations
from typing import Tuple

import numpy as np
import scipy.special
import torch
from torch import Tensor

from gnnshap.utils import get_logger

from ._base import BaseSampler

log = get_logger(__name__)

class SVXSampler(BaseSampler):
    """ SVXSampler is based on GraphSVX’s "SmarterSeparate" sampling method. 
    Source: https://raw.githubusercontent.com/AlexDuvalinho/GraphSVX/master/src/explainers.py
    """

    def __init__(self, nplayers: int, nsamples: int, **kwargs) -> None:
        """number of players and number of samples are required.

        Args:
            nplayers (int): number of players
            nsamples (int): number of samples
            size_lim (int): maximum size of coalitions to sample from. Defaults to 3.
        """
        super().__init__(nplayers=nplayers, nsamples=nsamples)
        self.size_lim = kwargs.get('size_lim', 3)
    
    def shapley_kernel(self, s, M):
        """ Computes a weight for each newly created sample 

        Args:
            s (tensor): contains dimension of z for all instances
                (number of features + neighbours included)
            M (tensor): total number of features/nodes in dataset

        Returns:
                [tensor]: shapley kernel value for each sample
        """
        shapley_kernel = []

        for i in range(s.shape[0]):
            a = s[i].item()
            if a == 0 or a == M:
                # Enforce high weight on full/empty coalitions
                shapley_kernel.append(1000)
            elif scipy.special.binom(M, a) == float('+inf'):
                # Treat specific case - impossible computation
                shapley_kernel.append(1/ (M**2))
            else:
                shapley_kernel.append(
                    (M-1)/(scipy.special.binom(M, a)*a*(M-a)))

        shapley_kernel = np.array(shapley_kernel)
        shapley_kernel = np.where(shapley_kernel<1.0e-40, 1.0e-40,shapley_kernel)
        return torch.tensor(shapley_kernel)
    
    def smarter_separate(self):
        num_samples = self.nsamples
        M = self.nplayers
        args_K = self.size_lim
        z_ = torch.ones(num_samples, M)
        z_[1::2] = torch.zeros(num_samples//2, M)
        i = 0 # modified by sakkas. We don't need empty and full coalitions
        k = 1
        # Loop until all samples are created
        while i < num_samples:
            # Look at each feat/nei individually if have enough sample
            # Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
            if i + 2 * M < num_samples and k == 1:
                z_[i:i+M, :] = torch.ones(M, M)
                z_[i:i+M, :].fill_diagonal_(0)
                z_[i+M:i+2*M, :] = torch.zeros(M, M)
                z_[i+M:i+2*M, :].fill_diagonal_(1)
                i += 2 * M
                k += 1

            else:
                # Split in two number of remaining samples
                # Half for specific coalitions with low k and rest random samples
                #samp = i + 9*(num_samples - i)//10
                samp = num_samples
                while i < samp and k <= min(args_K, M):
                    # Sample coalitions of k1 neighbours or k1 features without repet and order.
                    L = list(combinations(range(0, M), k))
                    random.shuffle(L)
                    L = L[:samp+1]

                    for j in range(len(L)):
                        # Coalitions (All nei, All-k feat) or (All feat, All-k nei)
                        z_[i, L[j]] = torch.zeros(k)
                        i += 1
                        # If limit reached, sample random coalitions
                        if i == samp:
                            z_[i:, :] = torch.empty(num_samples-i, M).random_(2)
                            return z_
                        # Coalitions (No nei, k feat) or (No feat, k nei)
                        z_[i, L[j]] = torch.ones(k)
                        i += 1
                        # If limit reached, sample random coalitions
                        if i == samp:
                            z_[i:, :] = torch.empty(num_samples-i, M).random_(2)
                            return z_
                    k += 1

                # Sample random coalitions
                z_[i:, :] = torch.empty(num_samples-i, M).random_(2)
                return z_
        return z_

    def sample(self) -> Tuple[Tensor, Tensor]:
        """Returns all possible coalitions and weights. Note that it doesn't include empty and full
        coalitions (Solver does not need them).

        Returns:
            Tuple[Tensor, Tensor]: mask_matrix and kernel weights.
        """
        z_bis = self.smarter_separate()
        # z_bis = z_bis[torch.randperm(z_bis.size()[0])] # no need to shuffle
        s = (z_bis != 0).sum(dim=1)
        weights = self.shapley_kernel(s, self.nplayers)
        
        return z_bis, weights