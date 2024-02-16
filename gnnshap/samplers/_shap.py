import copy
import itertools
from typing import Tuple

import numpy as np
import torch
from scipy.special import binom
from torch import Tensor

from gnnshap.utils import get_logger

from ._base import BaseSampler

log = get_logger(__name__)


class SHAPSampler(BaseSampler):
    r"""`"This sampling algorithm is a modified version of KernelSHAP from:" 
        <https://github.com/slundberg/shap/blob/master/shap/explainers/_kernel.py>`. This one skips
        uniqueness check. It is faster than the original one.
        """

    def __init__(self, nplayers: int, nsamples: int, **kwargs) -> None:
        """number of players and number of samples are required.

        Args:
            nplayers (int): number of players
            nsamples (int): number of samples
        """
        super().__init__(nplayers=nplayers, nsamples=nsamples)

    def sample(self) -> Tuple[Tensor, Tensor]:
        r"""Returns sampled coalitions and weights. Note that it doesn't include empty and full
        coalitions(Solver does not need them). It's identical to ShapSampler except sampled 
        coalitions uniqueness check is skipped. Please refer to the ShapSampler for the details.

        Returns:
            Tuple[Tensor, Tensor]:  mask_matrix (boolean) and weights (float)
        """
        mask_matrix = np.zeros((self.nsamples, self.nplayers))
        kernel_weights = np.zeros(self.nsamples)
        nsamples_added = 0

        def addsample(m, w, n_samples_added):
            mask_matrix[n_samples_added, :] = m
            kernel_weights[n_samples_added] = w

        # weight the different subset sizes
        num_subset_sizes = int(np.ceil((self.nplayers - 1) / 2.0))

        # coalition size in the middle not a paired subset
        # if nplayers=4, 1 and 3 are pairs, 2 doesnt have a pair
        num_paired_subset_sizes = int(np.floor((self.nplayers - 1) / 2.0))

        weight_vector = np.array([(self.nplayers - 1.0) / (i * (
            self.nplayers - i)) for i in range(1, num_subset_sizes + 1)])
        weight_vector[:num_paired_subset_sizes] *= 2
        weight_vector /= np.sum(weight_vector)

        # fill out all the subset sizes we can completely enumerate
        # given nsamples*remaining_weight_vector[subset_size]
        num_full_subsets = 0
        num_samples_left = self.nsamples
        # no grouping in edge based shap
        group_inds = np.arange(self.nplayers, dtype='int64')
        mask = np.zeros(self.nplayers)
        remaining_weight_vector = copy.copy(weight_vector)

        for subset_size in range(1, num_subset_sizes + 1):

            # determine how many subsets (and their complements) are of the current size
            nsubsets = binom(self.nplayers, subset_size)
            if subset_size <= num_paired_subset_sizes:
                nsubsets *= 2

            # see if we have enough samples to enumerate all subsets of this size
            if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:
                num_full_subsets += 1
                num_samples_left -= nsubsets

                # rescale what's left of the remaining weight vector to sum to 1
                # it works as like not used samples distributed to other bins.
                if remaining_weight_vector[subset_size - 1] < 1.0:
                    remaining_weight_vector /= (1 -
                                                remaining_weight_vector[subset_size - 1])

                # add all the samples of the current subset size
                w = weight_vector[subset_size - 1] / \
                    binom(self.nplayers, subset_size)
                if subset_size <= num_paired_subset_sizes:
                    w /= 2.0
                for inds in itertools.combinations(group_inds, subset_size):
                    mask[:] = 0.0
                    mask[np.array(inds, dtype='int64')] = 1.0
                    addsample(mask, w, nsamples_added)
                    nsamples_added += 1

                    if subset_size <= num_paired_subset_sizes:
                        mask[:] = np.abs(mask - 1)
                        addsample(mask, w, nsamples_added)
                        nsamples_added += 1
            else:
                break

        # add random samples from what is left of the subset space
        nfixed_samples = nsamples_added
        samples_left = self.nsamples - nsamples_added
        if num_full_subsets != num_subset_sizes:
            remaining_weight_vector = copy.copy(weight_vector)
            # because we draw two samples each below
            remaining_weight_vector[:num_paired_subset_sizes] /= 2
            remaining_weight_vector = remaining_weight_vector[num_full_subsets:]
            remaining_weight_vector /= np.sum(remaining_weight_vector)

            # four times generated since it does not sample same coalition twice.
            # we use random samples until we reach target number of samples.
            ind_set = np.random.choice(len(remaining_weight_vector), 4 * samples_left,
                                       p=remaining_weight_vector)
            ind_set_pos = 0
            # used_masks = {}
            while samples_left > 0 and ind_set_pos < len(ind_set):
                mask.fill(0.0)
                # we call np.random.choice once to save time and then just read it here
                ind = ind_set[ind_set_pos]
                ind_set_pos += 1
                subset_size = ind + num_full_subsets + 1
                mask[np.random.permutation(self.nplayers)[:subset_size]] = 1.0

                samples_left -= 1
                addsample(mask, 1.0, nsamples_added)
                nsamples_added += 1
                # add the symmetric sample
                if samples_left > 0 and subset_size <= num_paired_subset_sizes:
                    mask[:] = np.abs(mask - 1)
                    samples_left -= 1
                    addsample(mask, 1.0, nsamples_added)
                    nsamples_added += 1

            # normalize the kernel weights for the random samples to equal the weight left after
            # the fixed enumerated samples have been already counted
            weight_left = np.sum(weight_vector[num_full_subsets:])
            kernel_weights[nfixed_samples:] *= weight_left / kernel_weights[nfixed_samples:].sum()

        return (torch.tensor(mask_matrix, requires_grad=False).bool(),
                torch.tensor(kernel_weights, requires_grad=False))