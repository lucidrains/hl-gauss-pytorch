from __future__ import annotations
from math import sqrt

import torch
from torch.special import erf
import torch.nn.functional as F
from torch import nn, tensor, stack, linspace
from torch.nn import Module, ModuleList

from einx import subtract
from einops import rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

# proposed gaussian histogram loss by Imani et al. https://arxiv.org/abs/1806.04613

class BinaryHLGaussLoss(Module):
    """
    lifted from Appendix A in https://arxiv.org/abs/2403.03950
    """

    def __init__(
        self,
        sigma = 0.15,
        min_value = 0.,
        max_value = 1.,
        eps = 1e-10,
        clamp_to_range = True,
    ):
        super().__init__()
        self.eps = eps

        assert max_value > min_value
        bin_size_half = (max_value - min_value) / 2.

        support = linspace(min_value - bin_size_half, max_value + bin_size_half, 3).float()

        self.sigma = sigma
        assert self.sigma > 0.

        self.min_value = min_value
        self.max_value = max_value
        self.clamp_to_range = clamp_to_range

        self.register_buffer('support', support, persistent = False)
        self.register_buffer('centers', (support[:-1] + support[1:]) / 2, persistent = False)
        self.sigma_times_sqrt_two = sqrt(2.) * sigma

    def transform_from_logits(self, logit):
        prob = logit.sigmoid()
        return self.transform_from_probs(prob)

    def transform_to_logprobs(self, values):
        probs = self.transform_to_probs(values)
        return log(probs)

    def transform_to_probs(self, target, eps = None):
        eps = default(eps, self.eps)

        assert self.sigma > 0.

        cdf_evals = erf(subtract('bins, ... -> ... bins', self.support, target) / self.sigma_times_sqrt_two)

        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_prob = cdf_evals[..., -1] - cdf_evals[..., -2]

        return bin_prob / z.clamp(min = eps)

    def transform_from_probs(self, probs):
        probs = stack([1. - probs, probs], dim = -1)
        return (probs * self.centers).sum(dim = -1)

    @torch.autocast('cuda', enabled = False)
    def forward(
        self,
        logits,
        target = None,
        reduction = 'mean'
    ):
        return_loss = exists(target)

        if return_loss and self.clamp_to_range:
            target = target.clamp(min = self.min_value, max = self.max_value)

        if return_loss:
            target_probs = self.transform_to_probs(target)
            return F.binary_cross_entropy(logits.sigmoid(), target_probs, reduction = reduction)

        # if targets are not given, return the predicted value

        return self.transform_from_logits(logits)

# quick test

if __name__ == '__main__':
    binary_hl_gauss = BinaryHLGaussLoss(sigma= 0.1)

    rand_ones_zeros = torch.randint(0, 2, (2, 5)).float()
    logits = rand_ones_zeros * 10 - 5

    loss = binary_hl_gauss(logits, rand_ones_zeros)

    pred_value = binary_hl_gauss(logits)
