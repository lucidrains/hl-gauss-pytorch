import torch
from torch.special import erf
import torch.nn.functional as F
from torch import nn, tensor, linspace
from torch.nn import Module, ModuleList

import einx

class HLGaussLoss(Module):
    """
    lifted from Appendix A in https://arxiv.org/abs/2403.03950
    """

    def __init__(
        self,
        min_value,
        max_value,
        num_bins,
        sigma
    ):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.sigma = sigma

        support = linspace(min_value, max_value, num_bins + 1).float()

        self.register_buffer('support', support)
        self.register_buffer('centers', (support[:-1] - support[1:]) / 2)
        self.register_buffer('sigma_times_sqrt_two', tensor(2.0).sqrt() * sigma)

    def transform_to_probs(self, target):
        cdf_evals = erf(einx.subtract('bins, ... -> ... bins', self.support, target) / self.sigma_times_sqrt_two)

        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]

        return einx.divide('... bins, ... -> ... bins', bin_probs, z)

    def transform_from_probs(self, probs):
        return (probs * centers).sum(dim = -1)

    def forward(self, logits, target):
        return F.cross_entropy(logits, self.transform_to_probs(target))
