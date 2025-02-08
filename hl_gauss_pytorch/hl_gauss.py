from __future__ import annotations
from math import sqrt

import torch
from torch.special import erf
import torch.nn.functional as F
from torch import nn, tensor, linspace, is_tensor
from torch.nn import Module, ModuleList

from einx import subtract, divide
from einops import rearrange

# constants

DEFAULT_SIGMA_TO_BIN_RATIO = 2.

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

# proposed gaussian histogram loss by Imani et al. https://arxiv.org/abs/1806.04613

def HLGaussLoss(
    min_value,
    max_value,
    num_bins,
    sigma = None,
    sigma_to_bin_ratio = None,
    eps = 1e-10,
    clamp_to_range = False
):
    support = linspace(min_value, max_value, num_bins + 1).float()

    loss_fn = HLGaussLossFromSupport(
        support,
        sigma = sigma,
        sigma_to_bin_ratio = sigma_to_bin_ratio,
        eps = eps,
        clamp_to_range = clamp_to_range
    )

    return loss_fn

class HLGaussLossFromSupport(Module):
    """
    lifted from Appendix A in https://arxiv.org/abs/2403.03950
    """

    def __init__(
        self,
        support: list[float] | Tensor,
        sigma = None,
        sigma_to_bin_ratio = None,
        eps = 1e-10,
        clamp_to_range = False
    ):
        super().__init__()
        self.eps = eps
        assert exists(sigma) or exists(sigma_to_bin_ratio), 'either `sigma` or `sigma_to_bin_ratio` is set but not both'

        if not is_tensor(support):
            support = tensor(support)

        assert ((support[1:] - support[:-1]) > 0.).all(), 'support must be increasing in value throughout'

        mean_bin_size = (support[1:] - support[:-1]).mean().item()

        sigma_to_bin_ratio = default(sigma_to_bin_ratio, DEFAULT_SIGMA_TO_BIN_RATIO)

        sigma = default(sigma, sigma_to_bin_ratio * mean_bin_size) # default sigma to ratio of 2. with bin size, from fig 6 of https://arxiv.org/html/2402.13425v2

        self.sigma = sigma
        assert self.sigma > 0.

        self.num_bins = support.numel() - 1
        self.min_value = support[0]
        self.max_value = support[-1]
        self.clamp_to_range = clamp_to_range

        self.register_buffer('support', support, persistent = False)
        self.register_buffer('centers', (support[:-1] + support[1:]) / 2, persistent = False)
        self.sigma_times_sqrt_two = sqrt(2.) * sigma

    def transform_to_logprobs(self, values):
        probs = self.transform_to_probs(values)
        return log(probs)

    def transform_to_probs(self, target, eps = None):
        eps = default(eps, self.eps)

        assert self.sigma > 0.

        cdf_evals = erf(subtract('bins, ... -> ... bins', self.support, target) / self.sigma_times_sqrt_two)

        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]

        return divide('... bins, ... -> ... bins', bin_probs, z.clamp(min = eps))

    def transform_from_probs(self, probs):
        return (probs * self.centers).sum(dim = -1)

    def forward_kl_div(
        self,
        pred,
        target
    ):
        """
        allow for predicted value to be passed in, in which it will also be binned to probs and kl div used with target
        """
        assert pred.shape == target.shape

        logprob_pred = self.transform_to_logprobs(pred)
        logprob_target = self.transform_to_logprobs(target)

        return F.kl_div(logprob_pred, logprob_target, log_target = True, reduction = 'mean')

    @torch.autocast('cuda', enabled = False)
    def forward(
        self,
        logits,
        target = None
    ):
        return_loss = exists(target)

        if return_loss and self.clamp_to_range:
            target = target.clamp(min = self.min_value, max = self.max_value)

        if return_loss and logits.shape == target.shape:
            return self.forward_kl_div(logits, target)

        assert logits.shape[-1] == self.num_bins

        if return_loss:
            target_probs = self.transform_to_probs(target)
            return F.cross_entropy(logits, target_probs)

        # if targets are not given, return the predicted value

        pred_probs = logits.softmax(dim = -1)

        return self.transform_from_probs(pred_probs)

# running stats using welford algorithm 1962

class RunningMeanVariance(Module):
    def __init__(
        self,
        eps = 1e-5,
        time_dilate_factor = 1.
    ):
        super().__init__()
        self.eps = eps
        self.time_dilate_factor = time_dilate_factor

        self.register_buffer('step', tensor(1))
        self.register_buffer('running_mean', tensor(100.))
        self.register_buffer('running_estimate_p', tensor(1.))

    def reset_step(self, reset_value = 0.):
        self.step.copy_(reset_value)

    @property
    def time(self):
        return self.step / self.time_dilate_factor

    @property
    def running_variance(self):
        p = self.running_estimate_p

        if self.step == 1:
            return torch.ones_like(p)

        return (p / (self.time - 1. / self.time_dilate_factor))

    @property
    def running_std_dev(self):
        return self.running_variance.clamp(min = self.eps).sqrt()

    def forward(self, values):
        time = self.time.item()
        mean = self.running_mean
        estimate_p = self.running_estimate_p

        new_mean = values.mean()
        delta = new_mean - mean

        mean = mean + delta / time
        estimate_p = estimate_p + (new_mean - mean) * delta

        self.running_mean.copy_(mean)
        self.running_estimate_p.copy_(estimate_p)

        self.step.add_(1)

# a hl gauss loss that automatically determines the supports from the running mean and variance

class HLGaussLossFromRunningStats(Module):
    def __init__(
        self,
        num_bins,
        sigma_to_bin_ratio = 2.,
        std_dev_range = 3,
        running_stats_kwargs: dict = dict(),
        freeze_stats = False,
        clamp_to_range = False
    ):
        super().__init__()
        assert num_bins >= (std_dev_range * 2)
        self.running_stats = RunningMeanVariance(**running_stats_kwargs)

        self.num_bins = num_bins
        self.std_dev_range = std_dev_range
        self.clamp_to_range = clamp_to_range
        self.sigma_to_bin_ratio = sigma_to_bin_ratio

        self.register_buffer('dummy', tensor(0), persistent = False)

        self.freeze_stats = freeze_stats
        self.set_hl_gauss_loss_from_running()

    def set_hl_gauss_loss_from_running(self):

        running_mean = self.running_stats.running_mean.item()
        running_std_dev = self.running_stats.running_std_dev.item()

        min_value = (running_mean - running_std_dev * self.std_dev_range)
        max_value = (running_mean + running_std_dev * self.std_dev_range)

        support = linspace(min_value, max_value, self.num_bins + 1, device = self.dummy.device)

        self.hl_gauss_loss = HLGaussLossFromSupport(
            support = support,
            sigma_to_bin_ratio = self.sigma_to_bin_ratio,
            clamp_to_range = self.clamp_to_range
        )
 
    def forward(
        self,
        logits,
        target = None,
        freeze_stats = None
    ):
        freeze_stats = default(freeze_stats, self.freeze_stats)

        if self.training and not freeze_stats and exists(target):
            self.running_stats(target)
            self.set_hl_gauss_loss_from_running()

        return self.hl_gauss_loss(logits, target)

# a layer that contains a projection from the embedding of a network to the predicted bins

class HLGaussLayer(Module):
    def __init__(
        self,
        dim,
        *,
        norm_embed = False,
        hl_gauss_loss: dict | HLGaussLossFromSupport | HLGaussLossFromRunningStats | None = None,
        regress_loss_fn: Module | Callable = nn.MSELoss(),
        use_regression = False, # can be disabled to compare with regular MSE regression
        regress_activation = nn.Identity(),
        aux_regress_loss_weight = 0.
    ):
        super().__init__()

        self.norm = nn.RMSNorm(dim) if norm_embed else nn.Identity()

        if isinstance(hl_gauss_loss, dict):
            hl_gauss_loss = HLGaussLoss(**hl_gauss_loss)

        self.hl_gauss_loss = hl_gauss_loss

        use_classification = not use_regression
        assert not (use_classification and not exists(hl_gauss_loss)), '`hl_gauss_loss` is not defined, only regression is permitted'

        # linear projection to either logits for classification, or single value for regression

        dim_pred = hl_gauss_loss.num_bins if use_classification else 1
        self.to_pred = nn.Linear(dim, dim_pred, bias = False)

        self.use_classification = use_classification

        # if using regression, activation to apply after the projection

        self.regress_loss_fn = regress_loss_fn
        self.regress_activation = regress_activation

        # regression auxiliary loss - todo: take a day doing experiments and figure out if it is helpful

        self.hax_aux_regress_loss = aux_regress_loss_weight > 0.
        self.aux_regress_loss_weight = aux_regress_loss_weight

    def forward_mse_regression(
        self,
        embed,
        target = None
    ):
        assert not self.use_classification

        embed = self.norm(embed)

        pred_value = self.to_pred(embed)
        pred_value = self.regress_activation(pred_value)
        pred_value = rearrange(pred_value, '... 1 -> ...')

        return_loss = exists(target)

        if not return_loss:
            return pred_value

        return self.regress_loss_fn(pred_value, target)

    def forward(
        self,
        embed,
        target = None,
        return_logits = False
    ):

        if not self.use_classification:
            assert not return_logits, 'no logits to return when using regression'
            return self.forward_mse_regression(embed, target)

        embed = self.norm(embed)
        logits = self.to_pred(embed)

        if return_logits:
            return logits

        return_loss = exists(target)

        if not return_loss:
            pred_value = self.hl_gauss_loss(logits)
            return pred_value

        loss = self.hl_gauss_loss(logits, target)

        if self.hax_aux_regress_loss:
            pred_value = self.hl_gauss_loss(logits)
            regress_loss = self.regress_loss_fn(pred_value, target)

            loss = loss + regress_loss * self.aux_regress_loss_weight

        return loss
