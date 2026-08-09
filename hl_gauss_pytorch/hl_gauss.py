from __future__ import annotations
from typing import Callable
from math import sqrt

import torch
from torch.special import erf
import torch.nn.functional as F
from torch import nn, tensor, linspace, is_tensor
from torch.nn import Module

from torch_einops_utils import maybe, masked_mean

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

def cast_tensor(t):
    return t if is_tensor(t) else tensor(t)

# symlog / symexp transforms, as proposed in DreamerV3 https://arxiv.org/abs/2301.04104

def symlog(t):
    return t.sign() * (t.abs() + 1.).log()

def symexp(t):
    return t.sign() * (t.abs().exp() - 1.)

# proposed gaussian histogram loss by Imani et al. https://arxiv.org/abs/1806.04613

def HLGaussLoss(
    min_value,
    max_value,
    num_bins,
    sigma = None,
    sigma_to_bin_ratio = None,
    eps = 1e-10,
    clamp_to_range = False,
    min_max_value_on_bin_center = False,
    transform: Callable | None = None,
    inverse_transform: Callable | None = None,
    use_symlog = False
):
    assert num_bins > 1, 'number of bins must be greater than 1'

    if use_symlog:
        assert not (exists(transform) or exists(inverse_transform)), 'either pass `use_symlog = True` or `transform` and `inverse_transform`, not both'
        transform = symlog
        inverse_transform = symexp

    min_value = maybe(transform)(cast_tensor(min_value)).item()
    max_value = maybe(transform)(cast_tensor(max_value)).item()

    if min_max_value_on_bin_center:
        adjustment = (max_value - min_value) / ((num_bins - 1) * 2)
        min_value -= adjustment
        max_value += adjustment

    support = linspace(min_value, max_value, num_bins + 1).float()

    loss_fn = HLGaussLossFromSupport(
        support,
        sigma = sigma,
        sigma_to_bin_ratio = sigma_to_bin_ratio,
        eps = eps,
        clamp_to_range = clamp_to_range,
        use_symlog = use_symlog
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
        clamp_to_range = False,
        transform: Callable | None = None,
        inverse_transform: Callable | None = None,
        use_symlog = False
    ):
        super().__init__()
        self.eps = eps

        if use_symlog:
            assert not (exists(transform) or exists(inverse_transform)), 'either pass `use_symlog = True` or `transform` and `inverse_transform`, not both'
            transform = symlog
            inverse_transform = symexp

        assert exists(transform) == exists(inverse_transform), 'both `transform` and `inverse_transform` must be passed in if either is'

        self.transform = transform
        self.inverse_transform = inverse_transform

        if exists(transform):
            dummy_target = torch.randn(10) * 10
            assert torch.allclose(inverse_transform(transform(dummy_target)), dummy_target, atol = 1e-4), 'inverse_transform must mathematically be the exact inverse of transform'

        assert not (exists(sigma) and exists(sigma_to_bin_ratio)), 'either `sigma` or `sigma_to_bin_ratio` is set but not both'

        if not is_tensor(support):
            support = tensor(support)

        assert ((support[1:] - support[:-1]) > 0.).all(), 'support must be increasing in value throughout'

        mean_bin_size = (support[1:] - support[:-1]).mean().item()

        sigma_to_bin_ratio = default(sigma_to_bin_ratio, DEFAULT_SIGMA_TO_BIN_RATIO)

        sigma = default(sigma, sigma_to_bin_ratio * mean_bin_size) # default sigma to ratio of 2. with bin size, from fig 6 of https://arxiv.org/html/2402.13425v2

        self.sigma = sigma
        assert self.sigma > 0.

        self.num_bins = support.numel() - 1
        self.min_value = support[0].item()
        self.max_value = support[-1].item()
        self.clamp_to_range = clamp_to_range

        self.register_buffer('support', support, persistent = False)
        self.register_buffer('centers', (support[:-1] + support[1:]) / 2, persistent = False)
        self.sigma_times_sqrt_two = sqrt(2.) * sigma

    def transform_from_logits(self, logit):
        prob = logit.softmax(dim = -1)
        return self.transform_from_probs(prob)

    def transform_to_logprobs(self, values):
        probs = self.transform_to_probs(values)
        return log(probs)

    def transform_to_probs(self, target, eps = None):
        target = maybe(self.transform)(target)

        if self.clamp_to_range:
            target = target.clamp(min = self.min_value, max = self.max_value)

        eps = default(eps, self.eps)

        assert self.sigma > 0.

        cdf_evals = erf(subtract('bins, ... -> ... bins', self.support, target) / self.sigma_times_sqrt_two)

        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]

        return divide('... bins, ... -> ... bins', bin_probs, z.clamp(min = eps))

    def transform_from_probs(self, probs):
        pred_value = (probs * self.centers).sum(dim = -1)
        return maybe(self.inverse_transform)(pred_value)

    @torch.autocast('cuda', enabled = False)
    def forward(
        self,
        logits,
        target = None,
        *,
        mask = None
    ):
        return_loss = exists(target)

        if return_loss and logits.shape == target.shape:
            logits = self.transform_to_logprobs(logits)

        assert logits.shape[-1] == self.num_bins

        # if targets are not given, return the predicted value

        if not return_loss:
            return self.transform_from_logits(logits)

        target_probs = self.transform_to_probs(target)

        if logits.ndim > 2:
            logits = rearrange(logits, 'b ... l -> b l ...')
            target_probs = rearrange(target_probs, 'b ... l -> b l ...')

        loss = F.cross_entropy(logits, target_probs, reduction = 'none')
        return masked_mean(loss, mask)

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

    def reset_step(self, reset_value = 1.):
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
        clamp_to_range = False,
        transform: Callable | None = None,
        inverse_transform: Callable | None = None,
        use_symlog = False
    ):
        super().__init__()
        assert num_bins >= (std_dev_range * 2)
        self.running_stats = RunningMeanVariance(**running_stats_kwargs)

        self.num_bins = num_bins
        self.std_dev_range = std_dev_range
        self.clamp_to_range = clamp_to_range
        self.sigma_to_bin_ratio = sigma_to_bin_ratio

        if use_symlog:
            assert not (exists(transform) or exists(inverse_transform)), 'either pass `use_symlog = True` or `transform` and `inverse_transform`, not both'
            transform = symlog
            inverse_transform = symexp

        self.transform = transform
        self.inverse_transform = inverse_transform

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
            clamp_to_range = self.clamp_to_range,
            transform = self.transform,
            inverse_transform = self.inverse_transform
        )

    def forward(
        self,
        logits,
        target = None,
        mask = None,
        freeze_stats = None
    ):
        freeze_stats = default(freeze_stats, self.freeze_stats)

        if self.training and not freeze_stats and exists(target):
            stats_target = maybe(self.transform)(target)
            self.running_stats(stats_target)
            self.set_hl_gauss_loss_from_running()

        return self.hl_gauss_loss(logits, target, mask = mask)

# a layer that contains a projection from the embedding of a network to the predicted bins

class HLGaussLayer(Module):
    def __init__(
        self,
        dim,
        *,
        network: Module = nn.Identity(),
        norm_embed = False,
        hl_gauss_loss: dict | HLGaussLossFromSupport | None = None,
        hl_gauss_loss_running_stats: dict | HLGaussLossFromRunningStats | None = None,
        regress_loss_fn: Callable = F.mse_loss,
        use_regression = False, # can be disabled to compare with regular MSE regression
        regress_activation = nn.Identity(),
        aux_regress_loss_weight = 0.
    ):
        super().__init__()
        assert exists(hl_gauss_loss) or exists(hl_gauss_loss_running_stats)

        # a network before the linear prediction
        # for a deeper value network off a transformer or some other backbone

        self.network = network

        # normalization before projection

        self.norm = nn.RMSNorm(dim) if norm_embed else nn.Identity()

        # instantiate hl gauss loss function

        if isinstance(hl_gauss_loss_running_stats, dict):
            hl_gauss_loss_running_stats = HLGaussLossFromRunningStats(**hl_gauss_loss_running_stats)

        if isinstance(hl_gauss_loss, dict):
            hl_gauss_loss = HLGaussLoss(**hl_gauss_loss)

        self.hl_gauss_loss = default(hl_gauss_loss_running_stats, hl_gauss_loss)

        use_classification = not use_regression
        assert not (use_classification and not exists(self.hl_gauss_loss)), '`hl_gauss_loss` is not defined, only regression is permitted'

        # linear projection to either logits for classification, or single value for regression

        dim_pred = self.hl_gauss_loss.num_bins if use_classification else 1
        self.to_pred = nn.Linear(dim, dim_pred, bias = False)

        self.use_classification = use_classification

        # if using regression, activation to apply after the projection

        self.regress_loss_fn = regress_loss_fn
        self.regress_activation = regress_activation

        # regression auxiliary loss - todo: take a day doing experiments and figure out if it is helpful

        self.has_aux_regress_loss = aux_regress_loss_weight > 0.
        self.aux_regress_loss_weight = aux_regress_loss_weight

    def forward_mse_regression(
        self,
        embed,
        target = None,
        return_logits = False, # 'logits' are just the raw predicted value
        return_value_and_logits = False,
        reduction = 'mean'
    ):
        assert not self.use_classification

        embed = self.norm(embed)
        embed = self.network(embed)

        pred_value = self.to_pred(embed)
        pred_value = self.regress_activation(pred_value)
        pred_value = rearrange(pred_value, '... 1 -> ...')

        return_loss = exists(target)

        if return_value_and_logits:
            return pred_value, pred_value

        if return_logits or not return_loss:
            return pred_value

        return self.regress_loss_fn(pred_value, target, reduction = reduction)

    def forward(
        self,
        embed,
        target = None,
        mask = None,
        return_logits = False,
        return_value_and_logits = False,
        reduction = 'mean'
    ):

        if not self.use_classification:
            return self.forward_mse_regression(
                embed,
                target,
                return_logits = return_logits,
                return_value_and_logits = return_value_and_logits,
                reduction = reduction
            )

        embed = self.norm(embed)
        embed = self.network(embed)

        logits = self.to_pred(embed)

        if return_logits:
            return logits

        return_loss = exists(target)

        if not return_loss:
            pred_value = self.hl_gauss_loss(logits)

            if not return_value_and_logits:
                return pred_value

            return pred_value, logits

        loss = self.hl_gauss_loss(logits, target, mask = mask)

        if self.has_aux_regress_loss:
            assert not exists(mask)

            pred_value = self.hl_gauss_loss(logits)
            regress_loss = self.regress_loss_fn(pred_value, target, reduction = reduction)

            loss = loss + regress_loss * self.aux_regress_loss_weight

        if return_value_and_logits:
            return loss, logits

        return loss
