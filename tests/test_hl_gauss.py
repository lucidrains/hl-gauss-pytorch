import pytest
param = pytest.mark.parametrize

import torch

@param('use_regression', (False, True))
def test_hl_gauss(
    use_regression
):
    from hl_gauss_pytorch import HLGaussLayer

    from x_mlps_pytorch import MLP

    hl_gauss_layer = HLGaussLayer(
        network = MLP(256, 128),
        dim = 128,
        use_regression = use_regression,
        hl_gauss_loss = dict(
            min_value = -100.,
            max_value = 100.,
            num_bins = 32
        )
    )

    embed = torch.randn(7, 256)
    targets = torch.randint(0, 5, (7,)).float()

    loss = hl_gauss_layer(embed, targets)
    loss.backward()

    # after much training

    pred_target = hl_gauss_layer(embed) # (7,)

    assert pred_target.shape == (7,)

def test_hl_gauss_symlog():
    from hl_gauss_pytorch import HLGaussLayer
    from x_mlps_pytorch import MLP

    def symlog(x):
        return x.sign() * (x.abs() + 1.0).log()

    def symexp(x):
        return x.sign() * (x.abs().exp() - 1.0)

    hl_gauss_layer = HLGaussLayer(
        network = MLP(256, 128),
        dim = 128,
        use_regression = False,
        hl_gauss_loss = dict(
            min_value = -100.,
            max_value = 100.,
            num_bins = 256,
            transform = symlog,
            inverse_transform = symexp
        )
    )

    embed = torch.randn(7, 256)
    targets = torch.randn(7)

    loss = hl_gauss_layer(embed, targets)
    loss.backward()

    pred_target = hl_gauss_layer(embed)

    assert pred_target.shape == (7,)

    hl_gauss_loss = hl_gauss_layer.hl_gauss_loss

    probs = hl_gauss_loss.transform_to_probs(targets)
    decoded = hl_gauss_loss.transform_from_probs(probs)

    assert torch.allclose(targets, decoded, atol = 1e-5)

@param('running_stats', (False, True))
def test_hl_gauss_symlog_flag(running_stats):
    from hl_gauss_pytorch import HLGaussLayer

    hl_gauss_kwargs = dict(use_symlog = True) if running_stats else dict(min_value = -100., max_value = 100., use_symlog = True)

    hl_gauss_layer = HLGaussLayer(
        dim = 128,
        hl_gauss_loss_running_stats = dict(num_bins = 32, **hl_gauss_kwargs) if running_stats else None,
        hl_gauss_loss = None if running_stats else dict(num_bins = 32, **hl_gauss_kwargs)
    )

    embed = torch.randn(7, 128)
    targets = torch.randn(7) * 100

    loss = hl_gauss_layer(embed, targets)
    loss.backward()

    pred_target = hl_gauss_layer(embed)

    assert pred_target.shape == (7,)

def test_hl_gauss_explicit_transform_matches_use_symlog_flag():
    from hl_gauss_pytorch import HLGaussLoss

    def symlog(x):
        return x.sign() * (x.abs() + 1.0).log()

    def symexp(x):
        return x.sign() * (x.abs().exp() - 1.0)

    loss_fn_flag = HLGaussLoss(min_value = -100., max_value = 100., num_bins = 256, use_symlog = True)
    loss_fn_transform = HLGaussLoss(min_value = -100., max_value = 100., num_bins = 256, transform = symlog, inverse_transform = symexp)

    targets = torch.randn(100) * 5

    probs_flag = loss_fn_flag.transform_to_probs(targets)
    probs_transform = loss_fn_transform.transform_to_probs(targets)

    assert torch.allclose(probs_flag, probs_transform, atol = 1e-6)

    decoded = loss_fn_transform.transform_from_probs(probs_transform)
    assert torch.allclose(targets, decoded, atol = 1e-3)

def test_hl_gauss_out_of_range_targets():
    from hl_gauss_pytorch import HLGaussLoss

    loss_fn = HLGaussLoss(min_value = -100., max_value = 100., num_bins = 32)

    logits = torch.randn(2, 32, requires_grad = True)
    targets = torch.tensor([175., -175.])

    probs = loss_fn.transform_to_probs(targets)
    assert torch.allclose(probs.sum(dim = -1), torch.ones(2), atol = 1e-6)

    loss = loss_fn(logits, targets)
    assert loss.item() > 0.

    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0.

def test_binary_hl_gauss_out_of_range_targets():
    from hl_gauss_pytorch import BinaryHLGaussLoss

    loss_fn = BinaryHLGaussLoss(sigma = 0.15)

    probs = loss_fn.transform_to_probs(torch.tensor([2.4, -2.4]))

    assert torch.allclose(probs, torch.tensor([1., 0.]), atol = 1e-7)

def test_hl_gauss_layer_mask_and_running_stats():
    from hl_gauss_pytorch import HLGaussLayer

    hl_gauss_layer = HLGaussLayer(
        dim = 128,
        hl_gauss_loss_running_stats = dict(num_bins = 32)
    )

    embed = torch.randn(7, 128)
    targets = torch.randn(7)

    loss = hl_gauss_layer(embed, targets, mask = torch.rand(7) > 0.3)
    loss.backward()

    assert loss.item() > 0.

def test_hl_gauss_reduction():
    from hl_gauss_pytorch import HLGaussLoss

    loss_fn = HLGaussLoss(min_value = -100., max_value = 100., num_bins = 32)

    logits = torch.randn(7, 32)
    targets = torch.randint(0, 5, (7,)).float()

    loss = loss_fn(logits, targets)

    loss_none = loss_fn(logits, targets, reduction = 'none')
    assert loss_none.shape == (7,)
    assert torch.allclose(loss, loss_none.mean())

    loss_sum = loss_fn(logits, targets, reduction = 'sum')
    assert torch.allclose(loss_sum, loss_none.sum())

    mask = torch.rand(7) > 0.3
    loss_mask = loss_fn(logits, targets, mask = mask)

    assert torch.allclose(loss_mask, loss_none[mask].mean())

    loss_mask_sum = loss_fn(logits, targets, mask = mask, reduction = 'sum')
    assert torch.allclose(loss_mask_sum, loss_none[mask].sum())

    loss_mask_none = loss_fn(logits, targets, mask = mask, reduction = 'none')
    assert loss_mask_none.shape == (7,)

def test_hl_gauss_layer_reduction():
    from hl_gauss_pytorch import HLGaussLayer

    hl_gauss_layer = HLGaussLayer(
        dim = 128,
        hl_gauss_loss = dict(min_value = -100., max_value = 100., num_bins = 32)
    )

    embed = torch.randn(7, 128)
    targets = torch.randn(7)

    loss = hl_gauss_layer(embed, targets, reduction = 'none')
    assert loss.shape == (7,)

    loss_mask = hl_gauss_layer(embed, targets, mask = torch.rand(7) > 0.3)
    loss_mask.backward()

    assert loss_mask.item() > 0.

