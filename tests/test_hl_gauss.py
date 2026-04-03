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
