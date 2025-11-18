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
