import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from tqdm import tqdm
from einops.layers.torch import Rearrange

from hl_gauss_pytorch.hl_gauss import HLGaussLayer, HLGaussLossFromRunningStats

# constants

NUM_BATCHES = 10_000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EVAL_EVERY = 500

NUM_BINS = 200
DIM_HIDDEN = 64
SIGMA = None # if not set, will default to bin size to sigma ratio of 2.

USE_RUNNING_STATS = True

USE_REGRESSION = False
AUX_REGRESS_LOSS_WEIGHT = 0.

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# functions

def divisible_by(num, den):
    return (num % den) == 0

# model

from hl_gauss_pytorch.hl_gauss import HLGaussLayer

class MLP(nn.Module):
    def __init__(
        self,
        dim_hidden = 64,
        running_stats = False
    ):
        super().__init__()
        self.to_embed = nn.Sequential(
            Rearrange('... -> ... 1'),
            nn.Linear(1, dim_hidden),
            nn.SiLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.SiLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.SiLU()
        )

        if running_stats:
            hl_gauss_loss_kwargs = dict(
                hl_gauss_loss_running_stats = HLGaussLossFromRunningStats(
                    num_bins = 200,
                    std_dev_range = 10.,
                    clamp_to_range = True
                )
            )
        else:
            hl_gauss_loss_kwargs = dict(
                hl_gauss_loss = dict(
                    min_value = -1.1,
                    max_value = 1.1,
                    num_bins = 200,
                    sigma_to_bin_ratio = 2.
                )
            )

        self.hl_gauss_layer = HLGaussLayer(
            dim_hidden,
            use_regression = USE_REGRESSION,
            norm_embed = True,
            **hl_gauss_loss_kwargs
        )

    def forward(self, inp, target = None):
        embed = self.to_embed(inp)
        return self.hl_gauss_layer(embed, target)

model = MLP(
    DIM_HIDDEN,
    running_stats = USE_RUNNING_STATS
).to(device)

# optimizer

opt = Adam(model.parameters(), lr = LEARNING_RATE)

# data

def fun(t):
    return t.sin()

def cycle(batch_size):
    while True:
        x = torch.randn(batch_size)
        yield (x.to(device), fun(x).to(device))

dl = cycle(BATCH_SIZE)

# train loop

for step in tqdm(range(NUM_BATCHES)):
    inp, out = next(dl)

    loss = model(inp, out)
    loss.backward()

    if divisible_by(step, 10):
        print(f'loss {loss.item():.3f}')

    opt.step()
    opt.zero_grad()

    if divisible_by(step + 1, EVAL_EVERY):
        inp, out = next(dl)
        pred = model(out)

        mae = F.l1_loss(pred, out)
        print(f'eval mae: {mae.item():.6f}')
