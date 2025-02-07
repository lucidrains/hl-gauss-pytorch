import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from tqdm import tqdm
from einops.layers.torch import Rearrange

# constants

NUM_BATCHES = 10_000
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EVAL_EVERY = 500

NUM_BINS = 100
DIM_HIDDEN = 64
USE_REGRESSION = False

# functions

def divisible_by(num, den):
    return (num % den) == 0

# model

from hl_gauss_pytorch.hl_gauss import HLGaussLayer

class MLP(nn.Module):
    def __init__(self, dim_hidden = 64):
        super().__init__()
        self.to_embed = nn.Sequential(
            Rearrange('... -> ... 1'),
            nn.Linear(1, dim_hidden),
            nn.SiLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.SiLU()
        )

        self.hl_gauss_layer = HLGaussLayer(
            dim_hidden,
            use_regression = USE_REGRESSION,
            norm_embed = True,
            hl_gauss_loss = dict(
                min_value = -1.1,
                max_value = 1.1,
                num_bins = NUM_BINS
            )
        )

    def forward(self, inp, target = None):
        embed = self.to_embed(inp)
        return self.hl_gauss_layer(embed, target)

model = MLP(DIM_HIDDEN)

# optimizer

opt = Adam(model.parameters(), lr = LEARNING_RATE)

# data

def fun(t):
    return t.sin()

def cycle(batch_size):
    while True:
        x = torch.randn(BATCH_SIZE)
        yield (x, fun(x))

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
