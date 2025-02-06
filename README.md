<img src="./fig2.png" width="400px"></img>

## HL Gauss - Pytorch (wip)

The proposed Gaussian Histogram Loss (HL-Gauss) proposed by Imani et al. with a few convenient wrappers, in Pytorch.

A team at Deepmind wrote a [paper](https://arxiv.org/abs/2403.03950) with a lot of positive findings for its use in value-based RL.

## Install

```bash
$ pip install hl-gauss-pytorch
```

## Usage

```python
import torch
from hl_gauss_pytorch import HLGaussLoss

hl_gauss = HLGaussLoss(0., 5., 32, sigma = 0.5)

logits = torch.randn(3, 16, 32).requires_grad_()
targets = torch.randint(0, 5, (3, 16)).float()

loss = hl_gauss(logits, targets)
loss.backward()

# after much training

pred_target = hl_gauss(logits) # (3, 16)
```

## Citations

```bibtex
@article{Imani2024InvestigatingTH,
    title   = {Investigating the Histogram Loss in Regression},
    author  = {Ehsan Imani and Kai Luedemann and Sam Scholnick-Hughes and Esraa Elelimy and Martha White},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2402.13425},
    url     = {https://api.semanticscholar.org/CorpusID:267770096}
}
```

```bibtex
@inproceedings{Imani2018ImprovingRP,
    title   = {Improving Regression Performance with Distributional Losses},
    author  = {Ehsan Imani and Martha White},
    booktitle = {International Conference on Machine Learning},
    year    = {2018},
    url     = {https://api.semanticscholar.org/CorpusID:48365278}
}
```

```bibtex
@article{Farebrother2024StopRT,
    title   = {Stop Regressing: Training Value Functions via Classification for Scalable Deep RL},
    author  = {Jesse Farebrother and Jordi Orbay and Quan Ho Vuong and Adrien Ali Taiga and Yevgen Chebotar and Ted Xiao and Alex Irpan and Sergey Levine and Pablo Samuel Castro and Aleksandra Faust and Aviral Kumar and Rishabh Agarwal},
    journal = {ArXiv},
    year   = {2024},
    volume = {abs/2403.03950},
    url    = {https://api.semanticscholar.org/CorpusID:268253088}
}
```
