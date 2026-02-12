This repository contains the reproduced code for the paper [Neural Common Neighbor with Completion for Link Prediction](https://arxiv.org/pdf/2302.00890.pdf).

```
For Term Project:
CS60078 - Complex Network Theory @IIT Kharagpur
Members -
Shreyan Naskar 25CS60R41
Swagnik Ghosh 25CS60R67
```

**Environment**

Tested Combination:
torch 1.13.0 + pyg 2.2.0 + ogb 1.3.5

```
conda env create -f env.yaml
```

**Prepare Datasets**

```
python ogbdataset.py
```

**Reproduce Results**

We implement the following models.

| name     | $model   | command change                   |
| -------- | -------- | -------------------------------- |
| GAE      | cn0      |                                  |
| NCN      | cn1      |                                  |
| NCNC     | incn1cn1 |                                  |
| NCNC2    | incn1cn1 | add --depth 2 --splitsize 131072 |
| GAE+CN   | scn1     |                                  |
| NCN2     | cn1.5    |                                  |
| NCN-diff | cn1res   |                                  |
| NoTLR    | cn1      | delete --maskinput               |

To reproduce the results, please modify the following commands as shown in the table above.

Cora

```
python NeighborOverlap.py   --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --predp 0.05 --gnndp 0.05  --probscale 4.3 --proboffset 2.8 --alpha 1.0  --gnnlr 0.0043 --prelr 0.0024  --batch_size 1152  --ln --lnnn --predictor $model --dataset Cora  --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact
```

Citeseer

```
python NeighborOverlap.py   --xdp 0.4 --tdp 0.0 --pt 0.75 --gnnedp 0.0 --preedp 0.0 --predp 0.55 --gnndp 0.75  --probscale 6.5 --proboffset 4.4 --alpha 0.4  --gnnlr 0.0085 --prelr 0.0078  --batch_size 384  --ln --lnnn --predictor $model --dataset Citeseer  --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1  --testbs 4096  --maskinput  --jk  --use_xlin  --tailact  --twolayerlin
```

Pubmed

```
python NeighborOverlap.py   --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --predp 0.05 --gnndp 0.1  --probscale 5.3 --proboffset 0.5 --alpha 0.3  --gnnlr 0.0097 --prelr 0.002  --batch_size 2048  --ln --lnnn --predictor $model --dataset Pubmed  --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact
```

collab

```
python NeighborOverlap.py   --xdp 0.25 --tdp 0.05 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --predp 0.3 --gnndp 0.1  --probscale 2.5 --proboffset 6.0 --alpha 1.05  --gnnlr 0.0082 --prelr 0.0037  --batch_size 65536  --ln --lnnn --predictor $model --dataset collab  --epochs 100 --runs 10 --model gcn --hiddim 64 --mplayers 1  --testbs 131072  --maskinput --use_valedges_as_input   --res  --use_xlin  --tailact
```

ppa

```
python NeighborOverlap.py  --xdp 0.0 --tdp 0.0 --gnnedp 0.1 --preedp 0.0 --predp 0.1 --gnndp 0.0 --gnnlr 0.0013 --prelr 0.0013  --batch_size 16384  --ln --lnnn --predictor $model --dataset ppa   --epochs 25 --runs 10 --model gcn --hiddim 64 --mplayers 3 --maskinput  --tailact  --res  --testbs 65536 --proboffset 8.5 --probscale 4.0 --pt 0.1 --alpha 0.9 --splitsize 131072
```
