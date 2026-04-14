"""
Author hyperparameter presets for the experiment notebooks.

The per-dataset hyperparameters follow the paper README, while the notebook
schedule is intentionally lighter: 100 epochs and 1 run by default so each
notebook executes a single author-style trial that is still easy to compare.

Use make_args(BASE, predictor='cn1', **overrides) to get a Namespace for
run_experiment().
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse


NOTEBOOK_EPOCHS = 100
NOTEBOOK_RUNS = 1

# ── Dataset base presets ────────────────────────────────────────────────────

CORA = dict(
    dataset='Cora', model='puregcn', hiddim=256, mplayers=1,
    xdp=0.7, tdp=0.3, gnndp=0.05, gnnedp=0.0,
    predp=0.05, preedp=0.4,
    gnnlr=0.0043, prelr=0.0024,
    batch_size=1152, testbs=8192,
    ln=True, lnnn=True, jk=True, use_xlin=True, tailact=True,
    maskinput=True, res=False, twolayerlin=False,
    probscale=4.3, proboffset=2.8, pt=0.75, alpha=1.0,
    use_valedges_as_input=False,
    epochs=NOTEBOOK_EPOCHS, runs=NOTEBOOK_RUNS,
)

CITESEER = dict(
    dataset='Citeseer', model='puregcn', hiddim=256, mplayers=1,
    xdp=0.4, tdp=0.0, gnndp=0.75, gnnedp=0.0,
    predp=0.55, preedp=0.0,
    gnnlr=0.0085, prelr=0.0078,
    batch_size=384, testbs=4096,
    ln=True, lnnn=True, jk=True, use_xlin=True, tailact=True,
    maskinput=True, res=False, twolayerlin=True,
    probscale=6.5, proboffset=4.4, pt=0.75, alpha=0.4,
    use_valedges_as_input=False,
    epochs=NOTEBOOK_EPOCHS, runs=NOTEBOOK_RUNS,
)

PUBMED = dict(
    dataset='Pubmed', model='puregcn', hiddim=256, mplayers=1,
    xdp=0.3, tdp=0.0, gnndp=0.1, gnnedp=0.0,
    predp=0.05, preedp=0.0,
    gnnlr=0.0097, prelr=0.002,
    batch_size=2048, testbs=8192,
    ln=True, lnnn=True, jk=True, use_xlin=True, tailact=True,
    maskinput=True, res=False, twolayerlin=False,
    probscale=5.3, proboffset=0.5, pt=0.5, alpha=0.3,
    use_valedges_as_input=False,
    epochs=NOTEBOOK_EPOCHS, runs=NOTEBOOK_RUNS,
)

COLLAB = dict(
    dataset='collab', model='gcn', hiddim=64, mplayers=1,
    xdp=0.25, tdp=0.05, gnndp=0.1, gnnedp=0.25,
    predp=0.3, preedp=0.0,
    gnnlr=0.0082, prelr=0.0037,
    batch_size=65536, testbs=131072,
    ln=True, lnnn=True, jk=False, use_xlin=True, tailact=True,
    maskinput=True, res=True, twolayerlin=False,
    probscale=2.5, proboffset=6.0, pt=0.1, alpha=1.05,
    use_valedges_as_input=True,
    epochs=NOTEBOOK_EPOCHS, runs=NOTEBOOK_RUNS,
)

PPA = dict(
    dataset='ppa', model='gcn', hiddim=64, mplayers=3,
    xdp=0.0, tdp=0.0, gnndp=0.0, gnnedp=0.1,
    predp=0.1, preedp=0.0,
    gnnlr=0.0013, prelr=0.0013,
    batch_size=16384, testbs=65536,
    ln=True, lnnn=True, jk=False, use_xlin=False, tailact=True,
    maskinput=True, res=True, twolayerlin=False,
    probscale=4.0, proboffset=8.5, pt=0.1, alpha=0.9,
    splitsize=131072, use_valedges_as_input=False,
    epochs=NOTEBOOK_EPOCHS, runs=NOTEBOOK_RUNS,
)

DDI = dict(
    dataset='ddi', model='puresum', hiddim=224, mplayers=1,
    xdp=0.05, tdp=0.0, gnndp=0.4, gnnedp=0.0,
    predp=0.6, preedp=0.0,
    gnnlr=0.0021, prelr=0.0018,
    batch_size=24576, testbs=131072,
    ln=True, lnnn=True, jk=False, use_xlin=True, tailact=False,
    maskinput=True, res=True, twolayerlin=True,
    probscale=10.0, proboffset=3.0, pt=0.1, alpha=0.5,
    use_valedges_as_input=False,
    epochs=NOTEBOOK_EPOCHS, runs=NOTEBOOK_RUNS,
)

PRESETS = {'Cora': CORA, 'Citeseer': CITESEER, 'Pubmed': PUBMED,
           'collab': COLLAB, 'ppa': PPA, 'ddi': DDI}

# ── Factory ─────────────────────────────────────────────────────────────────

_DEFAULTS = dict(
    predictor='cn1', nnlayers=3,
    beta=1.0, cndeg=-1, trndeg=-1, tstdeg=-1,
    depth=1, splitsize=-1, cnprob=0.0,
    learnpt=False, increasealpha=False,
    # improvement flags
    use_aa=False, use_ra=False,
    use_amp=False, grad_clip=0.0,
    use_gru=False, gru_layers=1,
    use_diff_feat=False, use_degree_feat=False,
    weight_decay=0.0, attn_temp=1.0,
    lrscheduler='none', lr_min=1e-6, lr_patience=10,
    # encoder selection ('gcn' or 'graphormer_node')
    encoder='gcn',
    gnn_nhead=None,
    # i/o
    load=None, loadx=False, loadmod=False,
    save_gemb=False, savex=False, savemod=False,
    require_cuda=False,
)


def make_args(base: dict, **overrides) -> argparse.Namespace:
    """
    Build an argparse.Namespace from a dataset preset + any overrides.

    Examples
    --------
    args = make_args(CORA, predictor='cn1')
    args = make_args(CORA, predictor='grucn1', use_amp=True, grad_clip=1.0)
    args = make_args(COLLAB, predictor='cn1', use_diff_feat=True)
    """
    cfg = {**_DEFAULTS, **base, **overrides}
    return argparse.Namespace(**cfg)
