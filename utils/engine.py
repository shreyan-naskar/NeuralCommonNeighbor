"""
Training engine — shared by all notebooks.
run_experiment(args)  runs the full training pipeline and returns a results dict.
"""
import sys, os, gc
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from functools import partial

from torch_sparse import SparseTensor
from torch_geometric.utils import negative_sampling
from ogb.linkproppred import Evaluator

from model import predictor_dict, GCN, encoder_dict
from graph_utils import PermIterator
from ogbdataset import loaddataset


# ── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_device(require_cuda: bool = False) -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        props = torch.cuda.get_device_properties(0)
        print(f"GPU  : {props.name}")
        print(f"VRAM : {props.total_memory/1e9:.1f} GB")
        return torch.device('cuda')
    if require_cuda:
        raise RuntimeError('CUDA not available.')
    print('No GPU found — falling back to CPU.')
    return torch.device('cpu')


def free_gpu():
    """Release cached GPU memory between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── One training epoch ────────────────────────────────────────────────────────

def _train_epoch(model, predictor, data, split_edge, optimizer,
                 batch_size, maskinput, scaler, grad_clip):
    model.train()
    predictor.train()
    pos_edge = split_edge['train']['edge'].to(data.x.device).t()
    adjmask  = torch.ones_like(pos_edge[0], dtype=torch.bool)
    negedge  = negative_sampling(data.edge_index.to(pos_edge.device),
                                  data.adj_t.sizes()[0])
    losses = []
    for perm in PermIterator(adjmask.device, adjmask.shape[0], batch_size):
        optimizer.zero_grad()
        if maskinput:
            adjmask[perm] = 0
            tei = pos_edge[:, adjmask]
            adj = SparseTensor.from_edge_index(
                tei, sparse_sizes=(data.num_nodes, data.num_nodes)
            ).to_device(pos_edge.device, non_blocking=True).to_symmetric()
            adjmask[perm] = 1
        else:
            adj = data.adj_t
        h = model(data.x, adj)
        pos_out = predictor.multidomainforward(h, adj, pos_edge[:, perm], cndropprobs=[])
        neg_out = predictor.multidomainforward(h, adj, negedge[:, perm],  cndropprobs=[])
        loss    = -F.logsigmoid(pos_out).mean() - F.logsigmoid(-neg_out).mean()

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(),     grad_clip)
            nn.utils.clip_grad_norm_(predictor.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(model, predictor, data, split_edge, evaluator,
              batch_size, use_valedges):
    model.eval(); predictor.eval()

    dev = data.x.device  # use x.device — SparseTensor.device() can return CPU

    def pred(edges):
        edges_dev = edges.to(dev)
        return torch.cat([
            predictor(h, adj, edges_dev[p].t()).squeeze().cpu()
            for p in PermIterator(dev, edges_dev.shape[0], batch_size, False)
        ])

    adj = data.adj_t
    h   = model(data.x, adj)
    p_tr = pred(split_edge['train']['edge'])
    p_va = pred(split_edge['valid']['edge'])
    n_va = pred(split_edge['valid']['edge_neg'])

    if use_valedges:
        adj = data.full_adj_t
        h   = model(data.x, adj)
    p_te = pred(split_edge['test']['edge'])
    n_te = pred(split_edge['test']['edge_neg'])

    out = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        out[f'Hits@{K}'] = (
            evaluator.eval({'y_pred_pos': p_tr, 'y_pred_neg': n_va})[f'hits@{K}'],
            evaluator.eval({'y_pred_pos': p_va, 'y_pred_neg': n_va})[f'hits@{K}'],
            evaluator.eval({'y_pred_pos': p_te, 'y_pred_neg': n_te})[f'hits@{K}'],
        )
    return out, h.cpu()


# ── OOM-safe wrappers ─────────────────────────────────────────────────────────

def _train_epoch_safe(model, predictor, data, split_edge, optimizer,
                      batch_size, maskinput, scaler, grad_clip):
    while batch_size >= 16:
        try:
            loss = _train_epoch(model, predictor, data, split_edge, optimizer,
                                batch_size, maskinput, scaler, grad_clip)
            return loss, batch_size
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad()
            free_gpu()
            batch_size = batch_size // 2
            print(f"  [OOM] train batch_size → {batch_size}", flush=True)
    raise RuntimeError("OOM even at batch_size=16.")


def _evaluate_safe(model, predictor, data, split_edge, evaluator,
                   batch_size, use_valedges):
    while batch_size >= 16:
        try:
            out, h = _evaluate(model, predictor, data, split_edge, evaluator,
                               batch_size, use_valedges)
            return out, h, batch_size
        except torch.cuda.OutOfMemoryError:
            free_gpu()
            batch_size = batch_size // 2
            print(f"  [OOM] eval  batch_size → {batch_size}", flush=True)
    raise RuntimeError("OOM even at eval batch_size=16.")


# ── Predictor factory ─────────────────────────────────────────────────────────

def _build_predfn(args):
    p   = args.predictor
    pfn = predictor_dict[p]

    if p != 'cn0':
        pfn = partial(pfn, cndeg=args.cndeg)
    if p in ['cn1','attncn1','transcn1','incn1cn1','attnincn1cn1',
             'transincn1cn1','scn1','catscn1','sincn1cn1',
             'graphormercn1','graphormerincn1cn1',
             'gatedgraphormercn1','gatedgraphormerincn1cn1',
             'setcn1','setincn1cn1','linearcn1','linearincn1cn1']:
        pfn = partial(pfn, use_xlin=args.use_xlin, tailact=args.tailact,
                      twolayerlin=args.twolayerlin, beta=args.beta)
    if p in ['cn1','attncn1','transcn1','aacn1','racn1','aaracn1',
             'incn1cn1','attnincn1cn1','transincn1cn1',
             'aaincn1cn1','raincn1cn1','aaraincn1cn1','grucn1','gruincn1cn1',
             'graphormercn1','graphormerincn1cn1',
             'gatedgraphormercn1','gatedgraphormerincn1cn1',
             'setcn1','setincn1cn1','linearcn1','linearincn1cn1']:
        pfn = partial(pfn, use_aa=args.use_aa, use_ra=args.use_ra)
    if getattr(args,'use_gru',False) and p in [
            'cn1','attncn1','transcn1','incn1cn1','attnincn1cn1','transincn1cn1']:
        pfn = partial(pfn, use_gru=True)
    if p in ['cn1','attncn1','transcn1','incn1cn1','attnincn1cn1',
             'transincn1cn1','grucn1','gruincn1cn1']:
        pfn = partial(pfn, gru_layers=args.gru_layers)
    if p in ['attncn1','attnincn1cn1']:
        pfn = partial(pfn, attn_temp=args.attn_temp)
    if getattr(args,'use_diff_feat',False) and p in [
            'cn0','cn1','attncn1','transcn1','aacn1','racn1','aaracn1','grucn1',
            'incn1cn1','attnincn1cn1','transincn1cn1',
            'aaincn1cn1','raincn1cn1','aaraincn1cn1','gruincn1cn1',
            'graphormercn1','graphormerincn1cn1',
            'gatedgraphormercn1','gatedgraphormerincn1cn1',
            'setcn1','setincn1cn1','linearcn1','linearincn1cn1']:
        pfn = partial(pfn, use_diff_feat=True)
    if p in ['incn1cn1','attnincn1cn1','transincn1cn1',
             'aaincn1cn1','raincn1cn1','aaraincn1cn1','gruincn1cn1',
             'graphormerincn1cn1','gatedgraphormerincn1cn1',
             'setincn1cn1','linearincn1cn1']:
        pfn = partial(pfn, depth=args.depth, splitsize=args.splitsize,
                      scale=args.probscale, offset=args.proboffset,
                      trainresdeg=args.trndeg, testresdeg=args.tstdeg,
                      pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)
    if p in ['setcn1','setincn1cn1']:
        pfn = partial(pfn, set_ninduce=getattr(args, 'set_ninduce', 32))
    return pfn


# ── Degree feature augmentation ───────────────────────────────────────────────

def _add_degree_feat(data, args):
    if getattr(args, 'use_degree_feat', False) and data.max_x < 0:
        deg = data.adj_t.sum(dim=1).to_dense().view(-1, 1)
        data.x = torch.cat([data.x, torch.log1p(deg)], dim=-1)
    return data


def _select_primary_metric(dataset: str) -> str:
    if dataset == 'collab':
        return 'Hits@50'
    if dataset == 'ddi':
        return 'Hits@20'
    return 'Hits@100'


# ── Main experiment runner ────────────────────────────────────────────────────

def run_experiment(args, verbose: bool = True) -> dict:
    """
    Run all `args.runs` seeds of the NCN/NCNC experiment defined by `args`.

    Returns
    -------
    dict with keys: val_mean, val_std, tst_mean, tst_std, per_run, metric
    """
    device = get_device(getattr(args, 'require_cuda', False))
    evaluator = (Evaluator('ogbl-ppa')
                 if args.dataset in ['Cora','Citeseer','Pubmed']
                 else Evaluator(f'ogbl-{args.dataset}'))

    predfn     = _build_predfn(args)
    metric_key = _select_primary_metric(args.dataset)

    train_bs = args.batch_size
    test_bs  = args.testbs

    if args.dataset not in ['Cora','Citeseer','Pubmed']:
        data, split_edge = loaddataset(args.dataset, args.use_valedges_as_input,
                                       getattr(args,'load',None))
        data = _add_degree_feat(data.to(device), args)
        data.adj_t      = data.adj_t.to(device)
        data.full_adj_t = data.full_adj_t.to(device)

    ret          = []
    ret_best_tst = []
    ret_all          = None
    ret_all_best_tst = None
    for run in range(args.runs):
        set_seed(run)

        if args.dataset in ['Cora','Citeseer','Pubmed']:
            data, split_edge = loaddataset(args.dataset, args.use_valedges_as_input,
                                           getattr(args,'load',None))
            data = _add_degree_feat(data.to(device), args)
            data.adj_t      = data.adj_t.to(device)
            data.full_adj_t = data.full_adj_t.to(device)

        encoder_name = getattr(args, 'encoder', 'gcn')
        if encoder_name == 'gcn':
            model = GCN(
                data.num_features, args.hiddim, args.hiddim, args.mplayers,
                args.gnndp, args.ln, args.res, data.max_x,
                args.model, args.jk, args.gnnedp,
                xdropout=args.xdp, taildropout=args.tdp,
                noinputlin=getattr(args,'loadx',False)
            ).to(device)
        else:
            EncoderCls = encoder_dict[encoder_name]
            model = EncoderCls(
                data.num_features, args.hiddim, args.hiddim, args.mplayers,
                dropout=args.gnndp, max_x=data.max_x, edrop=args.gnnedp,
                nhead=getattr(args, 'gnn_nhead', None),
                xdropout=getattr(args, 'xdp', 0.0),
            ).to(device)

        predictor = predfn(
            args.hiddim, args.hiddim, 1,
            args.nnlayers, args.predp, args.preedp, args.lnnn
        ).to(device)

        wd  = getattr(args, 'weight_decay', 0.0)
        pgs = [{'params': model.parameters(),     'lr': args.gnnlr},
               {'params': predictor.parameters(), 'lr': args.prelr}]
        optimizer = (torch.optim.AdamW(pgs, weight_decay=wd)
                     if wd > 0 else torch.optim.Adam(pgs))

        sched_type = getattr(args, 'lrscheduler', 'none')
        if sched_type == 'cosine':
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=getattr(args,'lr_min',1e-6))
        elif sched_type == 'plateau':
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=getattr(args,'lr_patience',10), factor=0.5)
        else:
            sched = None

        bestscore = None
        best_tst  = None   # tracks max test score per metric regardless of val
        for epoch in range(1, args.epochs + 1):
            alpha = (max(0, min((epoch-5)*0.1, 1))
                     if getattr(args,'increasealpha',False) else None)
            if alpha is not None:
                predictor.setalpha(alpha)

            t0 = time.time()

            loss, train_bs = _train_epoch_safe(
                model, predictor, data, split_edge, optimizer,
                train_bs, args.maskinput, None,
                getattr(args,'grad_clip',0.0))

            res, _, test_bs = _evaluate_safe(
                model, predictor, data, split_edge, evaluator,
                test_bs, args.use_valedges_as_input)

            if bestscore is None:
                bestscore = {k: list(v) for k, v in res.items()}
                best_tst  = {k: list(v) for k, v in res.items()}
            for k, (tr, va, te) in res.items():
                if va > bestscore[k][1]:
                    bestscore[k] = [tr, va, te]
                if te > best_tst[k][2]:
                    best_tst[k] = [tr, va, te]

            if sched is not None:
                if sched_type == 'plateau':
                    sched.step(np.mean([res[k][1] for k in res]))
                else:
                    sched.step()

            if verbose:
                v = res[metric_key][1]; t = res[metric_key][2]
                print(f"  run {run+1}/{args.runs}  ep {epoch:3d}/{args.epochs}"
                      f"  loss={loss:.4f}  {metric_key}=({v:.4f},{t:.4f})"
                      f"  ({time.time()-t0:.1f}s)", flush=True)

        ret.append(bestscore[metric_key][-2:])
        ret_best_tst.append(best_tst[metric_key][-2:])
        if getattr(args, "report_all_hits", False):
            if ret_all is None:
                ret_all          = {k: [] for k in bestscore.keys()}
                ret_all_best_tst = {k: [] for k in best_tst.keys()}
            for k, (_tr, va, te) in bestscore.items():
                ret_all[k].append([va, te])
            for k, (_tr, va, te) in best_tst.items():
                ret_all_best_tst[k].append([va, te])

        del model, predictor, optimizer
        free_gpu()

    ret     = np.array(ret)
    ret_bst = np.array(ret_best_tst)
    out = dict(
        val_mean=float(np.mean(ret[:,0])), val_std=float(np.std(ret[:,0])),
        tst_mean=float(np.mean(ret[:,1])), tst_std=float(np.std(ret[:,1])),
        # best test score across all epochs (independent of val)
        best_tst_mean=float(np.mean(ret_bst[:,1])), best_tst_std=float(np.std(ret_bst[:,1])),
        per_run=ret.tolist(),
        metric=metric_key,
    )
    if getattr(args, "report_all_hits", False) and ret_all is not None:
        all_metrics          = {}
        all_metrics_best_tst = {}
        for k, vals in ret_all.items():
            arr = np.array(vals)
            all_metrics[k] = dict(
                val_mean=float(np.mean(arr[:,0])),
                val_std=float(np.std(arr[:,0])),
                tst_mean=float(np.mean(arr[:,1])),
                tst_std=float(np.std(arr[:,1])),
                per_run=arr.tolist(),
            )
        for k, vals in ret_all_best_tst.items():
            arr = np.array(vals)
            all_metrics_best_tst[k] = dict(
                val_mean=float(np.mean(arr[:,0])),
                val_std=float(np.std(arr[:,0])),
                tst_mean=float(np.mean(arr[:,1])),
                tst_std=float(np.std(arr[:,1])),
                per_run=arr.tolist(),
            )
        out["all_metrics"]          = all_metrics
        out["all_metrics_best_tst"] = all_metrics_best_tst
    print(f"\n{'='*60}")
    print(f"  FINAL {metric_key}  val {out['val_mean']:.4f} ±{out['val_std']:.4f}"
          f"   tst {out['tst_mean']:.4f} ±{out['tst_std']:.4f}"
          f"   best_tst {out['best_tst_mean']:.4f}")
    print(f"{'='*60}")
    return out
