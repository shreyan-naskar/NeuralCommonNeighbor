from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
import torch
from graph_utils import adjoverlap
from torch_sparse.matmul import spmm_max, spmm_mean, spmm_add
from torch_sparse import SparseTensor
import torch_sparse
from torch_scatter import scatter_add, scatter_max
from typing import Iterable, Final
from functools import partial

# a vanilla message passing layer 
class PureConv(nn.Module):
    aggr: Final[str]
    def __init__(self, indim, outdim, aggr="gcn") -> None:
        super().__init__()
        self.aggr = aggr
        if indim == outdim:
            self.lin = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x, adj_t):
        x = self.lin(x)
        if self.aggr == "mean":
            return spmm_mean(adj_t, x)
        elif self.aggr == "max":
            return spmm_max(adj_t, x)[0]
        elif self.aggr == "sum":
            return spmm_add(adj_t, x)
        elif self.aggr == "gcn":
            norm = torch.rsqrt_((1+adj_t.sum(dim=-1))).reshape(-1, 1)
            x = norm * x
            x = spmm_add(adj_t, x) + x
            x = norm * x
            return x
    

convdict = {
    "gcn":
    GCNConv,
    "gcn_cached":
    lambda indim, outdim: GCNConv(indim, outdim, cached=True),
    "sage":
    lambda indim, outdim: GCNConv(
        indim, outdim, aggr="mean", normalize=False, add_self_loops=False),
    "gin":
    lambda indim, outdim: GCNConv(
        indim, outdim, aggr="sum", normalize=False, add_self_loops=False),
    "max":
    lambda indim, outdim: GCNConv(
        indim, outdim, aggr="max", normalize=False, add_self_loops=False),
    "puremax": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="max"),
    "puresum": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="sum"),
    "puremean": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="mean"),
    "puregcn": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="gcn"),
    "none":
    None
}

predictor_dict = {}

# Edge dropout
class DropEdge(nn.Module):

    def __init__(self, dp: float = 0.0) -> None:
        super().__init__()
        self.dp = dp

    def forward(self, edge_index: Tensor):
        if self.dp == 0:
            return edge_index
        mask = torch.rand_like(edge_index[0], dtype=torch.float) > self.dp
        return edge_index[:, mask]

# Edge dropout with adjacency matrix as input
class DropAdj(nn.Module):
    doscale: Final[bool] # whether to rescale edge weight
    def __init__(self, dp: float = 0.0, doscale=True) -> None:
        super().__init__()
        self.dp = dp
        self.register_buffer("ratio", torch.tensor(1/(1-dp)))
        self.doscale = doscale

    def forward(self, adj: SparseTensor)->SparseTensor:
        if self.dp < 1e-6 or not self.training:
            return adj
        mask = torch.rand_like(adj.storage.col(), dtype=torch.float) > self.dp
        adj = torch_sparse.masked_select_nnz(adj, mask, layout="coo")
        if self.doscale:
            if adj.storage.has_value():
                adj.storage.set_value_(adj.storage.value()*self.ratio, layout="coo")
            else:
                adj.fill_value_(1/(1-self.dp), dtype=torch.float)
        return adj


# Vanilla MPNN composed of several layers.
class GCN(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 ln=False,
                 res=False,
                 max_x=-1,
                 conv_fn="gcn",
                 jk=False,
                 edrop=0.0,
                 xdropout=0.0,
                 taildropout=0.0,
                 noinputlin=False):
        super().__init__()
        
        self.adjdrop = DropAdj(edrop)
        
        if max_x >= 0:
            tmp = nn.Embedding(max_x + 1, hidden_channels)
            nn.init.orthogonal_(tmp.weight)
            self.xemb = nn.Sequential(tmp, nn.Dropout(dropout))
            in_channels = hidden_channels
        else:
            self.xemb = nn.Sequential(nn.Dropout(xdropout)) #nn.Identity()
            if not noinputlin and ("pure" in conv_fn or num_layers==0):
                self.xemb.append(nn.Linear(in_channels, hidden_channels))
                self.xemb.append(nn.Dropout(dropout, inplace=True) if dropout > 1e-6 else nn.Identity())
        
        self.res = res
        self.jk = jk
        if jk:
            self.register_parameter("jkparams", nn.Parameter(torch.randn((num_layers,))))
            
        if num_layers == 0 or conv_fn =="none":
            self.jk = False
            return
        
        convfn = convdict[conv_fn]
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        if num_layers == 1:
            hidden_channels = out_channels

        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        if "pure" in conv_fn:
            self.convs.append(convfn(hidden_channels, hidden_channels))
            for i in range(num_layers-1):
                self.lins.append(nn.Identity())
                self.convs.append(convfn(hidden_channels, hidden_channels))
            self.lins.append(nn.Dropout(taildropout, True))
        else:
            self.convs.append(convfn(in_channels, hidden_channels))
            self.lins.append(
                nn.Sequential(lnfn(hidden_channels, ln), nn.Dropout(dropout, True),
                            nn.ReLU(True)))
            for i in range(num_layers - 1):
                self.convs.append(
                    convfn(
                        hidden_channels,
                        hidden_channels if i == num_layers - 2 else out_channels))
                if i < num_layers - 2:
                    self.lins.append(
                        nn.Sequential(
                            lnfn(
                                hidden_channels if i == num_layers -
                                2 else out_channels, ln),
                            nn.Dropout(dropout, True), nn.ReLU(True)))
                else:
                    self.lins.append(nn.Identity())
        

    def forward(self, x, adj_t):
        x = self.xemb(x)
        jkx = []
        for i, conv in enumerate(self.convs):
            x1 = self.lins[i](conv(x, self.adjdrop(adj_t)))
            if self.res and x1.shape[-1] == x.shape[-1]: # residual connection
                x = x1 + x
            else:
                x = x1
            if self.jk:
                jkx.append(x)
        if self.jk: # JumpingKnowledge Connection
            jkx = torch.stack(jkx, dim=0)
            sftmax = self.jkparams.reshape(-1, 1, 1)
            x = torch.sum(jkx*sftmax, dim=0)
        return x


# ── Graphormer node encoder (replaces GCN) ───────────────────────────────────

class _GraphormerNodeLayer(nn.Module):
    """Sparse multi-head self-attention over 1-hop neighbours with degree-aware bias."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead  = nhead
        self.d_head = d_model // nhead
        self.scale  = self.d_head ** -0.5

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff  = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout))

    def forward(self, x: Tensor, adj_t: SparseTensor,
                spatial_bias_emb: nn.Embedding, deg_bucket: Tensor) -> Tensor:
        N, d = x.shape
        H, dh = self.nhead, self.d_head

        row, col, _ = adj_t.coo()   # row=dst, col=src

        Q = self.Wq(x).view(N, H, dh)
        K = self.Wk(x).view(N, H, dh)
        V = self.Wv(x).view(N, H, dh)

        # dot-product scores per edge
        scores = (Q[row] * K[col]).sum(-1) * self.scale   # [E, H]

        # degree-aware spatial bias
        bias = (spatial_bias_emb(deg_bucket[row]) +
                spatial_bias_emb(deg_bucket[col]))         # [E, H]
        scores = scores + bias

        # stable softmax grouped by destination node
        max_s = scatter_max(scores, row, dim=0, dim_size=N)[0]  # [N, H]
        exp_s = torch.exp(scores - max_s[row])                  # [E, H]
        sum_s = scatter_add(exp_s, row, dim=0, dim_size=N)      # [N, H]
        alpha = exp_s / sum_s[row].clamp_min_(1e-12)            # [E, H]

        # weighted sum of values
        agg = scatter_add(
            alpha.unsqueeze(-1) * V[col],   # [E, H, dh]
            row, dim=0, dim_size=N)          # [N, H, dh]

        out = self.Wo(agg.view(N, d))
        x   = self.ln1(x + out)
        x   = self.ln2(x + self.ff(x))
        return x


class GraphormerNodeEncoder(nn.Module):
    """
    Drop-in replacement for GCN that uses Graphormer-style node encoding:
      1. Project input features (or look up node-ID embeddings for ddi/ppa)
      2. Add centrality encoding  (degree-bucket embedding)
      3. Run num_layers of sparse graph self-attention with spatial attention bias
    """

    NUM_DEGREE_BUCKETS: Final[int] = 64

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int,
                 dropout: float = 0.0,
                 max_x: int = -1,
                 edrop: float = 0.0,
                 nhead: int | None = None,
                 xdropout: float = 0.0,
                 **kwargs):          # absorb unused GCN kwargs (ln, res, jk, …)
        super().__init__()
        d = hidden_channels
        B = self.NUM_DEGREE_BUCKETS

        self.adjdrop = DropAdj(edrop)

        if max_x >= 0:
            tmp = nn.Embedding(max_x + 1, d)
            nn.init.orthogonal_(tmp.weight)
            # For node-ID datasets (ddi/ppa), xdropout acts as embedding dropout
            self.xemb = nn.Sequential(tmp, nn.Dropout(xdropout if xdropout > 0 else dropout))
        else:
            # xdropout on raw features (matches GCN's heavy input dropout, e.g. 0.7 for Cora)
            self.xemb = nn.Sequential(
                nn.Dropout(xdropout),
                nn.Linear(in_channels, d),
                nn.Dropout(dropout, inplace=True))

        nhead = nhead or next((h for h in (8, 4, 2, 1) if d % h == 0), 1)

        self.degree_emb   = nn.Embedding(B, d, padding_idx=0)
        self.spatial_bias = nn.ModuleList(
            [nn.Embedding(B, nhead) for _ in range(num_layers)])
        self.layers = nn.ModuleList(
            [_GraphormerNodeLayer(d, nhead, d * 2, dropout)
             for _ in range(num_layers)])

    @staticmethod
    def _deg_bucket(deg: Tensor, B: int = 64) -> Tensor:
        return torch.floor(torch.log2(deg.float().clamp_min_(1.0))).long().clamp_(1, B - 1)

    def forward(self, x: Tensor, adj_t: SparseTensor) -> Tensor:
        adj_t = self.adjdrop(adj_t)
        x     = self.xemb(x)                                    # [N, d]

        # Degree from the actual adjacency (without self-loops) for centrality encoding
        deg        = adj_t.sum(dim=-1).to_dense()               # [N]
        deg_bucket = self._deg_bucket(deg, self.NUM_DEGREE_BUCKETS)
        x = x + self.degree_emb(deg_bucket)                     # centrality encoding

        # Add self-loops so each node can attend to itself (mirrors GCN's +x self-connection)
        adj_with_self = adj_t.set_diag()

        for i, layer in enumerate(self.layers):
            x = layer(x, adj_with_self, self.spatial_bias[i], deg_bucket)
        return x


encoder_dict = {
    'gcn':            GCN,
    'graphormer_node': GraphormerNodeEncoder,
}


# NCN predictor
class CNLinkPredictor(nn.Module):
    cndeg: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 use_attention=False,
                 use_transformer=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0,
                 use_aa=False,
                 use_ra=False,
                 use_gru=False,
                 gru_layers=1,
                 use_diff_feat=False,
                 attn_temp=1.0):
        super().__init__()

        if sum(bool(mode) for mode in (use_attention, use_transformer, use_gru)) > 1:
            raise ValueError(
                "use_attention, use_transformer, and use_gru are mutually exclusive"
            )

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.use_diff_feat = use_diff_feat
        xij_in = in_channels * 2 if use_diff_feat else in_channels
        self.xijlin = nn.Sequential(
            nn.Linear(xij_in, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())

        self.use_aa = use_aa
        self.use_ra = use_ra
        self.aalin = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels)) if use_aa else None
        self.ralin = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels)) if use_ra else None

        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg
        self.use_attention = use_attention
        self.use_transformer = use_transformer
        self.use_gru = use_gru
        self.attn_temp = attn_temp
        self.attlin = None
        self.cn_tokenlin = None
        self.cn_transformer = None
        if use_attention:
            self.attlin = nn.Sequential(
                nn.Linear(in_channels * 4, hidden_channels),
                lnfn(hidden_channels, ln),
                nn.Dropout(dropout, inplace=True),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, 1))
        if use_transformer:
            nhead = next((h for h in (8, 4, 2) if in_channels % h == 0), 1)
            self.cn_tokenlin = nn.Sequential(
                nn.Linear(in_channels * 4, in_channels),
                lnfn(in_channels, ln),
                nn.Dropout(dropout, inplace=True),
                nn.ReLU(inplace=True))
            self.cn_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=in_channels,
                    nhead=nhead,
                    dim_feedforward=max(hidden_channels, in_channels) * 2,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True),
                num_layers=1)
        if use_gru:
            self.gru = nn.GRU(
                input_size=in_channels,
                hidden_size=hidden_channels // 2,
                num_layers=gru_layers,
                bidirectional=True,
                batch_first=True,
                dropout=dropout if gru_layers > 1 else 0.0)

    def aggregate_common_neighbors(self,
                                   cn: SparseTensor,
                                   x: Tensor,
                                   xi: Tensor,
                                   xj: Tensor) -> Tensor:
        if self.use_transformer:
            row, col, value = cn.coo()
            dim_size = xi.shape[0]
            if row.numel() == 0:
                return x.new_zeros((dim_size, x.shape[-1]))

            order = torch.argsort(row)
            row = row[order]
            col = col[order]
            if value is not None:
                value = value[order]

            xw = x[col]
            if value is not None:
                xw = xw * value.unsqueeze(-1)

            tokens = self.cn_tokenlin(
                torch.cat((xw, xi[row], xj[row], xi[row] * xj[row]), dim=-1))

            counts = torch.bincount(row, minlength=dim_size)
            active_rows = torch.nonzero(counts, as_tuple=False).flatten()
            active_counts = counts[active_rows]
            seqs = torch.split(tokens, active_counts.tolist())
            padded = pad_sequence(seqs, batch_first=True)

            maxlen = padded.shape[1]
            steps = torch.arange(maxlen, device=x.device).unsqueeze(0)
            padmask = steps >= active_counts.unsqueeze(1)

            transformed = self.cn_transformer(
                padded, src_key_padding_mask=padmask)
            transformed = transformed.masked_fill(padmask.unsqueeze(-1), 0.0)
            pooled = transformed.sum(dim=1) / active_counts.unsqueeze(-1).clamp_min_(1)

            ret = x.new_zeros((dim_size, x.shape[-1]))
            ret[active_rows] = pooled
            return ret

        if self.use_gru:
            row, col, value = cn.coo()
            dim_size = xi.shape[0]
            if row.numel() == 0:
                return x.new_zeros((dim_size, x.shape[-1]))

            order = torch.argsort(row)
            row = row[order]
            col = col[order]
            if value is not None:
                value = value[order]

            xw = x[col]
            if value is not None:
                xw = xw * value.unsqueeze(-1)

            counts = torch.bincount(row, minlength=dim_size)
            active_rows = torch.nonzero(counts, as_tuple=False).flatten()
            active_counts = counts[active_rows]
            seqs = torch.split(xw, active_counts.tolist())
            padded = pad_sequence(seqs, batch_first=True)

            packed = torch.nn.utils.rnn.pack_padded_sequence(
                padded, active_counts.cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.gru(packed)
            # hidden: (2*num_layers, num_active, hidden//2) — cat fwd+bwd final states
            cn_agg_active = torch.cat([hidden[-2], hidden[-1]], dim=-1)

            ret = x.new_zeros((dim_size, x.shape[-1]))
            ret[active_rows] = cn_agg_active.to(ret.dtype)
            return ret

        if not self.use_attention:
            return spmm_add(cn, x)

        row, col, value = cn.coo()
        dim_size = xi.shape[0]
        if row.numel() == 0:
            return x.new_zeros((dim_size, x.shape[-1]))

        xw = x[col]
        pairfeat = torch.cat((xi[row], xj[row], xw, xi[row] * xj[row]), dim=-1)
        logits = self.attlin(pairfeat).flatten() / self.attn_temp

        max_per_row = scatter_max(logits, row, dim=0, dim_size=dim_size)[0]
        att = torch.exp(logits - max_per_row[row])
        att = att / scatter_add(att, row, dim=0, dim_size=dim_size)[row].clamp_min_(1e-12)

        if value is not None:
            att = att * value

        return scatter_add(att.unsqueeze(-1) * xw, row, dim=0, dim_size=dim_size)

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []):
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        x = x + self.xlin(x)
        cn = adjoverlap(adj, adj, tar_ei, filled1, cnsampledeg=self.cndeg)
        xcns = [self.aggregate_common_neighbors(cn, x, xi, xj)]
        if self.use_diff_feat:
            xij = self.xijlin(torch.cat([xi * xj, (xi - xj).abs()], dim=-1))
        else:
            xij = self.xijlin(xi * xj)

        if self.use_aa or self.use_ra:
            deg = adj.sum(dim=-1).to_dense().clamp_min_(1.0)
            if self.use_aa:
                aa_weight = (1.0 / deg.log().clamp_min_(1e-6)).clamp_max_(1e3).unsqueeze(-1)
                aa_score = spmm_add(cn, aa_weight).reshape(-1, 1)
                aa_score = torch.nan_to_num(aa_score, nan=0.0, posinf=1e3, neginf=0.0)
                xij = xij + self.aalin(aa_score)
            if self.use_ra:
                ra_weight = (1.0 / deg).clamp_max_(1e3).unsqueeze(-1)
                ra_score = spmm_add(cn, ra_weight).reshape(-1, 1)
                ra_score = torch.nan_to_num(ra_score, nan=0.0, posinf=1e3, neginf=0.0)
                xij = xij + self.ralin(ra_score)

        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns],
            dim=-1)
        return xs

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])

# NCNC predictor
class IncompleteCN1Predictor(CNLinkPredictor):
    learnablept: Final[bool]
    depth: Final[int]
    splitsize: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 use_attention=False,
                 use_transformer=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0,
                 alpha=1.0,
                 scale=5,
                 offset=3,
                 trainresdeg=8,
                 testresdeg=128,
                 pt=0.5,
                 learnablept=False,
                 depth=1,
                 splitsize=-1,
                 use_aa=False,
                 use_ra=False,
                 use_gru=False,
                 gru_layers=1,
                 use_diff_feat=False,
                 attn_temp=1.0,
                 ):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout, edrop, ln, cndeg, use_xlin, use_attention, use_transformer, tailact, twolayerlin, beta, use_aa, use_ra, use_gru, gru_layers, use_diff_feat, attn_temp)
        self.learnablept= learnablept
        self.depth = depth
        self.splitsize = splitsize
        self.lins = nn.Sequential()
        self.register_buffer("alpha", torch.tensor([alpha]))
        self.register_buffer("pt", torch.tensor([pt]))
        self.register_buffer("scale", torch.tensor([scale]))
        self.register_buffer("offset", torch.tensor([offset]))

        self.trainresdeg = trainresdeg
        self.testresdeg = testresdeg
        self.ptlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True), nn.Linear(hidden_channels, 1), nn.Sigmoid())

    def clampprob(self, prob, pt):
        p0 = torch.sigmoid_(self.scale*(prob-self.offset))
        return self.alpha*pt*p0/(pt*p0+1-p0)

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = [],
                           depth: int=None):
        assert len(cndropprobs) == 0
        if depth is None:
            depth = self.depth
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        xij_raw = xi * xj
        if self.use_diff_feat:
            xij_input = torch.cat([xij_raw, (xi - xj).abs()], dim=-1)
        else:
            xij_input = xij_raw
        x = x + self.xlin(x)
        if depth > 0.5:
            cn, cnres1, cnres2 = adjoverlap(
                    adj,
                    adj,
                    tar_ei,
                    filled1,
                    calresadj=True,
                    cnsampledeg=self.cndeg,
                    ressampledeg=self.trainresdeg if self.training else self.testresdeg)
        else:
            cn = adjoverlap(
                    adj,
                    adj,
                    tar_ei,
                    filled1,
                    calresadj=False,
                    cnsampledeg=self.cndeg,
                    ressampledeg=self.trainresdeg if self.training else self.testresdeg)
        xcns = [self.aggregate_common_neighbors(cn, x, xi, xj)]

        if depth > 0.5:
            potcn1 = cnres1.coo()
            potcn2 = cnres2.coo()
            with torch.no_grad():
                if self.splitsize < 0:
                    ei1 = torch.stack((tar_ei[1][potcn1[0]], potcn1[1]))
                    probcn1 = self.forward(
                        x, adj, ei1,
                        filled1, depth-1).flatten()
                    ei2 = torch.stack((tar_ei[0][potcn2[0]], potcn2[1]))
                    probcn2 = self.forward(
                        x, adj, ei2,
                        filled1, depth-1).flatten()
                else:
                    num1 = potcn1[1].shape[0]
                    ei1 = torch.stack((tar_ei[1][potcn1[0]], potcn1[1]))
                    probcn1 = torch.empty_like(potcn1[1], dtype=torch.float)
                    for i in range(0, num1, self.splitsize):
                        probcn1[i:i+self.splitsize] = self.forward(x, adj, ei1[:, i: i+self.splitsize], filled1, depth-1).flatten()
                    num2 = potcn2[1].shape[0]
                    ei2 = torch.stack((tar_ei[0][potcn2[0]], potcn2[1]))
                    probcn2 = torch.empty_like(potcn2[1], dtype=torch.float)
                    for i in range(0, num2, self.splitsize):
                        probcn2[i:i+self.splitsize] = self.forward(x, adj, ei2[:, i: i+self.splitsize],filled1, depth-1).flatten()
            if self.learnablept:
                pt = self.ptlin(xij_raw)
                probcn1 = self.clampprob(probcn1, pt[potcn1[0]])
                probcn2 = self.clampprob(probcn2, pt[potcn2[0]])
            else:
                probcn1 = self.clampprob(probcn1, self.pt)
                probcn2 = self.clampprob(probcn2, self.pt)
            probcn1 = probcn1 * potcn1[-1]
            probcn2 = probcn2 * potcn2[-1]
            cnres1.set_value_(probcn1, layout="coo")
            cnres2.set_value_(probcn2, layout="coo")
            xcn1 = self.aggregate_common_neighbors(cnres1, x, xi, xj)
            xcn2 = self.aggregate_common_neighbors(cnres2, x, xi, xj)
            xcns[0] = xcns[0] + xcn2 + xcn1

        xij = self.xijlin(xij_input)

        if self.use_aa or self.use_ra:
            deg = adj.sum(dim=-1).to_dense().clamp_min_(1.0)
            if self.use_aa:
                aa_weight = (1.0 / deg.log().clamp_min_(1e-6)).clamp_max_(1e3).unsqueeze(-1)
                aa_score = spmm_add(cn, aa_weight).reshape(-1, 1)
                aa_score = torch.nan_to_num(aa_score, nan=0.0, posinf=1e3, neginf=0.0)
                xij = xij + self.aalin(aa_score)
            if self.use_ra:
                ra_weight = (1.0 / deg).clamp_max_(1e3).unsqueeze(-1)
                ra_score = spmm_add(cn, ra_weight).reshape(-1, 1)
                ra_score = torch.nan_to_num(ra_score, nan=0.0, posinf=1e3, neginf=0.0)
                xij = xij + self.ralin(ra_score)

        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns],
            dim=-1)
        return xs

    def setalpha(self, alpha: float):
        self.alpha.fill_(alpha)
        print(f"set alpha: {alpha}")

    def forward(self,
                x,
                adj,
                tar_ei,
                filled1: bool = False,
                depth: int = None):
        if depth is None:
            depth = self.depth
        return self.multidomainforward(x, adj, tar_ei, filled1, [],
                                       depth)

# ── Graphormer-style CN predictor ────────────────────────────────────────────

class _GraphormerAttention(nn.Module):
    """Multi-head self-attention with additive per-head bias (Graphormer spatial encoding)."""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead    = nhead
        self.head_dim = d_model // nhead
        self.scale    = self.head_dim ** -0.5
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj     = nn.Linear(d_model, d_model)
        self.drop     = nn.Dropout(dropout)

    def forward(self, x: Tensor,
                attn_bias: Tensor | None = None,
                key_padding_mask: Tensor | None = None) -> Tensor:
        B, L, D = x.shape
        H, E = self.nhead, self.head_dim
        q, k, v = self.qkv(x).reshape(B, L, 3, H, E).unbind(2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)

        # Build combined additive mask (spatial bias + padding) for SDPA
        mask = None
        if attn_bias is not None or key_padding_mask is not None:
            mask = x.new_zeros(B, H, L, L)
            if attn_bias is not None:
                mask = mask + attn_bias
            if key_padding_mask is not None:
                mask = mask.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))

        # Fused kernel: QK^T * scale + mask → softmax → dropout → V
        dp = self.drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dp)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


class _GraphormerLayer(nn.Module):
    """Pre-LN transformer layer with optional additive attention bias."""
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = _GraphormerAttention(d_model, nhead, dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout))

    def forward(self, x: Tensor,
                attn_bias: Tensor | None = None,
                key_padding_mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x), attn_bias, key_padding_mask)
        x = x + self.ff(self.norm2(x))
        return x


class GraphormerCNLinkPredictor(CNLinkPredictor):
    """
    NCN predictor that replaces the vanilla transformer aggregation with a
    Graphormer-style encoder:

    1. **Centrality encoding** – a learned embedding indexed by log2-bucketed
       degree of each common-neighbor node is *added* to each CN token before
       the transformer sees it.  This lets the model distinguish hub CNs from
       peripheral CNs without requiring any graph pre-computation at test time.

    2. **Spatial attention bias** – for each pair of CN tokens (w_i, w_j), a
       per-head learnable scalar bias b[bucket(deg(w_i)), bucket(deg(w_j))]
       is added to the raw attention logits.  The bias matrix is shared across
       the batch and factored as deg_i_emb · deg_j_emb to keep parameters O(B)
       instead of O(B²).

    These are the two core ideas from the Graphormer paper adapted to the
    variable-length common-neighbor set rather than a full graph.
    """

    NUM_DEGREE_BUCKETS: Final[int] = 64   # log2-scale; covers degrees up to 2^63

    def __init__(self, *args, **kwargs):
        # strip use_transformer so super().__init__ doesn't build cn_transformer
        kwargs.pop('use_transformer', None)
        super().__init__(*args, use_transformer=False, **kwargs)

        # infer d_model from the first positional arg (in_channels)
        in_ch = args[0] if args else kwargs.get('in_channels', kwargs.get('hidden_channels', 64))
        nhead = next((h for h in (8, 4, 2, 1) if in_ch % h == 0), 1)
        B     = self.NUM_DEGREE_BUCKETS

        # tokenlin: same as transcn1
        self.gf_tokenlin = nn.Sequential(
            nn.Linear(in_ch * 4, in_ch),
            nn.LayerNorm(in_ch),
            nn.Dropout(kwargs.get('dropout', 0.0), inplace=False),
            nn.ReLU(inplace=True))

        # centrality encoding
        self.gf_degree_emb = nn.Embedding(B, in_ch, padding_idx=0)

        # spatial bias: factored b[i,j,h] = (W_i · W_j) per head
        # W: [B, nhead]  →  bias[b_i, b_j, h] = W[b_i, h] + W[b_j, h]
        self.gf_spatial_bias = nn.Embedding(B, nhead)

        self.gf_layer = _GraphormerLayer(
            d_model=in_ch, nhead=nhead,
            dim_ff=max(in_ch, kwargs.get('hidden_channels', in_ch)) * 2,
            dropout=kwargs.get('dropout', 0.0))

        self.use_graphormer = True

    @staticmethod
    def _deg_bucket(deg: Tensor, num_buckets: int) -> Tensor:
        """Map degree → log2 bucket, clamped to [1, num_buckets-1] (0 = padding)."""
        return torch.floor(torch.log2(deg.float().clamp_min_(1.0))).long().clamp_(1, num_buckets - 1)

    def aggregate_common_neighbors(self,
                                   cn: SparseTensor,
                                   x: Tensor,
                                   xi: Tensor,
                                   xj: Tensor,
                                   adj: SparseTensor | None = None) -> Tensor:
        row, col, value = cn.coo()
        dim_size = xi.shape[0]
        if row.numel() == 0:
            return x.new_zeros((dim_size, x.shape[-1]))

        order = torch.argsort(row)
        row   = row[order];  col = col[order]
        if value is not None:
            value = value[order]

        xw = x[col]
        if value is not None:
            xw = xw * value.unsqueeze(-1)

        # ── centrality encoding ───────────────────────────────────────────────
        if adj is not None:
            deg = adj.sum(dim=-1).to_dense()               # [N]
        else:
            deg = torch.ones(x.shape[0], device=x.device)
        deg_col    = deg[col]                               # degree of each CN
        deg_bucket = self._deg_bucket(deg_col, self.NUM_DEGREE_BUCKETS)  # [E]
        cent_emb   = self.gf_degree_emb(deg_bucket)        # [E, d_model]

        tokens = self.gf_tokenlin(
            torch.cat((xw, xi[row], xj[row], xi[row] * xj[row]), dim=-1))
        tokens = tokens + cent_emb                         # add centrality

        # ── pack into padded batch ────────────────────────────────────────────
        counts      = torch.bincount(row, minlength=dim_size)
        active_rows = torch.nonzero(counts, as_tuple=False).flatten()
        active_cnt  = counts[active_rows]

        # Cap L to avoid O(L²) memory explosion on dense graphs (e.g. collab)
        # Split FIRST with original counts, then truncate each sequence.
        MAX_CN = 64
        seqs     = [s[:MAX_CN] for s in torch.split(tokens,     active_cnt.tolist())]
        deg_seqs = [s[:MAX_CN] for s in torch.split(deg_bucket, active_cnt.tolist())]
        active_cnt = active_cnt.clamp(max=MAX_CN)

        padded      = pad_sequence(seqs,     batch_first=True)   # [A, L, d]
        deg_padded  = pad_sequence(deg_seqs, batch_first=True)   # [A, L]

        A, L, d = padded.shape
        H       = self.gf_layer.attn.nhead

        # ── spatial bias  b[i,j,h] = bias_emb[bucket_i,h] + bias_emb[bucket_j,h] ──
        sb = self.gf_spatial_bias(deg_padded)              # [A, L, H]
        attn_bias = sb.unsqueeze(2) + sb.unsqueeze(1)      # [A, L, L, H]
        attn_bias = attn_bias.permute(0, 3, 1, 2)          # [A, H, L, L]

        steps   = torch.arange(L, device=x.device).unsqueeze(0)
        padmask = steps >= active_cnt.unsqueeze(1)         # [A, L]

        transformed = self.gf_layer(padded, attn_bias=attn_bias,
                                    key_padding_mask=padmask)
        transformed = transformed.masked_fill(padmask.unsqueeze(-1), 0.0)
        pooled      = transformed.sum(1) / active_cnt.unsqueeze(-1).clamp_min_(1)

        ret = x.new_zeros((dim_size, d))
        ret[active_rows] = pooled
        return ret

    def multidomainforward(self, x, adj, tar_ei, filled1=False, cndropprobs=()):
        adj_raw = adj
        adj   = self.dropadj(adj)
        xi    = x[tar_ei[0]];  xj = x[tar_ei[1]]
        x     = x + self.xlin(x)
        cn    = adjoverlap(adj, adj, tar_ei, filled1, cnsampledeg=self.cndeg)
        # pass adj so centrality encoding can read degrees
        xcns  = [self.aggregate_common_neighbors(cn, x, xi, xj, adj=adj_raw)]

        if self.use_diff_feat:
            xij = self.xijlin(torch.cat([xi * xj, (xi - xj).abs()], dim=-1))
        else:
            xij = self.xijlin(xi * xj)

        if self.use_aa or self.use_ra:
            deg = adj_raw.sum(dim=-1).to_dense().clamp_min_(1.0)
            if self.use_aa:
                aa_weight = (1.0 / deg.log().clamp_min_(1e-6)).clamp_max_(1e3).unsqueeze(-1)
                aa_score = spmm_add(cn, aa_weight).reshape(-1, 1)
                aa_score = torch.nan_to_num(aa_score, nan=0.0, posinf=1e3, neginf=0.0)
                xij = xij + self.aalin(aa_score)
            if self.use_ra:
                ra_weight = (1.0 / deg).clamp_max_(1e3).unsqueeze(-1)
                ra_score = spmm_add(cn, ra_weight).reshape(-1, 1)
                ra_score = torch.nan_to_num(ra_score, nan=0.0, posinf=1e3, neginf=0.0)
                xij = xij + self.ralin(ra_score)

        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns], dim=-1)
        return xs

    def forward(self, x, adj, tar_ei, filled1=False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])


class GraphormerIncompleteCN1Predictor(IncompleteCN1Predictor):
    """NCNC (completion) predictor with Graphormer CN aggregation."""

    NUM_DEGREE_BUCKETS: Final[int] = 64

    def __init__(self, *args, **kwargs):
        kwargs.pop('use_transformer', None)
        super().__init__(*args, use_transformer=False, **kwargs)

        in_ch = args[0] if args else kwargs.get('in_channels', kwargs.get('hidden_channels', 64))
        nhead = next((h for h in (8, 4, 2, 1) if in_ch % h == 0), 1)
        B     = self.NUM_DEGREE_BUCKETS

        self.gf_tokenlin    = nn.Sequential(
            nn.Linear(in_ch * 4, in_ch), nn.LayerNorm(in_ch),
            nn.Dropout(kwargs.get('dropout', 0.0), inplace=False), nn.ReLU(inplace=True))
        self.gf_degree_emb  = nn.Embedding(B, in_ch, padding_idx=0)
        self.gf_spatial_bias = nn.Embedding(B, nhead)
        self.gf_layer       = _GraphormerLayer(
            d_model=in_ch, nhead=nhead,
            dim_ff=max(in_ch, kwargs.get('hidden_channels', in_ch)) * 2,
            dropout=kwargs.get('dropout', 0.0))
        self.use_graphormer = True

    # reuse the same methods via mixin-style delegation
    _deg_bucket          = staticmethod(GraphormerCNLinkPredictor._deg_bucket)
    aggregate_common_neighbors = GraphormerCNLinkPredictor.aggregate_common_neighbors
    multidomainforward   = GraphormerCNLinkPredictor.multidomainforward
    forward              = GraphormerCNLinkPredictor.forward

# ── Graphormer + Attention CN predictor ──────────────────────────────────────

class GraphormerAttnCNLinkPredictor(GraphormerCNLinkPredictor):
    """Graphormer CN encoding followed by attention pooling conditioned on the target pair."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_ch   = args[0] if args else kwargs.get('in_channels', kwargs.get('hidden_channels', 64))
        dropout = kwargs.get('dropout', 0.0)
        # Scores each Graphormer-enriched CN token against the target pair
        self.gf_attn_scorer = nn.Sequential(
            nn.Linear(in_ch * 4, in_ch),
            nn.LayerNorm(in_ch),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch, 1))

    def aggregate_common_neighbors(self,
                                   cn: SparseTensor,
                                   x: Tensor,
                                   xi: Tensor,
                                   xj: Tensor,
                                   adj: SparseTensor | None = None) -> Tensor:
        row, col, value = cn.coo()
        dim_size = xi.shape[0]
        if row.numel() == 0:
            return x.new_zeros((dim_size, x.shape[-1]))

        order = torch.argsort(row)
        row   = row[order];  col = col[order]
        if value is not None:
            value = value[order]

        xw = x[col]
        if value is not None:
            xw = xw * value.unsqueeze(-1)

        # centrality encoding
        if adj is not None:
            deg = adj.sum(dim=-1).to_dense()
        else:
            deg = torch.ones(x.shape[0], device=x.device)
        deg_col    = deg[col]
        deg_bucket = self._deg_bucket(deg_col, self.NUM_DEGREE_BUCKETS)
        cent_emb   = self.gf_degree_emb(deg_bucket)

        tokens = self.gf_tokenlin(
            torch.cat((xw, xi[row], xj[row], xi[row] * xj[row]), dim=-1))
        tokens = tokens + cent_emb

        # pack into padded batch
        counts      = torch.bincount(row, minlength=dim_size)
        active_rows = torch.nonzero(counts, as_tuple=False).flatten()
        active_cnt  = counts[active_rows]

        MAX_CN = 64
        seqs     = [s[:MAX_CN] for s in torch.split(tokens,     active_cnt.tolist())]
        deg_seqs = [s[:MAX_CN] for s in torch.split(deg_bucket, active_cnt.tolist())]
        active_cnt = active_cnt.clamp(max=MAX_CN)

        padded      = pad_sequence(seqs,     batch_first=True)   # [A, L, d]
        deg_padded  = pad_sequence(deg_seqs, batch_first=True)   # [A, L]

        A, L, d = padded.shape
        H       = self.gf_layer.attn.nhead

        # spatial bias
        sb        = self.gf_spatial_bias(deg_padded)             # [A, L, H]
        attn_bias = sb.unsqueeze(2) + sb.unsqueeze(1)            # [A, L, L, H]
        attn_bias = attn_bias.permute(0, 3, 1, 2)               # [A, H, L, L]

        steps   = torch.arange(L, device=x.device).unsqueeze(0)
        padmask = steps >= active_cnt.unsqueeze(1)               # [A, L]

        transformed = self.gf_layer(padded, attn_bias=attn_bias,
                                    key_padding_mask=padmask)
        transformed = transformed.masked_fill(padmask.unsqueeze(-1), 0.0)

        # attention pooling: score each token against the target pair, then weighted sum
        xi_act = xi[active_rows]                                 # [A, d]
        xj_act = xj[active_rows]
        xi_pad = xi_act.unsqueeze(1).expand(-1, L, -1)          # [A, L, d]
        xj_pad = xj_act.unsqueeze(1).expand(-1, L, -1)

        scorer_in = torch.cat([transformed, xi_pad, xj_pad,
                                xi_pad * xj_pad], dim=-1)        # [A, L, 4d]
        logits  = self.gf_attn_scorer(scorer_in).squeeze(-1)    # [A, L]
        logits  = logits.masked_fill(padmask, float('-inf'))
        weights = torch.softmax(logits, dim=-1)
        weights = weights.masked_fill(padmask, 0.0)             # zero out padding
        pooled  = (weights.unsqueeze(-1) * transformed).sum(dim=1)  # [A, d]

        ret = x.new_zeros((dim_size, d))
        ret[active_rows] = pooled
        return ret


class GraphormerAttnIncompleteCN1Predictor(GraphormerIncompleteCN1Predictor):
    """NCNC predictor with Graphormer encoding + attention pooling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_ch   = args[0] if args else kwargs.get('in_channels', kwargs.get('hidden_channels', 64))
        dropout = kwargs.get('dropout', 0.0)
        self.gf_attn_scorer = nn.Sequential(
            nn.Linear(in_ch * 4, in_ch),
            nn.LayerNorm(in_ch),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch, 1))

    aggregate_common_neighbors = GraphormerAttnCNLinkPredictor.aggregate_common_neighbors



predictor_dict = {
    # NCN variants
    "cn1":      CNLinkPredictor,
    "attncn1":  partial(CNLinkPredictor, use_attention=True),
    "transcn1": partial(CNLinkPredictor, use_transformer=True),
    # NCNC variants
    "incn1cn1":      IncompleteCN1Predictor,
    "attnincn1cn1":  partial(IncompleteCN1Predictor, use_attention=True),
    "transincn1cn1": partial(IncompleteCN1Predictor, use_transformer=True),
    # Graphormer predictors
    "graphormercn1":      GraphormerCNLinkPredictor,
    "graphormerincn1cn1": GraphormerIncompleteCN1Predictor,
    # Graphormer + Attention predictors
    "graphattncn1":      GraphormerAttnCNLinkPredictor,
    "graphattnincn1cn1": GraphormerAttnIncompleteCN1Predictor,
}
