# NCN — Experiment Notebooks

Extends the paper [Neural Common Neighbor with Completion for Link Prediction](https://arxiv.org/pdf/2302.00890.pdf) with three new aggregation variants built on top of the base NCN model.

---

## Implemented Models

| Notebook                | Predictor      | Type | Aggregation                                          |
| ----------------------- | -------------- | ---- | ---------------------------------------------------- |
| `00_baseline.ipynb`     | `cn1`          | NCN  | Sum pooling over common neighbours                   |
| `01_attncn1.ipynb`      | `attncn1`      | NCN  | Learned dot-product encoder with degree-aware biases |
| `02_graphattncn1.ipynb` | `graphattncn1` | NCN  | Graphormer encoding + attention pooling over CNs     |

### What Each Variant Adds

**Attention (`attncn1`)**

- Replaces sum pooling with a learned per-neighbour attention weight
- Each common neighbour `w` is scored against the target pair `(u, v)` using features `[h_u, h_v, h_w, h_u ⊙ h_v]`
- Softmax-normalised weights produce a weighted sum instead of a plain sum
- AMP disabled — FP16 overflows in the softmax


**Graphormer + Attention (`graphattncn1`)**

- Runs the full Graphormer encoder over CN tokens (same centrality + spatial bias as above)
- Replaces the final mean pooling with **attention pooling**: each Graphormer-enriched token is scored against the target pair `(u, v)` using features `[h_w', h_u, h_v, h_u ⊙ h_v]`, softmax-normalised weights produce a weighted sum
- Combines structural awareness from Graphormer with pair-conditioned selectivity from attention
- AMP enabled

### Improvement Stack (applied to all variants)

| Flag                 | What it does                                                           |
| -------------------- | ---------------------------------------------------------------------- |
| `use_aa=True`        | Adds Adamic-Adar edge score as an extra input feature                  |
| `use_ra=True`        | Adds Resource-Allocation edge score as an extra input feature          |
| `use_diff_feat=True` | Appends element-wise difference `\|h_u − h_v\|` to the predictor input |
| `grad_clip=1.0`      | Clips gradient norm to 1.0 during training                             |

---

## Setup

**1. Create the environment**

```bash
conda env create -f env.yaml
conda activate cnt
```

Tested with: PyTorch 1.13 + PyG 2.2 + OGB 1.3.5

**2. Install Jupyter**

```bash
conda install jupyter
# or
pip install notebook
```

**Prepare Datasets**

```
python ogbdataset.py
```

---

## Running the Notebooks

Launch Jupyter from the `experiments/` directory:

```bash
cd experiments/
jupyter notebook
```

Then open any notebook inside `notebooks/`.

### Notebook Layout

Every notebook follows the same structure:

| Cell          | Purpose                                                            |
| ------------- | ------------------------------------------------------------------ |
| Imports       | Sets up sys.path, patches torch.load, imports utils                |
| Device check  | Prints GPU name and available VRAM                                 |
| Configuration | Sets `PREDICTOR`, `CFG` dict, CSV path, and `save_result()` helper |
| Experiments   | One markdown + one code cell per dataset                           |

### Running a Single Dataset

Each dataset has its own independent cell — you can run just that cell without running any other. Results are automatically saved to `results.csv` after each run and any previous entry for that `(dataset, predictor)` pair is replaced.

