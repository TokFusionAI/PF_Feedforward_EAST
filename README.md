# PF-Feedforward-EAST: imitation learning for feedforward coil-current trajectory design on EAST

This repository contains the complete codebase behind the paper
**"Imitation learning for feedforward coil current trajectory design on EAST"**
(J. Lu *et al.*). It implements the full four-stage pipeline of the paper:
building a state–action dataset from the EAST discharge archive, training a
Transformer policy by behavioral cloning, filtering the predictions with
operational constraints, and validating the resulting currents with
free-boundary Grad–Shafranov equilibrium solves (FreeGSNKE).

Given a prescribed plasma-boundary evolution (8 LCFS control points, plasma
centroid, plasma current, on the EFIT reconstruction time grid), the trained
model outputs the 12 independent pre-programmed PF coil-current trajectories
(PF1–PF8, PF11–PF14) in a single pass — no plasma simulator in the training
loop, no iterative inverse-equilibrium solves at inference time.

```text
state  s_t ∈ R^19 : [R_1..R_8, Z_1..Z_8, R_c, Z_c, I_p]   (EFIT grid, per time slice)
action a_t ∈ R^12 : [I_PF1..I_PF8, I_PF11..I_PF14]        (PCS feedforward currents)
policy  â_{0:T-1} = π_θ(s_{0:T-1})                        (non-causal Transformer)
```

## Method summary

| Stage | Paper section | What it does | Code |
|---|---|---|---|
| (a) Dataset | §2.1 | Align EFIT boundary points, centroid, Ip and PCS coil currents to each shot's EFIT `ATIME` grid; normal-shot selection; chronological split | `scripts/`, `bc/data/`, `meta/` |
| (b) Behavioral cloning | §2.2 | Bidirectional Transformer (d_model 256, 6 layers, 8 heads, d_ff 1024, 4.75 M params) with sinusoidal **physical-time** positional encoding; masked-MSE loss; AdamW + OneCycle | `ablation/`, `configs_ablation/` |
| (c) Operational filter | §2.3 | Amplitude limit ±14.5 kA; per-coil, per-phase \|dI/dt\| thresholds (99.9th percentile of the training set), applied causally along the EFIT grid | `bc/evaluation/inference_filter.py`, `bc/analysis/analyze_didt.py` |
| (d) Equilibrium validation | §2.4 | FreeGSNKE forward solves with EAST coil/vessel geometry and EFIT-frozen profiles; recovered 8 control points vs. the requested boundary | `bc/gs_forward/`, `paper_igbt/`, `paper_betan/` |

## Repository layout

```text
ablation/            Model zoo: Transformer (± physical-time PE, ± causal), LSTM, MLP;
                     factory, masked losses, PFDataset (19-D state), DDP training loop
bc/                  Core pipeline: constants, phase detection, normalization, split
                     builders, evaluation metrics, dI/dt filter + statistics,
                     FreeGSNKE forward-equilibrium machinery (gs_forward/),
                     batch orchestration (batch_freegsnke/)
bc_notime/           Minimal subset of the no-time-encoding variant imported by the
                     paper packages (gs_forward geometry/mapping, phases)
scan_data/           MDSplus data-access layer (compat, bootstrap, signal configs)
                     and the EAST machine description (machine/EAST_config/*.xml)
scripts/             Dataset build CLI (build_atime_dataset.py, dataset_io.py) and
                     the Fig. 1 R8/Z8 schematic (plot_R8Z8_pf.py)
scripts_ablation/    Chronological-split training/eval launchers, config generator,
                     aggregator for the architecture-comparison table (Table 2),
                     cross-beta_N split construction (for the top-100 validation set)
configs_ablation/    One YAML per model variant of Table 2 (+ smoke configs)
paper_igbt/          Paper package on the chronological IGBT-era split (main results):
                     test-set inference, Table 2 metrics, Figs. 3–5, whole-shot
                     FreeGSNKE validation of the representative discharge #156588
paper_betan/         Paper package for the equilibrium statistics (Table 3, Fig. 6):
                     top-100 shot selection, batch FreeGSNKE test-set evaluation,
                     per-phase RMSE / shape-R² figures
meta/                Shot lists, split manifests, normalization statistics
                     (split_by_order_igbt/, split_by_order_betan/)
results/             Small result artifacts that reproduce the paper numbers:
                     ablation metrics JSONs, Table 2/3 CSVs, per-frame FreeGSNKE
                     statistics, paper figures, and the two trained checkpoints
figures/             Final paper figures (PDF/PNG)
docs/PIPELINE.md     End-to-end runbook (English)
```

## Data availability

Training/evaluation data come from the EAST MDSplus archive (EFIT equilibrium
reconstructions + PCS feedforward commands). Following institutional data
policy they are **not** redistributed with this repository; access can be
requested through the corresponding author of the paper. Everything else
needed to re-run the pipeline — split lists (shot numbers), normalization
statistics, dI/dt thresholds, both trained checkpoints, and all result files
behind Tables 2–3 and Figures 3–6 — is included here.

A small self-contained example shot file (`meta/demo/`-style HDF5) is **not**
included on purpose; the dataset layout expected by `PFDataset` is documented
in `bc/data/dataset.py` (one HDF5 per shot with `/time`, `/R8`, `/Z8`,
`/PCPF*`, `/mask_*` datasets, resampled on the EFIT `ATIME` grid).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# equilibrium validation additionally needs FreeGSNke:
pip install "freegsnke[freegs4e]"
```

Data access (optional, only for re-building the dataset from MDSplus) needs an
`MDSplus` client and the EAST server address in `MDS_HOSTNAME`
(default `mds.ipp.ac.cn`, see `scan_data/mds_bootstrap.py`).

## Checkpoints

Two trained checkpoints of the proposed model (`transformer_bidir_on`,
bidirectional Transformer + physical-time PE) are included:

| File | Trained on | Used for |
|---|---|---|
| `results/igbt_ablation/transformer_bidir_on/transformer_bidir_on_igbt_s44/checkpoints/best_val.pt` | chronological IGBT-era split (17 038 train shots, §2.1) | Table 2, Figs. 3–5 |
| `results/betan_ablation/transformer_bidir_on/transformer_bidir_on_betan_s44/checkpoints/best_val.pt` | cross-β_N chronological split | Table 3, Fig. 6 (top-100 equilibrium statistics) |

Each checkpoint embeds its full training config (`extra["cfg"]`), so the model
rebuilds itself from the checkpoint alone:

```python
from paper_igbt.model import load_best_model
model, cfg, extra = load_best_model()   # reads BEST_CKPT by default
```

## Reproducing the paper

Full command-by-command runbook: [`docs/PIPELINE.md`](docs/PIPELINE.md).
Quick reference:

```bash
# 1) build the per-shot ATIME-aligned dataset from MDSplus (needs data access)
python scripts/build_atime_dataset.py --workers 12 --run-tag full_v1 --canonical

# 2) chronological split + normalization statistics (or use the shipped meta/)
python -m bc.data.split_shots_by_order_v2 --out-dir meta/split_by_order_igbt \
    --ratios 0.80 0.10 0.10 --min-shot 117203 --min-flat-top-s 2.0 --min-ip-ka 200
python -m bc.data.normalization --shots-file meta/split_by_order_igbt/train_shots.txt \
    --dataset-root /data/PF_ATIME_dataset --out meta/split_by_order_igbt/norm_stats_notime.npz

# 3) dI/dt phase-aware thresholds for the operational filter
python -m bc.analysis.analyze_didt --shots-file meta/split_by_order_igbt/train_shots.txt \
    --out-dir results/didt_stats

# 4) train the proposed model (single node, 8 GPUs shown)
torchrun --standalone --nnodes=1 --nproc_per_node=8 -m ablation.training.train \
    --config configs_ablation/ablation_transformer_bidir_on.yaml --tag run_igbt_s44 \
    --override seed=44 \
    --override data.train_shots=meta/split_by_order_igbt/train_shots.txt \
    --override data.val_shots=meta/split_by_order_igbt/val_shots.txt \
    --override data.test_shots=meta/split_by_order_igbt/test_shots.txt \
    --override data.norm_stats=meta/split_by_order_igbt/norm_stats_notime.npz \
    --override log.out_root=results/igbt_ablation/transformer_bidir_on

# 5) all six variants of Table 2 (6 configs x 3 seeds):
CFG=transformer_bidir_on SEED=44 bash scripts_ablation/run_igbt_ablation.sh   # per pair
python scripts_ablation/aggregate_igbt_ablation.py   # -> results/igbt_ablation/table_igbt.{csv,md}

# 6) main-results package: test inference, metrics, Figs. 3–5
python -m paper_igbt.predict --split test
python -m paper_igbt.eval_test
python -m paper_igbt.plot_figures

# 7) equilibrium validation of the representative shot #156588 (Fig. 5)
python -m paper_igbt.freegsnke_whole_shot --shot 156588 \
    --precursor results/freegsnke_precursors/156588/precursor.npz
python -m paper_igbt.plot_pred_wall_dist --shot 156588 \
    --whole-shot-root results/paper_igbt/freegsnke_whole_shot

# 8) top-100 equilibrium statistics (Table 3, Fig. 6)
python -m paper_betan.select_top100_shots --n 100   # -> meta/split_by_order_betan/freegsnke_100best.txt
python -m paper_betan.freegsnke_testset_eval --n-shots 999999 --nx 65 \
    --test-shots meta/split_by_order_betan/freegsnke_100best.txt
python -m paper_betan.aggregate_freegsnke_testset
python -m paper_betan.plot_freegsnke_stats
```

The shipped `results/` directory already contains the outputs of steps 5–8,
so every number and figure in the paper can be checked without re-running:

- **Table 2** ← `results/igbt_ablation/table_igbt.csv` (RMSE 1.150 kA, median R² 0.884)
- **Fig. 3/4** ← `results/paper_igbt/figures/` + `results/paper_igbt/eval_test/metrics_summary.json`
- **Fig. 5** ← `results/paper_igbt/freegsnke_whole_shot/` (shot #156588)
- **Table 3 / Fig. 6** ← `results/paper_betan/freegsnke_testset/per_frame.csv`
  (990 converged slices; mean 8-control-point RMSE 14.2 cm; phase-wise 17.8/12.5/7.5 cm)

## Paper → code map

| Paper item | Script | Output |
|---|---|---|
| Fig. 1 (R8/Z8 control points) | `scripts/plot_R8Z8_pf.py` | `figures/R8Z8_PF.pdf` |
| Fig. 2 (method overview) | schematic (drawn externally) | `figures/PF-Method-V7.pdf` |
| Table 2 (model comparison) | `scripts_ablation/aggregate_igbt_ablation.py` | `results/igbt_ablation/table_igbt.csv` |
| Fig. 3 (per-channel scatter) | `paper_igbt/plot_figures.py`, `paper_igbt/plot_scatter_density.py` | `figures/per_channel_scatter.pdf` |
| Fig. 4 (shot #156588 time series) | `paper_igbt/plot_figures.py` | `figures/shot_timeseries.pdf` |
| Fig. 5 (FreeGSNKE wall-gap validation) | `paper_igbt/freegsnke_whole_shot.py` + `paper_igbt/plot_pred_wall_dist.py` | `figures/pred_wall_dist_shot156588.pdf` |
| Table 3 + Fig. 6 (top-100 equilibrium statistics) | `paper_betan/freegsnke_testset_eval.py`, `paper_betan/aggregate_freegsnke_testset.py`, `paper_betan/plot_freegsnke_stats.py` | `results/paper_betan/freegsnke_testset/per_frame.csv`, `figures/freegsnke_stats.pdf` |

## Citation

If you use this code, please cite the paper and this repository:

```bibtex
@software{pf_feedforward_east_2026,
  author  = {Lu, Jingjing and Wan, Chenguang and EAST Team},
  title   = {{PF-Feedforward-EAST}: imitation learning for feedforward coil-current trajectory design on EAST},
  year    = {2026},
  url     = {https://github.com/TokFusionAI/PF_Feedforward_EAST}
}
```

## License

MIT — see [LICENSE](LICENSE).
