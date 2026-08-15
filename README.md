# PF-Feedforward-EAST: physics-informed imitation learning for feedforward coil-current trajectory design on EAST

This repository contains the complete codebase behind the paper
**"Physics-informed imitation learning for feedforward coil-current
trajectory design on EAST"** (J. Lu *et al.*, Communications Physics
submission). It implements the full four-stage pipeline: building a
state–action dataset from the EAST discharge archive, training a Transformer
policy by behavioral cloning, filtering the predictions with operational
constraints, and validating the resulting currents with free-boundary
Grad–Shafranov equilibrium solves (FreeGSNKE).

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

## Study design

The model is deliberately evaluated under a combined temporal and
operating-regime shift. From the 21,297 normal IGBT-era discharges
(shot ≥ 117203), the pool is partitioned by discharge time and normalized
beta together: **training (8,048 shots, #117383–#152761) and validation
(1,022 shots, #152777–#156389) use the β_N < 0.8 band of the earlier two
chronological windows, and the test set (931 shots, #156434–#159876) uses
the β_N > 0.8 band of the latest window (β_N from 0.8 to 1.5)**. The model
therefore trains on earlier, lower-β_N operation and is evaluated on later,
higher-β_N discharges — an extrapolation regime that mirrors how new
campaigns progress toward higher performance. On this test set the proposed
model reaches an overall RMSE of 0.98 kA and a mean per-coil R² of 0.845
(Table 1, Figs. 2–3).

Equilibrium consistency is then assessed with FreeGSNKE across the held-out
cohort: reconstructions driven by the predicted PF currents are compared
both with those driven by the recorded PCS currents (Table 2) and, in the
appendix, with the EFIT target boundary (Table 5).

## Repository layout

```text
ablation/            Model zoo: Transformer (± physical-time PE, ± causal), LSTM, MLP;
                     factory, masked losses, PFDataset (19-D state), DDP training loop
bc/                  Core pipeline: constants, phase detection, normalization, split
                     builders, evaluation metrics, dI/dt filter + statistics, and the
                     FreeGSNKE forward-equilibrium machinery (gs_forward/)
bc_notime/           Minimal subset of the no-time-encoding variant imported by the
                     paper package (gs_forward geometry/mapping, phases)
scan_data/           MDSplus data-access layer (compat, bootstrap, signal configs)
                     and the EAST machine description (machine/EAST_config/*.xml)
scripts/             Dataset build CLI (build_atime_dataset.py, dataset_io.py) and the
                     Fig. 5 R8/Z8 schematic (plot_R8Z8_pf.py)
scripts_ablation/    Cross-βN split construction (make_split_by_betan.py + extractors),
                     training/eval launchers and aggregator (Table 1), and the
                     Figure 1 / Figure 3 / Figure 7 plotting scripts
paper_betan/         Paper package: test-set inference and metrics, Figures 2 and 4,
                     whole-shot FreeGSNKE validation, the batch test-set evaluation
                     (Table 2 / Table 5 runner), and the predicted-vs-recorded
                     comparison (compare_pred_vs_pcs.py)
configs_ablation/    One YAML per architecture variant of Table 1 (+ smoke configs)
meta/                Shot lists, split manifests, per-shot βN table, normalization
                     statistics, discharge dates (split_by_order_betan/); IGBT-era
                     pool lists used as the split basis (split_by_order_igbt/)
results/             Result artifacts that reproduce the paper numbers: Table 1/2/5
                     data, ablation metrics, per-shot predictions, FreeGSNKE
                     per-frame statistics, figures, and the two trained checkpoints
figures/             Final paper figures (PDF/PNG)
docs/PIPELINE.md     End-to-end runbook (English)
```

## Data availability

Training/evaluation data come from the EAST MDSplus archive (EFIT equilibrium
reconstructions + PCS feedforward commands). Following institutional data
policy they are **not** redistributed with this repository; access can be
requested through the corresponding author of the paper. Everything else
needed to re-run the pipeline — split lists (shot numbers), per-shot βN
values, normalization statistics, dI/dt thresholds, both trained
checkpoints, and all result files behind Tables 1–2/5 and Figures 1–7 — is
included here.

The dataset layout expected by `PFDataset` is documented in
`bc/data/dataset.py`: one HDF5 file per shot with `/time`, `/R8`, `/Z8`,
`/PCPF*`, `/mask_*` datasets, resampled on the EFIT `ATIME` grid.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# equilibrium validation additionally needs FreeGSNke:
pip install "freegsnke[freegs4e]"
```

Data acquisition (optional, only for re-building the dataset from MDSplus)
needs an `MDSplus` client and the EAST server address in `MDS_HOSTNAME`
(default `mds.ipp.ac.cn`, see `scan_data/mds_bootstrap.py`).

## Checkpoints

Two trained checkpoints of the proposed model (`transformer_bidir_on`,
bidirectional Transformer + physical-time PE, 4.75 M parameters) are
included:

| File | Purpose |
|---|---|
| `results/betan_ablation/transformer_bidir_on/transformer_bidir_on_betan_s44/checkpoints/best_val.pt` | Figures 1, 2 and 4 (test-set inference and equilibrium validation) |
| `results/betan_ablation/transformer_bidir_on/transformer_bidir_on_betan_s20260424/checkpoints/best_val.pt` | Table 1 proposed row (overall RMSE 0.98 kA) |

Each checkpoint embeds its full training configuration (`extra["cfg"]`), so
the model rebuilds itself from the checkpoint alone:

```python
from paper_betan.model import load_paper_model
model, cfg, extra = load_paper_model()   # reads MODEL_CKPT by default
```

## Reproducing the paper

Full command-by-command runbook: [`docs/PIPELINE.md`](docs/PIPELINE.md).
Quick reference:

```bash
# 1) dataset construction from MDSplus (needs data access)
python scripts/build_atime_dataset.py --workers 12 --run-tag full_v1 --canonical

# 2) cross-beta_N split + normalization statistics (or use the shipped meta/)
python -m scripts_ablation.make_split_by_betan          # -> meta/split_by_order_betan/
python -m bc.data.normalization \
    --shots-file meta/split_by_order_betan/train_shots.txt \
    --dataset-root /data/PF_ATIME_dataset \
    --out meta/split_by_order_betan/norm_stats_notime.npz

# 3) phase-aware |dI/dt| thresholds for the operational filter
python -m bc.analysis.analyze_didt \
    --shots-file meta/split_by_order_betan/train_shots.txt --out-dir results/didt_stats

# 4) train the proposed model (Table 1); all six variants via the launcher
CFG=transformer_bidir_on SEED=44 bash scripts_ablation/run_betan_ablation.sh
python scripts_ablation/aggregate_betan_ablation.py     # -> results/betan_ablation/table_betan.csv

# 5) test-set inference + metrics; Figures 1–3
python -m paper_betan.predict --split test
python -m paper_betan.eval_test
SHOT=156715 python scripts_ablation/plot_pf_timeseries_shot.py   # Fig. 1
python -m paper_betan.plot_scatter_density                        # Fig. 2
python scripts_ablation/plot_betan_degradation_r2.py --bins 13 --drop-pf6-neg-r2 \
    --out betan_r2_13bins_mean                                    # Fig. 3

# 6) FreeGSNKE validation of the representative discharge #156715 (Fig. 4)
python -m paper_betan.freegsnke_whole_shot --shot 156715 \
    --precursor results/freegsnke_precursors/156715/precursor.npz
python -m paper_betan.plot_pred_wall_dist --shot 156715 \
    --whole-shot-root results/paper_betan/freegsnke_whole_shot

# 7) test-set equilibrium comparison: predicted vs recorded currents
python -m paper_betan.freegsnke_testset_eval --current-source pred --nx 65 ...
python -m paper_betan.freegsnke_testset_eval --current-source pcs  --nx 65 ...
python -m paper_betan.compare_pred_vs_pcs --stat mean          # Table 5 (vs EFIT)
python -m paper_betan.compare_pred_vs_pcs --stat mean --direct # Table 2 (paired)
```

The shipped `results/` directory already contains the outputs of steps 4–7,
so every number and figure in the paper can be checked without re-running.

## Paper → code map

| Paper item | Script | Output |
|---|---|---|
| Table 1 (architecture comparison) | `scripts_ablation/aggregate_betan_ablation.py` | `results/betan_ablation/table_betan.csv` |
| Fig. 1 (time traces, discharge #156715) | `scripts_ablation/plot_pf_timeseries_shot.py` | `figures/shot_timeseries_156715.pdf` |
| Fig. 2 (per-channel scatter) | `paper_betan/plot_scatter_density.py` | `figures/per_channel_scatter.pdf` |
| Fig. 3 (per-shot R² vs β_N, 13 bins) | `scripts_ablation/plot_betan_degradation_r2.py` | `figures/betan_r2_13bins_mean.pdf` |
| Fig. 4 (FreeGSNKE wall-gap validation) | `paper_betan/freegsnke_whole_shot.py` + `paper_betan/plot_pred_wall_dist.py` | `figures/pred_wall_dist_shot156715.pdf` |
| Fig. 5 (R8/Z8 control setup) | `scripts/plot_R8Z8_pf.py` | `figures/R8Z8_PF.pdf` |
| Fig. 6 (method overview) | schematic (drawn externally) | `figures/PF-Method-V7.pdf` |
| Fig. 7 (train/val/test partition) | `scripts_ablation/plot_betan_split.py` | `figures/split_overview_betan.pdf` |
| Table 2 (predicted vs recorded currents) | `paper_betan/compare_pred_vs_pcs.py --direct --stat mean` | printed table; data in `results/paper_betan/freegsnke_testset_pf6keep_{pred100,pcs100}/` |
| Table 3/4 (model I/O, hyperparameters) | `ablation/models/`, `configs_ablation/ablation_transformer_bidir_on.yaml` | — |
| Table 5 (appendix, vs EFIT) | `paper_betan/compare_pred_vs_pcs.py --stat mean` | printed table; same data |

## Citation

If you use this code, please cite the paper and this repository:

```bibtex
@software{pf_feedforward_east_2026,
  author  = {Lu, Jingjing and Wan, Chenguang and EAST Team},
  title   = {{PF-Feedforward-EAST}: physics-informed imitation learning for feedforward coil-current trajectory design on EAST},
  year    = {2026},
  url     = {https://github.com/TokFusionAI/PF_Feedforward_EAST}
}
```

## License

MIT — see [LICENSE](LICENSE).
