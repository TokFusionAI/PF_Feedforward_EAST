# End-to-end pipeline

Runbook for the full behavioral-cloning workflow, in dependency order. All
commands run from the repository root. `/data/PF_ATIME_dataset` and
`/data/EFIT` refer to local copies of the ATIME-aligned dataset and the EFIT
reconstruction database — replace them with your own locations (most entry
points expose `--dataset-root` / `--efit-dir` flags or environment
variables).

## 0. Environment

```bash
pip install -r requirements.txt
pip install "freegsnke[freegs4e]"   # only needed for the equilibrium stage
```

Data acquisition needs an MDSplus client and access to the EAST server
(`MDS_HOSTNAME`, default `mds.ipp.ac.cn`). Everything after the dataset is
built works offline from the HDF5 files.

## 1. Build the ATIME-aligned dataset

One HDF5 file per shot, resampled on the shot's EFIT `ATIME` grid:
`/time`, `/R8`, `/Z8` (8 boundary control points), `/PCPF*` (12 commanded
coil currents), `/mask_*` quality flags, and scalar signals (centroid,
plasma current). Also writes a per-shot quality index.

```bash
python scripts/build_atime_dataset.py --limit 20 --dry-run --run-tag demo  # smoke
python scripts/build_atime_dataset.py --workers 12 --run-tag full_v1 --canonical --resume
```

Layout and quality flags: `scripts/dataset_io.py`, `bc/data/dataset.py`.

## 2. Shot selection and the cross-β_N partition

Normal-shot criteria (shot ≥ 117203 after the Nov-2022 IGBT-PWM
power-supply upgrade, complete key signals, flat-top > 2 s, peak Ip
> 200 kA) yield 21,297 discharges — the lists in
`meta/split_by_order_igbt/`. A discharge-mean β_N is computed per shot
from the EFIT β_N signal (`scripts_ablation/extract_betan_per_shot.py`,
cached in `meta/split_by_order_betan/betan_per_shot.parquet`), and the
pool is partitioned by discharge time and β_N together
(`scripts_ablation/make_split_by_betan.py`):

- training 8,048 shots (#117383–#152761), β_N < 0.8
- validation 1,022 shots (#152777–#156389), β_N < 0.8
- test 931 shots (#156434–#159876), β_N from 0.8 to 1.5

```bash
python -m scripts_ablation.extract_betan_per_shot     # per-shot beta_N table
python -m scripts_ablation.make_split_by_betan        # -> meta/split_by_order_betan/
```

The shipped `meta/split_by_order_betan/` contains the exact lists and the
split `manifest.json`; re-running is only needed when rebuilding from a
fresh database.

## 3. Normalization statistics

Mean/std of the 19-D state and 12-D action computed on the training split
only, applied to all subsets:

```bash
python -m bc.data.normalization \
    --shots-file meta/split_by_order_betan/train_shots.txt \
    --dataset-root /data/PF_ATIME_dataset \
    --out meta/split_by_order_betan/norm_stats_notime.npz
```

## 4. (Optional) phase-aware |dI/dt| thresholds

Per-coil, per-phase (ramp-up / flat-top / ramp-down) current-rate quantiles
of the expert trajectories; the P99.9 values feed the operational filter:

```bash
python -m bc.analysis.analyze_didt \
    --shots-file meta/split_by_order_betan/train_shots.txt \
    --out-dir results/didt_stats
```

Runtime dependency: `results/didt_stats/proposed_phase_thresholds.json`
(shipped).

## 5. Training

DDP training loop; one config per architecture variant. The **proposed
model** is `transformer_bidir_on` (bidirectional Transformer + sinusoidal
physical-time positional encoding, 4.75 M parameters, masked MSE, AdamW +
OneCycle, global batch 256).

```bash
# per (CFG, SEED) pair; Table 1 uses seeds 11/44/20260424
CFG=transformer_bidir_on SEED=44 bash scripts_ablation/run_betan_ablation.sh
CFG=transformer_bidir_on SEED=44 bash scripts_ablation/run_betan_ablation_eval.sh
python scripts_ablation/aggregate_betan_ablation.py   # -> table_betan.{csv,md}
```

Variants: `transformer_bidir_on` (proposed) / `transformer_off` (causal,
no PE) / `transformer_on` (causal + time PE) / `transformer_bidir_off`
(bidirectional, no PE) / `lstm_off` / `mlp_off`. The proposed row of
Table 1 reports the `s20260424` run (overall RMSE 0.98 kA); the ablation
rows report three-seed means.

## 6. Test-set evaluation + operational filter

```bash
python -m paper_betan.predict --split test   # per_shot_preds.npz (shipped for 3 seeds)
python -m paper_betan.eval_test              # metrics_summary.json + violation rates
```

Operational filter (amplitude clip ±14.5 kA + causal per-phase dI/dt
clipping): `bc/evaluation/inference_filter.py::apply_didt_filter`.

## 7. Prediction figures (Figs. 1–3)

```bash
SHOT=156715 python scripts_ablation/plot_pf_timeseries_shot.py    # Fig. 1
python -m paper_betan.plot_scatter_density                          # Fig. 2 (hex style)
python scripts_ablation/plot_betan_degradation_r2.py --bins 13 \
    --drop-pf6-neg-r2 --out betan_r2_13bins_mean                    # Fig. 3
```

Figure 2 and Figure 3 exclude test discharges whose per-shot PF6 R² is
negative (`results/paper_betan/figures/_pf6_neg_r2_shots.csv`, written by
`plot_scatter_density.py`).

## 8. FreeGSNKE equilibrium validation

**Representative discharge (Fig. 4).** Solve the free-boundary
Grad–Shafranov equation along the whole discharge with the EAST machine
(`scan_data/machine/EAST_config/`), EFIT-frozen profiles, and the filtered
predicted currents; then render the three-phase wall-gap figure:

```bash
python -m paper_betan.freegsnke_whole_shot --shot 156715 \
    --precursor results/freegsnke_precursors/156715/precursor.npz \
    --out-root results/paper_betan/freegsnke_whole_shot
python -m paper_betan.plot_pred_wall_dist --shot 156715 \
    --whole-shot-root results/paper_betan/freegsnke_whole_shot
```

**Cohort-level comparison (Tables 2 and 5).** Run the batch test-set
evaluation twice over the evaluation shot list — once driving FreeGSNKE
with the predicted currents, once with the recorded PCS currents — then
aggregate:

```bash
python -m paper_betan.freegsnke_testset_eval --current-source pred --nx 65 ...
python -m paper_betan.freegsnke_testset_eval --current-source pcs  --nx 65 ...
python -m paper_betan.aggregate_freegsnke_testset

python -m paper_betan.compare_pred_vs_pcs --stat mean           # Table 5 (vs EFIT)
python -m paper_betan.compare_pred_vs_pcs --stat mean --direct  # Table 2 (paired direct)
```

Boundary RMSE comes from the per-frame CSVs alone; the translation-invariant
shape R² additionally reads the EFIT R8/Z8 control points from the local
EFIT h5 database (`--efit-dir`, default `/data/EFIT`).

## What each stage produces

| Stage | Products |
|---|---|
| Dataset build | per-shot HDF5 + shot-quality index |
| Split / normalization | reproducible shot lists + norm stats (shipped) |
| Training | `best_val.pt` per run, train log, TensorBoard |
| Evaluation | masked MSE, per-channel R², violation rates |
| FreeGSNKE | convergence info, boundary RMSE, wall gaps, Tables 2/5 |

## Package map

- `ablation/` — model zoo (Transformer/LSTM/MLP + factory + masked losses),
  19-D `PFDataset`, phase detection, DDP trainer. The paper model is
  `transformer_bidir_on`.
- `bc/` — shared constants (channel names, T_max, 14.5 kA limit, dI/dt
  tables), normalization/split builders, evaluation metrics, inference
  filter, dI/dt statistics, `gs_forward/` (MDS/EFIT readers, EAST machine
  build, FreeGSNKE driver, precursor export).
- `bc_notime/` — minimal subset of the no-time-encoding variant whose
  geometry/mapping modules are imported by `paper_betan`.
- `scan_data/` — MDSplus access layer + EAST machine XML geometry.
- `scripts_ablation/` — split construction, launchers, Table 1 aggregator,
  Figures 1/3/7 scripts.
- `paper_betan/` — the paper package on the cross-β_N split.
