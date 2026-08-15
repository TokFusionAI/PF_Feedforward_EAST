# End-to-end pipeline

This is the runbook for the full behavioral-cloning workflow, in dependency
order. All commands are executed from the repository root. Paths containing
`/data/PF_ATIME_dataset` and `/data/EFIT` refer to the local copies of the
ATIME-aligned dataset and the EFIT reconstruction database — replace them with
your own locations (most entry points expose `--dataset-root` / `--efit-dir`
flags or the `DATASET_ROOT` / `EFIT_DIR` environment variables).

## 0. Environment

```bash
pip install -r requirements.txt
pip install "freegsnke[freegs4e]"   # only needed for the equilibrium stage
```

Data acquisition stages need an MDSplus client and access to the EAST server
(`MDS_HOSTNAME`, default `mds.ipp.ac.cn`). Everything after the dataset is
built works offline from the HDF5 files.

## 1. Build the ATIME-aligned dataset

One HDF5 file per shot (`<shot>.h5`) resampled on the shot's EFIT `ATIME`
grid, containing `/time`, `/R8`, `/Z8`, `/PCPF*`, `/mask_*` and scalar
signals. Also writes a per-shot quality index (`meta/shot_index*.parquet`).

```bash
# small dry run (no files written)
python scripts/build_atime_dataset.py --limit 20 --dry-run --run-tag demo
# full build
python scripts/build_atime_dataset.py --workers 12 --run-tag full_v1 --canonical --resume
```

The dataset layout and quality flags consumed downstream are defined in
`scripts/dataset_io.py` and `bc/data/dataset.py`.

## 2. Shot selection and chronological split

Implements the §2.1 criteria of the paper: shots after the Nov-2022 IGBT-PWM
power-supply upgrade (`shot >= 117203`), all key signals present
(`all_ok`), flat-top longer than 2 s, EFIT available, peak plasma current
above 200 kA → 21 297 discharges, split 80/10/10 **by shot number**
(equivalent to chronological order):

- train 17 038 shots (#117383–#152761)
- val    2 130 shots (#152764–#156432)
- test   2 129 shots (#156434–#159876)

```bash
python -m bc.data.split_shots_by_order_v2 \
    --out-dir meta/split_by_order_igbt \
    --ratios 0.80 0.10 0.10 \
    --min-shot 117203 --min-flat-top-s 2.0 --min-ip-ka 200
```

The shipped `meta/split_by_order_igbt/` already contains the exact lists,
the split `manifest.json` (selection funnel), and the normalization
statistics; re-running is only needed when rebuilding from a fresh database.
(The split builder reads the shot-quality index, flat-top table and logbook
summary — these are cluster-side artifacts and are not redistributed; the
manifest documents every selection threshold.)

## 3. Normalization statistics

Mean/std of the 19-D state and 12-D action computed **on the training split
only**, then applied to all subsets:

```bash
python -m bc.data.normalization \
    --shots-file meta/split_by_order_igbt/train_shots.txt \
    --dataset-root /data/PF_ATIME_dataset \
    --out meta/split_by_order_igbt/norm_stats_notime.npz
```

## 4. (Optional) phase-aware |dI/dt| thresholds

Per-coil, per-phase (ramp-up / flat-top / ramp-down) current-rate quantiles
of the expert trajectories. The P99.9 values are the thresholds used by the
operational filter (§2.3):

```bash
python -m bc.analysis.analyze_didt \
    --shots-file meta/split_by_order_igbt/train_shots.txt \
    --out-dir results/didt_stats
```

Output consumed at runtime: `results/didt_stats/proposed_phase_thresholds.json`
(shipped with this repository).

## 5. Training

DDP training loop; one config per architecture variant. The **proposed
model** is `transformer_bidir_on` (bidirectional Transformer + sinusoidal
physical-time positional encoding, 4.75 M parameters, masked MSE, AdamW +
OneCycle, global batch 256):

```bash
# smoke test (~1 min)
CFG=transformer_bidir_on SEED=44 bash scripts_ablation/run_igbt_ablation.sh

# full single-node run (8 GPUs)
torchrun --standalone --nnodes=1 --nproc_per_node=8 -m ablation.training.train \
    --config configs_ablation/ablation_transformer_bidir_on.yaml --tag transformer_bidir_on_igbt_s44 \
    --override seed=44 \
    --override data.train_shots=meta/split_by_order_igbt/train_shots.txt \
    --override data.val_shots=meta/split_by_order_igbt/val_shots.txt \
    --override data.test_shots=meta/split_by_order_igbt/test_shots.txt \
    --override data.norm_stats=meta/split_by_order_igbt/norm_stats_notime.npz \
    --override log.out_root=results/igbt_ablation/transformer_bidir_on \
    --override log.tb_dir=results/igbt_ablation/transformer_bidir_on/tb
```

Products: `results/<root>/<tag>/checkpoints/best_val.pt`, `train.log`,
TensorBoard events.

### Table 2 — architecture comparison

Six variants × three seeds, launched per (CFG, SEED) pair:

```bash
# per pair, e.g.:
CFG=transformer_off SEED=11 bash scripts_ablation/run_igbt_ablation.sh
# evaluate a finished run:
CFG=transformer_off SEED=11 bash scripts_ablation/run_igbt_ablation_eval.sh
# aggregate all runs into the paper table:
python scripts_ablation/aggregate_igbt_ablation.py
```

Variants: `transformer_bidir_on` (proposed) / `transformer_off` (causal,
no PE) / `transformer_on` (causal + time PE) / `transformer_bidir_off`
(bidirectional, no PE) / `lstm_off` / `mlp_off`.

## 6. Test-set evaluation + operational filter

```bash
python -m paper_igbt.predict --split test   # writes results/paper_igbt/predictions/per_shot_preds.npz
python -m paper_igbt.eval_test              # metrics_summary.json + physical_violation.csv
python -m paper_igbt.constraints_fig        # constraint-satisfaction figure
```

The operational filter itself (amplitude clip ±14.5 kA + causal per-phase
dI/dt clipping) is `bc/evaluation/inference_filter.py::apply_didt_filter`,
used by the evaluation and every FreeGSNKE stage below.

## 7. FreeGSNKE forward equilibrium validation (single shot)

Given predicted PF currents + EFIT-frozen profiles, solve the free-boundary
Grad–Shafranov equation on the EAST machine (`scan_data/machine/EAST_config/`)
and compare the recovered 8 control points with the requested boundary:

- implementation: `bc/gs_forward/` (`freegsnke_east_machine.py` builds the
  tokamak from the XMLs; `precursor_export.py` builds self-contained slice
  bundles; `run_freegsnke_eval.py` drives the solver)
- whole-shot orchestration for the paper's representative discharge:
  `python -m paper_igbt.freegsnke_whole_shot --shot 156588 --precursor ...`
- Fig. 5 rendering: `python -m paper_igbt.plot_pred_wall_dist --shot 156588 ...`

## 8. FreeGSNKE test-set statistics (Table 3 / Fig. 6)

On the cross-β_N chronological split (`meta/split_by_order_betan/`), select
the 100 best-predicted test discharges, run FreeGSNKE on sampled slices per
phase (65×65 grid, EFIT profiles frozen), and aggregate:

```bash
python -m paper_betan.select_top100_shots --n 100     # -> freegsnke_100best.txt
python -m paper_betan.freegsnke_testset_eval --n-shots 999999 --nx 65 \
    --test-shots meta/split_by_order_betan/freegsnke_100best.txt \
    --out-dir results/paper_betan/freegsnke_testset
python -m paper_betan.aggregate_freegsnke_testset
python -m paper_betan.plot_freegsnke_stats
python -m paper_betan.plot_freegsnke_phase_r2          # shape-R² companion figure
```

Selection is transparent about its bias: these are the best-predicted test
discharges; the full test-set current-prediction accuracy is reported
separately (Table 2 / Fig. 3).

## What each stage produces

| Stage | Products |
|---|---|
| Dataset build | per-shot HDF5 + shot-quality index |
| Split / normalization | reproducible shot lists + norm stats (shipped) |
| Training | `best_val.pt`, train log, TensorBoard |
| Evaluation | masked MSE, per-channel R², violation rates, Fig. 3/4 |
| FreeGSNKE | convergence info, boundary RMSE, wall gaps, Figs. 5/6, Table 3 |

## Package map

- `ablation/` — model zoo (Transformer/LSTM/MLP + factory + masked losses),
  19-D `PFDataset`, phase detection, DDP trainer. **The paper model lives
  here** (`transformer_bidir_on`).
- `bc/` — pipeline package: shared constants (channel names, T_max, 14.5 kA
  limit, dI/dt tables), normalization/split builders, evaluation metrics,
  inference filter, dI/dt statistics, `gs_forward/` (MDS/EFIT readers, EAST
  machine build, FreeGSNKE driver, precursor export) and `batch_freegsnke/`
  (multi-shot orchestration).
- `bc_notime/` — minimal subset of the no-time-encoding variant whose
  geometry/mapping modules are imported by the paper packages.
- `scan_data/` — MDSplus access layer + EAST machine XML geometry.
- `paper_igbt/` — main-results package on the chronological IGBT-era split
  (Tables 2, Figs. 3–5).
- `paper_betan/` — equilibrium-statistics package on the cross-β_N split
  (Table 3, Fig. 6).
