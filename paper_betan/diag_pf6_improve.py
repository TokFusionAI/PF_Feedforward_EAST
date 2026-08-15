"""扫描提升 PF6 R² 的合法手段 (在计算节点跑):
1. variant×seed 网格 (架构/随机种子)
2. transformer_bidir_on 3-seed 集成 (按炮对齐)
3. PF6 分相位 R²
4. PF6 的 Pearson r (压缩不敏感) vs R²
5. 其它 split (betanmax / ipmode / IID paper_best) 的 PF6 R²
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_betan.plot_figures import PF_NAMES, _metrics_1d, load_records  # noqa

BASE = Path("results/betan_ablation")
VARIANTS = ["transformer_on", "transformer_off", "transformer_bidir_on",
            "transformer_bidir_off", "lstm_off", "mlp_off"]
SEEDS = ["s11", "s44", "s20260424"]
PF6 = 5
SPLITS = {
    "betan (s44)":       BASE / "transformer_bidir_on" / "transformer_bidir_on_betan_s44" / "per_shot_preds.npz",
    "betanmax":          Path("results/paper_betanmax/predictions/per_shot_preds.npz"),
    "ipmode":            Path("results/paper_ipmode/predictions/per_shot_preds.npz"),
    "IID (paper_best)":  Path("results/paper_best/predictions/per_shot_preds.npz"),
}


def chan_arrays(records, ch):
    ts = np.concatenate([r["target_kA"][:, ch][r["action_mask"][:, ch]] for r in records])
    ps = np.concatenate([r["pred_kA"][:, ch][r["action_mask"][:, ch]] for r in records])
    return ts, ps


def per_channel_r2(records):
    # _metrics_1d(preds, truths): 注意 chan_arrays 返回 (ts, ps), 参数顺序不能反
    return np.array([_metrics_1d(*chan_arrays(records, ch)[::-1])[0] for ch in range(12)])


def pearson_r(ts, ps):
    return float(np.corrcoef(ts, ps)[0, 1])


def main():
    print("=== 1. variant×seed PF6 R² (grid) ===", flush=True)
    best_pf6 = (-9, None)
    for v in VARIANTS:
        for s in SEEDS:
            f = BASE / v / f"{v}_betan_{s}" / "per_shot_preds.npz"
            if not f.exists():
                continue
            r2 = per_channel_r2(load_records(f))
            print(f"  {v:20s} {s:>10s}: PF6={r2[PF6]:.4f} mean={r2.mean():.4f}", flush=True)
            if r2[PF6] > best_pf6[0]:
                best_pf6 = (r2[PF6], f"{v}/{s}")
    print(f"  >>> best PF6 in grid: {best_pf6[0]:.4f}  ({best_pf6[1]})", flush=True)

    # 2. 3-seed ensemble (transformer_bidir_on), 按炮对齐
    print("\n=== 2. transformer_bidir_on 3-seed 集成 ===", flush=True)
    seed_recs = {s: load_records(BASE / "transformer_bidir_on" / f"transformer_bidir_on_betan_{s}" / "per_shot_preds.npz") for s in SEEDS}
    ref = seed_recs["s44"]
    for s in SEEDS:
        assert [r["shot"] for r in seed_recs[s]] == [r["shot"] for r in ref], f"shot order mismatch {s}"
    ens_r2 = np.zeros(12); single_r2 = np.zeros(12)
    for ch in range(12):
        ts = []; ens = []; sng = []
        for i in range(len(ref)):
            m = ref[i]["action_mask"][:, ch]
            ts.append(ref[i]["target_kA"][m, ch])
            sng.append(seed_recs["s44"][i]["pred_kA"][m, ch])
            ens.append(np.mean([seed_recs[s][i]["pred_kA"][m, ch] for s in SEEDS], axis=0))
        ts = np.concatenate(ts); ens = np.concatenate(ens); sng = np.concatenate(sng)
        ens_r2[ch] = _metrics_1d(ens, ts)[0]; single_r2[ch] = _metrics_1d(sng, ts)[0]
    for ch in [2,3,4,5]:
        print(f"  {PF_NAMES[ch]:5s}: single(s44)={single_r2[ch]:.4f}  ensemble={ens_r2[ch]:.4f}  Δ={ens_r2[ch]-single_r2[ch]:+.4f}", flush=True)
    print(f"  >>> PF6 ensemble = {ens_r2[PF6]:.4f}  (s44 single = {single_r2[PF6]:.4f})", flush=True)

    # 3. PF6 分相位 (s44)
    print("\n=== 3. PF6 分相位 R² (betan s44) ===", flush=True)
    recs = load_records(SPLITS["betan (s44)"])
    for pid, pn in [(0,"ramp_up"),(1,"flat_top"),(2,"ramp_down")]:
        ts, ps = [], []
        for r in recs:
            m = r["action_mask"][:, PF6] & (r["phase_ids"] == pid)
            if m.sum(): ts.append(r["target_kA"][m, PF6]); ps.append(r["pred_kA"][m, PF6])
        if ts:
            ts = np.concatenate(ts); ps = np.concatenate(ps)
            print(f"  {pn:10s}: PF6 R²={_metrics_1d(ps,ts)[0]:.4f}  n={ts.size}", flush=True)

    # 4. Pearson r vs R² (s44)
    print("\n=== 4. Pearson r vs R² (betan s44) ===", flush=True)
    for ch in range(12):
        ts, ps = chan_arrays(recs, ch)
        print(f"  {PF_NAMES[ch]:5s} R²={_metrics_1d(ps,ts)[0]:.4f}  r={pearson_r(ts,ps):.4f}" + ("  <--PF6" if ch==PF6 else ""), flush=True)

    # 5. 其它 split 的 PF6
    print("\n=== 5. 各 split 的 PF6 R² / r ===", flush=True)
    for name, f in SPLITS.items():
        if not f.exists():
            print(f"  {name:20s}: (missing {f})", flush=True); continue
        rr = load_records(f)
        ts, ps = chan_arrays(rr, PF6)
        print(f"  {name:20s}: PF6 R²={_metrics_1d(ps,ts)[0]:.4f}  r={pearson_r(ts,ps):.4f}  n={ts.size}", flush=True)

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
