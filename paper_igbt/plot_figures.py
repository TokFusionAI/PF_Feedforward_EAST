"""F1/F2/F3/F9/N2 —— 全部读 per_shot_preds.npz (不重推理), 复用 bc 的 apply_didt_filter。

F1 per_channel_scatter / F2 best_shot_timeseries / F3 loss_curves (s44 + 8-seed val 带) /
F9 fig_filter_comparison (P99.9 dIdt 滤波前后) / N2 分相位误差。
画图函数复制自 scripts_notime/plot_paper_figures.py + plot_filter_comparison.py (避免 import 那两个
顶层 import bc_notime 模型的模块)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bc.common.constants import PCPF_NAMES  # noqa: E402
from bc.data.phases import PHASE_NAMES  # noqa: E402
from bc.evaluation.inference_filter import apply_didt_filter  # noqa: E402 (模型无关)

PF_NAMES = [n.replace("PCPF", "PF") for n in PCPF_NAMES]


def _save(fig, out):
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out} + .pdf")


def _metrics_1d(preds, truths):
    res = preds - truths
    mae = float(np.mean(np.abs(res)))
    mean_t = float(truths.mean())
    r2 = 1.0 - float(np.sum(res ** 2)) / (float(np.sum((truths - mean_t) ** 2)) + 1e-12)
    return r2, mae


def load_records(npz_path: Path) -> list[dict]:
    d = np.load(npz_path, allow_pickle=False)
    shots, T = d["shots"], d["T"]
    N = len(shots)
    recs = []
    for i in range(N):
        Tt = int(T[i])
        pred = d["pred_kA"][i, :Tt].astype(np.float32)
        tgt = d["target_kA"][i, :Tt].astype(np.float32)
        mask = d["action_mask"][i, :Tt]
        diff = pred - tgt
        loss = float((diff ** 2 * mask).sum() / max(mask.sum(), 1))
        recs.append(dict(
            shot=int(shots[i]), T=Tt,
            time=d["time"][i, :Tt].astype(np.float64),
            dt=d["dt"][i, :Tt].astype(np.float32),
            pred_kA=pred, target_kA=tgt, action_mask=mask, loss=loss,
            Ip_A=d["Ip_A"][i, :Tt].astype(np.float32),
            phase_ids=d["phase_ids"][i, :Tt].astype(np.int8),
        ))
    return recs


# ── F1 scatter ──
def plot_scatter(records, out):
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    rng = np.random.default_rng(0)
    for ch in range(12):
        ax = axes[ch // 4, ch % 4]
        preds = np.concatenate([r["pred_kA"][:, ch][r["action_mask"][:, ch]] for r in records])
        truths = np.concatenate([r["target_kA"][:, ch][r["action_mask"][:, ch]] for r in records])
        r2, mae = _metrics_1d(preds, truths)
        if preds.size > 200_000:
            idx = rng.choice(preds.size, 200_000, replace=False)
            preds, truths = preds[idx], truths[idx]
        ax.scatter(truths, preds, s=1, alpha=0.2, rasterized=True)
        lo, hi = float(min(truths.min(), preds.min())), float(max(truths.max(), preds.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=0.8)
        ax.set_title(PF_NAMES[ch])
        ax.grid(alpha=0.3)
        ax.text(0.03, 0.97, f"$R^2$={r2:.3f}\nMAE={mae:.2f} kA", transform=ax.transAxes,
                va="top", fontsize=8, bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray", boxstyle="round,pad=0.3"))
    for ax in axes[-1]:
        ax.set_xlabel("Truth (kA)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Prediction (kA)")
    plt.tight_layout()
    _save(fig, out)


# ── F2 best shot ──
def plot_best_shot(records, out):
    best = min(records, key=lambda r: r["loss"])
    t, pred, truth, mask = best["time"], best["pred_kA"], best["target_kA"], best["action_mask"]
    fig, axes = plt.subplots(4, 3, figsize=(10, 10), sharex=True)
    for ch in range(12):
        ax = axes[ch // 3, ch % 3]
        ax.plot(t, truth[:, ch], "C0", lw=1.0, label="truth")
        ax.plot(t, pred[:, ch], "C3", lw=0.9, alpha=0.85, label="pred")
        m = mask[:, ch]
        r2, _ = _metrics_1d(pred[:, ch][m], truth[:, ch][m])
        ax.set_title(f"{PF_NAMES[ch]}  ($R^2$={r2:.3f})", fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("t (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Current (kA)")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    _save(fig, out)
    print(f"  (best shot {best['shot']}, loss={best['loss']:.4f})")


# ── F3 loss curves (s44 + 8-seed val band) ──
def plot_loss_curves(tb_root: Path, out: Path, tag: str = "transformer_bidir_on_s44"):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    # s44 单曲线
    files = list((tb_root / tag).rglob("events.out.tfevents.*")) if (tb_root / tag).exists() else \
            list(tb_root.rglob("events.out.tfevents.*"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if files:
        ea = EventAccumulator(str(files[0])); ea.Reload()
        for ttag, lab in [("train/loss_epoch", "Train (s44)"), ("val/loss", "Validation (s44)")]:
            if ttag in ea.Tags()["scalars"]:
                evs = ea.Scalars(ttag)
                ax.plot([e.step for e in evs], [e.value for e in evs], lw=1.2, label=lab)
    # 8-seed val 带
    val_curves = []
    for sub in sorted(tb_root.glob("transformer_bidir_on_s*")):
        fs = list(sub.rglob("events.out.tfevents.*"))
        if not fs:
            continue
        e2 = EventAccumulator(str(fs[0])); e2.Reload()
        if "val/loss" in e2.Tags()["scalars"]:
            evs = e2.Scalars("val/loss")
            val_curves.append((np.array([e.step for e in evs]), np.array([e.value for e in evs])))
    if len(val_curves) >= 2:
        ref_x = val_curves[0][0]
        stack = []
        for x, y in val_curves:
            stack.append(np.interp(ref_x, x, y))
        stack = np.array(stack)
        ax.plot(ref_x, stack.mean(0), color="C3", lw=0.8, alpha=0.5)
        ax.fill_between(ref_x, stack.mean(0) - stack.std(0), stack.mean(0) + stack.std(0),
                        color="C3", alpha=0.12, label=f"Validation mean±std (8 seeds)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Masked MSE (normalized)")
    ax.set_title("Train / Validation loss (best model: bidir Transformer, time-sinusoidal PE)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, out)


# ── F9 filter comparison ──
def _score_shot(rec, MIN_DUR=8.0, MIN_CLIP=3, TAIL_FRAC=0.10, MAX_MAE=0.35):
    raw, filt, tgt, mask = rec["pred_kA"], rec["pred_filtered"], rec["target_kA"], rec["action_mask"]
    T = rec["T"]; t = rec["time"]
    if T < 10 or (t[-1] - t[0]) < MIN_DUR:
        return None
    n = max(mask.sum(), 1)
    if (np.abs(raw - tgt) * mask).sum() / n > MAX_MAE:
        return None
    changed = (filt != raw) & mask
    steps = np.any(changed, axis=1)
    if int(steps.sum()) < MIN_CLIP or int(steps[: int(T * (1 - TAIL_FRAC))].sum()) == 0:
        return None
    n_c = max(changed.sum(), 1)
    err_raw = float((np.abs(raw - tgt) * changed).sum() / n_c)
    err_filt = float((np.abs(filt - tgt) * changed).sum() / n_c)
    if err_raw - err_filt <= 0:
        return None
    return {"shot": rec["shot"], "clip_mag": float(np.abs(filt - raw)[changed].mean()),
            "n_ch": int(np.any(changed, axis=0).sum())}


def plot_filter_comparison(records, out):
    print(f"F9: applying P99.9 filter to {len(records)} shots ...")
    for rec in records:
        rec["pred_filtered"] = apply_didt_filter(
            rec["pred_kA"], rec["time"].astype(np.float64), rec["Ip_A"].astype(np.float64),
            which="P99.9", phase_ids=rec.get("phase_ids")).astype(np.float32)
    scored = [(r, _score_shot(r)) for r in records]
    scored = [(r, s) for r, s in scored if s is not None]
    if not scored:
        print("F9: no qualifying shot, skip"); return
    scored.sort(key=lambda rs: rs[1]["clip_mag"], reverse=True)
    rec, sc = scored[0]
    print(f"F9: selected shot {sc['shot']} clip_mag={sc['clip_mag']:.3f} n_ch={sc['n_ch']}")

    from matplotlib.lines import Line2D
    t, tgt = rec["time"], rec["target_kA"]
    raw, filt, mask = rec["pred_kA"], rec["pred_filtered"], rec["action_mask"]
    T = rec["T"]
    clip_ch = [c for c in range(12) if np.any((filt[:, c] != raw[:, c]) & mask[:, c])]

    def _nm(a, m):
        o = a.astype(np.float64).copy(); o[~m] = np.nan; return o
    fig = plt.figure(figsize=(15, 13))
    g = fig.add_gridspec(5, 3, hspace=0.50, wspace=0.32, height_ratios=[1, 1, 1, 1, 1.1])
    for ch in range(12):
        ax = fig.add_subplot(g[ch // 3, ch % 3])
        m = mask[:, ch].astype(bool)
        ax.plot(t, _nm(tgt[:, ch], m), color="#2563EB", lw=1.5, zorder=2)
        ax.plot(t, _nm(raw[:, ch], m), color="#DC2626", lw=1.2, ls="--", zorder=3)
        ax.plot(t, _nm(filt[:, ch], m), color="#16A34A", lw=1.2, zorder=4)
        ch_c = (filt[:, ch] != raw[:, ch]) & m
        if ch_c.any():
            idx = np.where(ch_c)[0]
            ax.axvspan(float(t[max(0, idx[0] - 2)]), float(t[min(T - 1, idx[-1] + 2)]),
                       color="#FF9800", alpha=0.20, zorder=0)
        ax.set_title(PF_NAMES[ch], fontsize=10, fontweight="bold"); ax.grid(alpha=0.25, lw=0.5)
        if ch % 3 == 0: ax.set_ylabel("Current (kA)", fontsize=9)
        if ch // 3 == 3: ax.set_xlabel("Time (s)", fontsize=9)
    if clip_ch:
        show = sorted(clip_ch)
        gz = g[4, :].subgridspec(1, len(show), wspace=0.40)
        for i, ch in enumerate(show):
            ax = fig.add_subplot(gz[0, i])
            m = mask[:, ch].astype(bool)
            ch_c = (filt[:, ch] != raw[:, ch]) & m
            idx = np.where(ch_c)[0]
            lo, hi = max(0, idx[0] - 4), min(T - 1, idx[-1] + 4)
            seg = slice(lo, hi + 1); ms = mask[seg, ch].astype(bool)
            er = (raw[seg, ch] - tgt[seg, ch]).astype(float); ef = (filt[seg, ch] - tgt[seg, ch]).astype(float)
            er[~ms] = np.nan; ef[~ms] = np.nan
            ax.axhline(0, color="#ADB5BD", lw=1.0, alpha=0.8)
            ax.plot(t[seg], er, color="#DC2626", lw=2.0, ls="--", label="raw error" if i == 0 else None)
            ax.plot(t[seg], ef, color="#16A34A", lw=2.0, label="filtered error" if i == 0 else None)
            sc_ = ch_c[seg] & ms
            if sc_.any():
                ax.scatter(t[seg][sc_], er[sc_], color="#FF9800", s=55, marker="v", edgecolors="k", linewidths=0.5)
                ax.scatter(t[seg][sc_], ef[sc_], color="#FF9800", s=55, marker="^", edgecolors="k", linewidths=0.5)
                ax.fill_between(t[seg], er, ef, where=sc_, alpha=0.25, color="#FF9800")
            ax.set_title(PF_NAMES[ch], fontsize=9, fontweight="bold"); ax.grid(alpha=0.25, lw=0.5)
            ax.set_facecolor("#FFFDE7")
            if i == 0: ax.set_ylabel("Pred $-$ Truth (kA)", fontsize=9)
            ax.set_xlabel("Time (s)", fontsize=8)
    handles = [Line2D([0], [0], color="#2563EB", lw=1.5, label="PCS (truth)"),
               Line2D([0], [0], color="#DC2626", lw=1.2, ls="--", label="BC pred (raw)"),
               Line2D([0], [0], color="#16A34A", lw=1.2, label="BC pred (P99.9 filtered)")]
    fig.subplots_adjust(top=0.93, bottom=0.05, left=0.06, right=0.97)
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.515, 0.975))
    _save(fig, out)


# ── N2 per-phase error ──
def plot_per_phase(records, out):
    se = np.zeros((3, 12)); cnt = np.zeros((3, 12)); st = np.zeros((3, 12)); st2 = np.zeros((3, 12))
    for r in records:
        m = r["action_mask"]; p = r["pred_kA"]; t = r["target_kA"]; pid = r["phase_ids"]
        phx = pid[:, None]
        for k in range(3):
            pm = m * (phx == k)
            se[k] += ((p - t) ** 2 * pm).sum((0, 1))
            cnt[k] += pm.sum((0, 1))
            st[k] += (t * pm).sum((0, 1)); st2[k] += ((t * t) * pm).sum((0, 1))
    c = np.maximum(cnt, 1)
    mse = se / c; mean_t = st / c; ss_tot = np.maximum(st2 - cnt * mean_t ** 2, 1e-12); r2 = 1 - se / ss_tot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(12)
    for ax, vals, lab in [(axes[0], mse, "MSE [kA²]"), (axes[1], r2, "R²")]:
        w = 0.27
        for i, ph in enumerate(PHASE_NAMES):
            ax.bar(x + (i - 1) * w, vals[i], w, label=ph)
        ax.set_xticks(x); ax.set_xticklabels(PF_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(lab); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    axes[0].set_title("Per-phase per-channel MSE"); axes[1].set_title("Per-phase per-channel R²")
    plt.tight_layout()
    _save(fig, out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", default="results/paper_igbt/predictions/per_shot_preds.npz")
    ap.add_argument("--tb-root", default="results/ablation/transformer_bidir_on/tb")
    ap.add_argument("--out-dir", default="results/paper_igbt/figures")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    recs = load_records(Path(args.preds))
    print(f"loaded {len(recs)} records")
    plot_scatter(recs, out / "per_channel_scatter.png")          # F1
    plot_best_shot(recs, out / "best_shot_timeseries.png")       # F2
    plot_loss_curves(Path(args.tb_root), out / "loss_curves.png")  # F3
    plot_filter_comparison(recs, out / "fig_filter_comparison.png")  # F9
    plot_per_phase(recs, out / "per_phase_error.png")            # N2
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
