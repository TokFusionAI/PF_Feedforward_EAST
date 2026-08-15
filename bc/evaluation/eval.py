"""测试集离线评估 + 全套结果图.

跑在计算节点 (单卡足够; 不需 DDP).

Usage:
    python -m bc.eval --config configs/bc_v1.yaml \
        --ckpt results/bc_v1/run1/checkpoints/best_val.pt \
        --split test \
        --out-dir results/bc_v1/run1/figures

产出 (在 --out-dir 下):
    metrics_summary.json            聚合数值
    per_shot_records.npz            每炮 (T, 12) pred / target / mask, 反归一化 kA
    loss_curves.png                 (从 TB events 读)
    per_channel_mse_bar.png         12 通道 MSE bar
    per_channel_scatter.png         4x3 网格 pred vs truth
    per_channel_r2.png              R^2 bar + 0.8 红线
    time_series_5shots.png          5 炮 12 通道叠图
    dIdt_distribution.png           pred / truth |dI/dt| 直方图
    physical_violation.csv          物理违规率
    error_vs_T_bin.png              loss by T bin
    error_vs_dt_bin.png             loss by dt_median bin (验证 time-aware PE)
    worst_10_shots.png              loss top-10 炮叠图
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from bc.common.constants import (
    DIDT_P99_KAPS,
    DIDT_P999_KAPS,
    DT_BUCKETS,
    HARD_LIMIT_DIDT_KAPS,
    HARD_LIMIT_I_KA,
    PCPF_NAMES,
    T_MAX,
    phase_limits_array,
)
from bc.data.dataset import PFDataset, load_norm_stats, read_shots_txt
from bc.data.phases import PHASE_NAMES, phase_ids_per_step, phase_step_ids
from bc.evaluation.inference_filter import apply_didt_filter
from bc.models.model import CausalTransformer
from bc.training.utils import deep_get, load_yaml


# ----------------------------- inference ----------------------------- #


@torch.no_grad()
def predict_split(
    model,
    shots: list[int],
    cfg: dict,
    norm_stats: dict[str, np.ndarray],
    device: str,
) -> list[dict]:
    """整炮 forward, 返回 list of per-shot dicts (反归一化到 kA).

    Each record also includes Ip_A (from sample state[18] denormalized) for phase
    detection. phase_step (T-1,) int8 is pre-computed for efficient downstream
    violation rate calculation.
    """
    model.eval()
    ds = PFDataset(
        shots,
        dataset_root=deep_get(cfg, "data.dataset_root"),
        norm_stats_path=deep_get(cfg, "data.norm_stats"),
        T_max=deep_get(cfg, "data.T_max", T_MAX),
    )
    a_mean = torch.as_tensor(norm_stats["action_mean"], device=device)
    a_std = torch.as_tensor(norm_stats["action_std"], device=device)

    # state channel 18 is PCRL01 (Ip, A); we denormalize it back to plot & detect phases
    state_mean_cpu = norm_stats["state_mean"]
    state_std_cpu = norm_stats["state_std"]
    ip_mean = float(state_mean_cpu[18])
    ip_std = float(state_std_cpu[18])

    records: list[dict] = []
    for i, shot in enumerate(shots):
        sample = ds[i]
        state = sample["state"].unsqueeze(0).to(device)
        time_phys = sample["time_phys"].unsqueeze(0).to(device)
        token_mask = sample["token_mask"].unsqueeze(0).to(device)
        action = sample["action"].unsqueeze(0).to(device)
        action_mask = sample["action_mask"].unsqueeze(0).to(device)

        pred = model(state, time_phys, token_mask)            # (1, T, 12) normalized
        pred_kA = (pred * a_std + a_mean) / 1e3                # to kA
        target_kA = (action * a_std + a_mean) / 1e3

        T_eff = int(sample["T"])
        time_np = sample["time_phys"][:T_eff].cpu().numpy().astype(np.float64)
        # recover Ip in Amperes from normalized state channel 18
        ip_norm = sample["state"][:T_eff, 18].cpu().numpy().astype(np.float64)
        Ip_A = ip_norm * ip_std + ip_mean
        pid = phase_ids_per_step(time_np, Ip_A, valid_len=T_eff)  # (T_eff,)
        step_pid = phase_step_ids(pid)                            # (T_eff-1,)

        records.append({
            "shot": int(shot),
            "T": T_eff,
            "time": time_np.astype(np.float32),
            "dt": sample["dt_phys"][:T_eff].cpu().numpy().astype(np.float32),
            "pred_kA": pred_kA[0, :T_eff].cpu().numpy().astype(np.float32),
            "target_kA": target_kA[0, :T_eff].cpu().numpy().astype(np.float32),
            "action_mask": action_mask[0, :T_eff].cpu().numpy(),
            "Ip_A": Ip_A.astype(np.float32),
            "phase_ids": pid.astype(np.int8),
            "step_phase_ids": step_pid.astype(np.int8),
        })
        if (i + 1) % 200 == 0:
            print(f"  predicted {i + 1}/{len(shots)} shots")
    return records


# ----------------------------- aggregate metrics ----------------------------- #


# phase-aware threshold tables 在模块加载时建一次 (若 JSON 不存在返回 empty)
_PHASE_MAX_LIMITS = phase_limits_array(which="max")        # {phase: [12]}
_PHASE_P9999_LIMITS = phase_limits_array(which="P99.99")
_PHASE_P999_LIMITS = phase_limits_array(which="P99.9")
_PHASE_P99_LIMITS = phase_limits_array(which="P99")


def _phase_limits_matrix(which: str) -> np.ndarray:
    """Return (3, 12) float ndarray indexed by PHASE_RAMP_UP/FLAT_TOP/RAMP_DOWN."""
    src = {
        "max": _PHASE_MAX_LIMITS,
        "P99.99": _PHASE_P9999_LIMITS,
        "P99.9": _PHASE_P999_LIMITS,
        "P99": _PHASE_P99_LIMITS,
    }[which]
    arr = np.full((3, 12), np.inf, dtype=np.float64)
    for pid, name in enumerate(PHASE_NAMES):
        if name in src:
            arr[pid] = np.asarray(src[name], dtype=np.float64)
    return arr


def per_shot_metrics(rec: dict) -> dict:
    """compute per-shot per-channel MSE/MAE in kA + dIdt rates.

    dI/dt 违规率同时给出:
      - 老 4 kA/s 一刀切 (pred_didt_violation_rate / truth_*)
      - per-channel P99 全炮 (pred_didt_violation_rate_P99 / truth_*)
      - per-channel P99.9 全炮 (pred_didt_violation_rate_P99_9 / truth_*)
      - per-phase per-channel max (_PHASE)
      - per-phase per-channel P99.9 (_PHASE_P99_9)
      - per-phase per-channel P99   (_PHASE_P99)
    """
    pred = rec["pred_kA"]
    tgt = rec["target_kA"]
    mask = rec["action_mask"]
    T = rec["T"]
    if T < 2:
        return {"shot": rec["shot"], "T": T, "dt_median": float("nan")}
    diff = pred - tgt
    mse_per_ch = ((diff ** 2 * mask).sum(axis=0) / np.maximum(mask.sum(axis=0), 1)).astype(np.float64)
    mae_per_ch = ((np.abs(diff) * mask).sum(axis=0) / np.maximum(mask.sum(axis=0), 1)).astype(np.float64)

    # dI/dt 物理违规率: 用真实 dt
    dt = rec["dt"]
    didt_pred = (pred[1:] - pred[:-1]) / np.maximum(dt[1:, None], 1e-6)
    didt_truth = (tgt[1:] - tgt[:-1]) / np.maximum(dt[1:, None], 1e-6)
    mask_didt = mask[1:] & mask[:-1]
    n_step = max(int(mask_didt.sum()), 1)
    n_step_per_ch = np.maximum(mask_didt.sum(axis=0), 1)  # (12,)
    pred_amp_v = float((np.abs(pred) > HARD_LIMIT_I_KA).sum() / max(int(mask.sum()), 1))

    # 多套阈值并存: 老 4 kA/s 一刀切, 以及 per-channel P99 / P99.9 数据驱动值
    flat_thr_4 = np.abs(didt_pred) > HARD_LIMIT_DIDT_KAPS
    flat_thr_4_truth = np.abs(didt_truth) > HARD_LIMIT_DIDT_KAPS
    pc_p99 = np.asarray([DIDT_P99_KAPS[n] for n in PCPF_NAMES])[None, :]
    pc_p999 = np.asarray([DIDT_P999_KAPS[n] for n in PCPF_NAMES])[None, :]
    pred_v_p99 = np.abs(didt_pred) > pc_p99
    pred_v_p999 = np.abs(didt_pred) > pc_p999
    truth_v_p99 = np.abs(didt_truth) > pc_p99
    truth_v_p999 = np.abs(didt_truth) > pc_p999

    def _rate(bad: np.ndarray) -> float:
        return float((bad & mask_didt).sum() / n_step)

    def _rate_pc(bad: np.ndarray) -> list[float]:
        return ((bad & mask_didt).sum(axis=0) / n_step_per_ch).astype(float).tolist()

    # ---- phase-aware violations ------------------------------------------
    step_pid = rec.get("step_phase_ids")  # (T-1,) int8, -1 if invalid
    out: dict = {
        "shot": rec["shot"],
        "T": T,
        "dt_median": float(np.median(dt[1:])),
        "mse_per_ch": mse_per_ch.tolist(),
        "mae_per_ch": mae_per_ch.tolist(),
        "loss": float(mse_per_ch.mean()),
        "pred_amp_violation_rate": pred_amp_v,
        # old one-size-fits-all 4 kA/s
        "pred_didt_violation_rate": _rate(flat_thr_4),
        "truth_didt_violation_rate": _rate(flat_thr_4_truth),
        # data-driven per-channel P99 / P99.9
        "pred_didt_violation_rate_P99": _rate(pred_v_p99),
        "truth_didt_violation_rate_P99": _rate(truth_v_p99),
        "pred_didt_violation_rate_P99_9": _rate(pred_v_p999),
        "truth_didt_violation_rate_P99_9": _rate(truth_v_p999),
        "pred_didt_violation_rate_P99_per_ch": _rate_pc(pred_v_p99),
    }

    if step_pid is not None and step_pid.size == didt_pred.shape[0]:
        # (T-1, 12) phase threshold matrix for this shot
        step_valid = step_pid >= 0
        out["n_steps_phase"] = {
            name: int((step_pid == pid).sum()) for pid, name in enumerate(PHASE_NAMES)
        }
        for label in ("max", "P99.9", "P99"):
            limits_mat = _phase_limits_matrix(label)  # (3, 12)
            # broadcast: for each step, use limits_mat[step_pid[k]] (invalid -> inf)
            safe_pid = np.where(step_valid, step_pid, 0)
            thr_per_step = limits_mat[safe_pid]  # (T-1, 12)
            bad_pred = (np.abs(didt_pred) > thr_per_step) & step_valid[:, None]
            bad_truth = (np.abs(didt_truth) > thr_per_step) & step_valid[:, None]
            slug = label.replace(".", "_")
            out[f"pred_didt_violation_rate_PHASE_{slug}"] = _rate(bad_pred)
            out[f"truth_didt_violation_rate_PHASE_{slug}"] = _rate(bad_truth)
            # per-phase breakdown (3 phases × 1 number each)
            for pid, name in enumerate(PHASE_NAMES):
                in_phase = (step_pid == pid)[:, None]
                n_ph = max(int((mask_didt & in_phase).sum()), 1)
                r_pred = float((bad_pred & in_phase).sum() / n_ph)
                r_truth = float((bad_truth & in_phase).sum() / n_ph)
                out[f"pred_didt_violation_rate_PHASE_{slug}_{name}"] = r_pred
                out[f"truth_didt_violation_rate_PHASE_{slug}_{name}"] = r_truth
    return out


def aggregate_r2(records: list[dict]) -> np.ndarray:
    """全集 per-channel R^2 (in kA domain)."""
    sum_n = np.zeros(12)
    sum_t = np.zeros(12)
    sum_t2 = np.zeros(12)
    ss_res = np.zeros(12)
    for r in records:
        m = r["action_mask"]
        t = r["target_kA"]
        p = r["pred_kA"]
        sum_n += m.sum(axis=0)
        sum_t += (t * m).sum(axis=0)
        sum_t2 += ((t * m) ** 2).sum(axis=0)
        ss_res += ((p - t) ** 2 * m).sum(axis=0)
    sum_n = np.maximum(sum_n, 1)
    mean = sum_t / sum_n
    ss_tot = sum_t2 - sum_n * mean ** 2
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


# ----------------------------- plots ----------------------------- #


def plot_per_channel_mse(per_shot: list[dict], out: Path) -> None:
    arr = np.array([r["mse_per_ch"] for r in per_shot if "mse_per_ch" in r])  # (N, 12)
    med = np.median(arr, axis=0)
    p25, p75 = np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(12)
    ax.bar(x, med, yerr=[med - p25, p75 - med], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(PCPF_NAMES, rotation=45)
    ax.set_ylabel("MSE [kA^2]")
    ax.set_title("Per-channel MSE (median, P25-P75 over test shots)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close(fig)


def plot_per_channel_r2(r2: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(12)
    ax.bar(x, r2, color=["C2" if v >= 0.8 else "C3" for v in r2])
    ax.axhline(0.8, color="r", linestyle="--", label="R^2 = 0.8")
    ax.set_xticks(x)
    ax.set_xticklabels(PCPF_NAMES, rotation=45)
    ax.set_ylabel("R^2")
    ax.set_title(f"Per-channel R^2 on test set (median = {np.median(r2):.3f})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close(fig)


def _metrics_1d(preds: np.ndarray, truths: np.ndarray) -> tuple[float, float, float]:
    """Return (R^2, MAE, RMSE)."""
    residual = preds - truths
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mean_t = float(truths.mean())
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((truths - mean_t) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot
    return r2, mae, rmse


def plot_per_channel_scatter(records: list[dict], out: Path, max_pts: int = 200_000) -> None:
    """12 子图 pred-vs-truth 散点 + 每图标 R²/MAE."""
    _display_names = [n.replace("PCPF", "PF") for n in PCPF_NAMES]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    rng = np.random.default_rng(0)
    for ch in range(12):
        ax = axes[ch // 4, ch % 4]
        preds = np.concatenate([r["pred_kA"][:, ch][r["action_mask"][:, ch]] for r in records])
        truths = np.concatenate([r["target_kA"][:, ch][r["action_mask"][:, ch]] for r in records])
        r2, mae, rmse = _metrics_1d(preds, truths)
        if preds.size > max_pts:
            idx = rng.choice(preds.size, size=max_pts, replace=False)
            preds_plot, truths_plot = preds[idx], truths[idx]
        else:
            preds_plot, truths_plot = preds, truths
        ax.scatter(truths_plot, preds_plot, s=1, alpha=0.2, rasterized=True)
        lo = float(min(truths.min(), preds.min()))
        hi = float(max(truths.max(), preds.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=0.8)
        ax.set_xlabel("truth [kA]"); ax.set_ylabel("pred [kA]")
        ax.set_title(_display_names[ch])
        ax.grid(alpha=0.3)
        ax.text(
            0.03, 0.97,
            f"$R^2$={r2:.3f}\nMAE={mae:.2f} kA",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray", boxstyle="round,pad=0.3"),
        )
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close(fig)


def plot_overall_scatter(records: list[dict], out: Path) -> None:
    """12 通道合并 hexbin 散点 + 全局 R²/MAE/RMSE."""
    preds_all, truths_all = [], []
    for r in records:
        m = r["action_mask"]
        preds_all.append(r["pred_kA"][m])
        truths_all.append(r["target_kA"][m])
    preds = np.concatenate(preds_all)
    truths = np.concatenate(truths_all)
    r2, mae, rmse = _metrics_1d(preds, truths)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    h = ax.hexbin(truths, preds, gridsize=80, cmap="viridis", bins="log", mincnt=1)
    lo = float(min(truths.min(), preds.min()))
    hi = float(max(truths.max(), preds.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.0, label="y = x")
    ax.set_xlabel("truth [kA]"); ax.set_ylabel("pred [kA]")
    ax.set_title(
        f"Overall pred vs truth (all 12 PCPF channels combined, "
        f"{preds.size:,} points)\n"
        f"R² = {r2:.4f}    MAE = {mae:.3f} kA    RMSE = {rmse:.3f} kA"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.colorbar(h, ax=ax, label="count (log scale)", shrink=0.9)
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close(fig)


def plot_residual_distribution(records: list[dict], out: Path) -> None:
    """12 通道残差直方图 + μ/σ/|res|_P99."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    for ch in range(12):
        ax = axes[ch // 4, ch % 4]
        residuals = []
        for r in records:
            m = r["action_mask"][:, ch]
            residuals.append(r["pred_kA"][:, ch][m] - r["target_kA"][:, ch][m])
        res = np.concatenate(residuals)
        if res.size == 0:
            ax.set_title(PCPF_NAMES[ch]); continue
        mean = float(res.mean()); std = float(res.std())
        p99 = float(np.quantile(np.abs(res), 0.99))
        # 截断到 ±4σ 的范围作图防止长尾压扁主峰
        xlim = 4 * std if std > 0 else 1.0
        bins = np.linspace(-xlim, xlim, 120)
        ax.hist(res, bins=bins, alpha=0.75, color="C0", density=True)
        ax.axvline(0, color="k", ls="--", lw=0.6)
        ax.axvline(mean, color="C3", ls="-", lw=0.6, label=f"μ={mean:+.3f}")
        ax.text(
            0.03, 0.97,
            f"μ = {mean:+.3f}\nσ = {std:.3f}\n|res|_P99 = {p99:.2f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray", boxstyle="round,pad=0.3"),
        )
        ax.set_title(PCPF_NAMES[ch], fontsize=9)
        ax.set_xlabel("residual = pred - truth [kA]")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close(fig)


def plot_error_vs_magnitude(records: list[dict], out: Path, n_bins: int = 20) -> None:
    """误差随 |truth| 分位 bin 变化, 查 large/small current 处的模型稳定性."""
    preds_all, truths_all = [], []
    for r in records:
        m = r["action_mask"]
        preds_all.append(r["pred_kA"][m])
        truths_all.append(r["target_kA"][m])
    preds = np.concatenate(preds_all)
    truths = np.concatenate(truths_all)
    abs_truth = np.abs(truths)
    if abs_truth.size < n_bins * 100:
        return
    edges = np.quantile(abs_truth, np.linspace(0, 1, n_bins + 1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    mae_b: list[float] = []; rmse_b: list[float] = []; n_b: list[int] = []
    for i in range(n_bins):
        mb = (abs_truth >= edges[i]) & (
            abs_truth < edges[i + 1] if i < n_bins - 1 else abs_truth <= edges[i + 1]
        )
        if int(mb.sum()) > 10:
            r = preds[mb] - truths[mb]
            mae_b.append(float(np.mean(np.abs(r))))
            rmse_b.append(float(np.sqrt(np.mean(r ** 2))))
        else:
            mae_b.append(float("nan")); rmse_b.append(float("nan"))
        n_b.append(int(mb.sum()))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(centers, mae_b, "o-", label="MAE", lw=1.0, markersize=4)
    ax.plot(centers, rmse_b, "s-", label="RMSE", lw=1.0, markersize=4)
    ax.set_xlabel("|truth| [kA] (quantile-bin center)")
    ax.set_ylabel("error [kA]")
    ax.set_title(
        f"Prediction error vs truth magnitude "
        f"({n_bins} quantile bins, ≈{int(np.mean(n_b)):,} pts/bin)"
    )
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close(fig)


def plot_time_series_5shots(records: list[dict], out: Path, n_show: int = 5) -> None:
    rng = np.random.default_rng(20260424)
    if len(records) > n_show:
        idx = rng.choice(len(records), size=n_show, replace=False)
    else:
        idx = np.arange(len(records))
    show = [records[i] for i in idx]

    fig, axes = plt.subplots(12, n_show, figsize=(3.0 * n_show, 1.4 * 12), sharex="col")
    for j, rec in enumerate(show):
        for ch in range(12):
            ax = axes[ch, j]
            t = rec["time"]
            ax.plot(t, rec["target_kA"][:, ch], color="C0", lw=0.8, label="truth" if ch == 0 and j == 0 else None)
            ax.plot(t, rec["pred_kA"][:, ch], color="C3", lw=0.8, label="pred" if ch == 0 and j == 0 else None)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(PCPF_NAMES[ch], fontsize=8)
            if ch == 0:
                ax.set_title(f"shot {rec['shot']}", fontsize=9)
        axes[-1, j].set_xlabel("time [s]")
    fig.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close(fig)


def plot_didt_distribution(records: list[dict], out: Path) -> None:
    pred_didt = []
    truth_didt = []
    for r in records:
        if r["T"] < 2:
            continue
        dt = np.maximum(r["dt"][1:], 1e-6)[:, None]
        m = r["action_mask"][1:] & r["action_mask"][:-1]
        pred_didt.append(((r["pred_kA"][1:] - r["pred_kA"][:-1]) / dt)[m])
        truth_didt.append(((r["target_kA"][1:] - r["target_kA"][:-1]) / dt)[m])
    pred_didt = np.concatenate(pred_didt)
    truth_didt = np.concatenate(truth_didt)
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(-15, 15, 121)
    ax.hist(truth_didt, bins=bins, alpha=0.4, color="C0", label="truth", density=True)
    ax.hist(pred_didt, bins=bins, alpha=0.4, color="C3", label="pred", density=True)
    ax.axvline(HARD_LIMIT_DIDT_KAPS, color="k", linestyle="--", lw=0.5)
    ax.axvline(-HARD_LIMIT_DIDT_KAPS, color="k", linestyle="--", lw=0.5)
    ax.set_xlabel("dI/dt [kA/s]")
    ax.set_ylabel("density")
    ax.set_title("dI/dt distribution (pred vs truth, all PCPF, all test steps)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close(fig)


def plot_error_by_bin(per_shot: list[dict], out: Path, key: str, bins: list[tuple[float, float]],
                      title: str, xlabel: str) -> None:
    arr = np.array([r[key] for r in per_shot])
    losses = np.array([r["loss"] for r in per_shot])
    fig, ax = plt.subplots(figsize=(8, 4))
    means, p25s, p75s, ns = [], [], [], []
    labels = []
    for lo, hi in bins:
        m = (arr >= lo) & (arr < hi)
        if m.sum() == 0:
            continue
        L = losses[m]
        means.append(np.median(L))
        p25s.append(np.percentile(L, 25))
        p75s.append(np.percentile(L, 75))
        ns.append(int(m.sum()))
        labels.append(f"[{lo:.2f},{hi:.2f})\nn={int(m.sum())}")
    x = np.arange(len(labels))
    ax.bar(x, means,
           yerr=[np.array(means) - np.array(p25s), np.array(p75s) - np.array(means)],
           capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("loss [kA^2]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close(fig)


def plot_worst_10_shots(records: list[dict], per_shot: list[dict], out: Path) -> None:
    losses = np.array([r["loss"] for r in per_shot])
    order = np.argsort(losses)[::-1][:10]
    show = [records[i] for i in order]
    n_show = len(show)
    fig, axes = plt.subplots(12, n_show, figsize=(2.4 * n_show, 1.2 * 12), sharex="col")
    for j, rec in enumerate(show):
        for ch in range(12):
            ax = axes[ch, j] if n_show > 1 else axes[ch]
            t = rec["time"]
            ax.plot(t, rec["target_kA"][:, ch], color="C0", lw=0.7)
            ax.plot(t, rec["pred_kA"][:, ch], color="C3", lw=0.7)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(PCPF_NAMES[ch], fontsize=7)
            if ch == 0:
                ax.set_title(f"shot {rec['shot']} loss={losses[order[j]]:.2f}", fontsize=8)
        if n_show > 1:
            axes[-1, j].set_xlabel("time [s]")
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close(fig)


def plot_loss_curves(tb_dir: Path, out: Path) -> None:
    """Read TensorBoard events and plot train/val loss vs epoch."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard not available, skip loss_curves.png")
        return
    files = list(Path(tb_dir).rglob("events.out.tfevents.*"))
    if not files:
        print(f"no TB events in {tb_dir}")
        return
    ea = EventAccumulator(str(files[0]))
    ea.Reload()

    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, label in [("train/loss_epoch", "train"), ("val/loss", "val")]:
        if tag in ea.Tags()["scalars"]:
            evs = ea.Scalars(tag)
            xs = [e.step for e in evs]
            ys = [e.value for e in evs]
            ax.plot(xs, ys, label=label, marker="o", ms=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked MSE (normalized)")
    ax.set_title("Train / Val loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close(fig)


# ----------------------------- main ----------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--tb-dir", type=Path, default=None,
                    help="optional: TB log dir for loss_curves.png")
    ap.add_argument(
        "--apply-filter",
        type=str,
        default="none",
        choices=["none", "max", "P99.99", "P99.9", "P99"],
        help=("post-inference phase-aware |dI/dt| hard filter; clip pred_kA so that "
              "per-step |dI/dt| never exceeds the chosen per-phase per-channel threshold. "
              "'none' keeps raw predictions."),
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    shots_path = deep_get(cfg, f"data.{args.split}_shots")
    shots = read_shots_txt(shots_path)
    if args.limit:
        shots = shots[: args.limit]
    print(f"{args.split} shots: {len(shots)}  (from {shots_path})")

    device = args.device if torch.cuda.is_available() else "cpu"
    norm_stats = load_norm_stats(deep_get(cfg, "data.norm_stats"))

    model = CausalTransformer(**cfg["model"]).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    print(f"loaded ckpt {args.ckpt} (epoch={ck.get('extra', {}).get('epoch')}, "
          f"val_loss={ck.get('extra', {}).get('val_loss')})")

    t0 = time.monotonic()
    records = predict_split(model, shots, cfg, norm_stats, device)
    print(f"forward done in {time.monotonic()-t0:.1f}s, {len(records)} records")

    # post-inference hard filter (optional)
    if args.apply_filter != "none":
        print(f"\napplying dI/dt hard filter: which={args.apply_filter}")
        n_clipped_total = 0
        for rec in records:
            pred_raw = rec["pred_kA"]
            pred_filt = apply_didt_filter(
                pred_raw,
                rec["time"].astype(np.float64),
                rec["Ip_A"].astype(np.float64),
                which=args.apply_filter,
                phase_ids=rec.get("phase_ids"),
            )
            rec["pred_kA_unfiltered"] = pred_raw
            rec["pred_kA"] = pred_filt.astype(np.float32)
            n_clipped_total += int(
                ((pred_filt - pred_raw) != 0).any(axis=1).sum()
            )
        print(f"filter done: {n_clipped_total} total (shot, step) positions modified")

    # per-shot metrics
    per_shot = [per_shot_metrics(r) for r in records if r["T"] >= 2]
    r2 = aggregate_r2(records)

    def _mean_field(key: str) -> float | None:
        vals = [r.get(key) for r in per_shot if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    # 全局 overall 指标 (12 通道合并, mask 后)
    _preds_all = np.concatenate([r["pred_kA"][r["action_mask"]] for r in records])
    _truths_all = np.concatenate([r["target_kA"][r["action_mask"]] for r in records])
    overall_r2, overall_mae, overall_rmse = _metrics_1d(_preds_all, _truths_all)

    summary = {
        "n_test_shots": len(per_shot),
        "apply_filter": args.apply_filter,
        "loss_kA2_median": float(np.median([r["loss"] for r in per_shot])),
        "loss_kA2_mean": float(np.mean([r["loss"] for r in per_shot])),
        "overall_r2": overall_r2,
        "overall_mae_kA": overall_mae,
        "overall_rmse_kA": overall_rmse,
        "r2_per_channel": r2.tolist(),
        "r2_median": float(np.median(r2)),
        "amp_violation_rate_mean": float(np.mean([r["pred_amp_violation_rate"] for r in per_shot])),
        # old 4 kA/s one-size-fits-all
        "didt_violation_rate_pred_mean": _mean_field("pred_didt_violation_rate"),
        "didt_violation_rate_truth_mean": _mean_field("truth_didt_violation_rate"),
        # data-driven per-channel P99 / P99.9 (全炮,不分 phase)
        "didt_violation_rate_pred_P99_mean": _mean_field("pred_didt_violation_rate_P99"),
        "didt_violation_rate_truth_P99_mean": _mean_field("truth_didt_violation_rate_P99"),
        "didt_violation_rate_pred_P99_9_mean": _mean_field("pred_didt_violation_rate_P99_9"),
        "didt_violation_rate_truth_P99_9_mean": _mean_field("truth_didt_violation_rate_P99_9"),
        "didt_violation_rate_pred_P99_per_ch": (
            np.stack([np.asarray(r["pred_didt_violation_rate_P99_per_ch"]) for r in per_shot])
            .mean(axis=0).tolist()
        ),
        # phase-aware violations (max / P99.9 / P99 per phase)
        "didt_violation_rate_PHASE_max": {
            "pred_overall": _mean_field("pred_didt_violation_rate_PHASE_max"),
            "truth_overall": _mean_field("truth_didt_violation_rate_PHASE_max"),
            "pred_per_phase": {
                name: _mean_field(f"pred_didt_violation_rate_PHASE_max_{name}")
                for name in PHASE_NAMES
            },
            "truth_per_phase": {
                name: _mean_field(f"truth_didt_violation_rate_PHASE_max_{name}")
                for name in PHASE_NAMES
            },
        },
        "didt_violation_rate_PHASE_P99_9": {
            "pred_overall": _mean_field("pred_didt_violation_rate_PHASE_P99_9"),
            "truth_overall": _mean_field("truth_didt_violation_rate_PHASE_P99_9"),
            "pred_per_phase": {
                name: _mean_field(f"pred_didt_violation_rate_PHASE_P99_9_{name}")
                for name in PHASE_NAMES
            },
            "truth_per_phase": {
                name: _mean_field(f"truth_didt_violation_rate_PHASE_P99_9_{name}")
                for name in PHASE_NAMES
            },
        },
        "didt_violation_rate_PHASE_P99": {
            "pred_overall": _mean_field("pred_didt_violation_rate_PHASE_P99"),
            "truth_overall": _mean_field("truth_didt_violation_rate_PHASE_P99"),
            "pred_per_phase": {
                name: _mean_field(f"pred_didt_violation_rate_PHASE_P99_{name}")
                for name in PHASE_NAMES
            },
            "truth_per_phase": {
                name: _mean_field(f"truth_didt_violation_rate_PHASE_P99_{name}")
                for name in PHASE_NAMES
            },
        },
    }
    # dt-bucket sub-summary
    dt_buckets_summary = []
    for lo, hi in DT_BUCKETS:
        sub = [r for r in per_shot if lo <= r["dt_median"] < hi]
        if not sub:
            continue
        sub_r2 = aggregate_r2([records[i] for i, r in enumerate(per_shot) if lo <= r["dt_median"] < hi])
        dt_buckets_summary.append({
            "bucket": f"[{lo:.2f},{hi:.2f})",
            "n_shots": len(sub),
            "loss_median": float(np.median([r["loss"] for r in sub])),
            "r2_median": float(np.median(sub_r2)),
        })
    summary["dt_buckets"] = dt_buckets_summary
    (args.out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"metrics summary: {json.dumps(summary, indent=2)}")

    # csv of physical violations
    vio_cols = [
        "shot", "T", "dt_median", "loss", "pred_amp_violation_rate",
        "pred_didt_violation_rate", "truth_didt_violation_rate",
        "pred_didt_violation_rate_P99", "truth_didt_violation_rate_P99",
        "pred_didt_violation_rate_P99_9", "truth_didt_violation_rate_P99_9",
        "pred_didt_violation_rate_PHASE_max", "truth_didt_violation_rate_PHASE_max",
        "pred_didt_violation_rate_PHASE_P99_9", "truth_didt_violation_rate_PHASE_P99_9",
        "pred_didt_violation_rate_PHASE_P99", "truth_didt_violation_rate_PHASE_P99",
    ]
    df_phys = pd.DataFrame(per_shot)
    vio_cols = [c for c in vio_cols if c in df_phys.columns]
    df_phys[vio_cols].to_csv(args.out_dir / "physical_violation.csv", index=False)

    # save raw records (compressed npz, one per shot is impractical -> single npz)
    np.savez_compressed(
        args.out_dir / "per_shot_records.npz",
        shots=np.array([r["shot"] for r in records]),
        T=np.array([r["T"] for r in records]),
    )

    # plots
    plot_per_channel_mse(per_shot, args.out_dir / "per_channel_mse_bar.png")
    plot_per_channel_r2(r2, args.out_dir / "per_channel_r2.png")
    plot_per_channel_scatter(records, args.out_dir / "per_channel_scatter.png")
    plot_overall_scatter(records, args.out_dir / "overall_scatter.png")
    plot_residual_distribution(records, args.out_dir / "residual_distribution.png")
    plot_error_vs_magnitude(records, args.out_dir / "error_vs_magnitude.png")
    plot_time_series_5shots(records, args.out_dir / "time_series_5shots.png")
    plot_didt_distribution(records, args.out_dir / "dIdt_distribution.png")
    plot_error_by_bin(per_shot, args.out_dir / "error_vs_T_bin.png", "T",
                      [(20, 60), (60, 100), (100, 128)],
                      "Loss by sequence length T", "T (per-shot effective length)")
    plot_error_by_bin(per_shot, args.out_dir / "error_vs_dt_bin.png", "dt_median",
                      DT_BUCKETS, "Loss by ATIME dt_median bucket", "dt_median [s]")
    plot_worst_10_shots(records, per_shot, args.out_dir / "worst_10_shots.png")
    if args.tb_dir is not None:
        plot_loss_curves(args.tb_dir, args.out_dir / "loss_curves.png")

    print(f"\n[DONE] all outputs in {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
