"""从数据库统计每路 PCPF 的真实 |dI/dt| 分布 (全炮 + 分 phase: ramp_up/flat_top/ramp_down).

用法:
    python -m bc.analyze_didt \
        --shots-file meta/train_shots.txt \
        --out-dir   results/didt_stats

产物 (全炮 / 不分 phase):
    per_channel_quantiles.csv
    didt_histogram.png  didt_cdf.png
    proposed_thresholds.json
    summary.md

产物 (按 phase 分段, phase ∈ {ramp_up, flat_top, ramp_down}):
    per_channel_quantiles_{phase}.csv
    didt_histogram_{phase}.png  didt_cdf_{phase}.png
    proposed_phase_thresholds.json      # 主结果: per-phase per-channel max/P99/P99.9/P99.99
    phase_stats.json                    # 每炮 phase 长度 / Ip_max 等
    phase_demo_ip_vs_time.png           # 3 炮示例: Ip(t) 标出 phase 边界

Phase 检测见 bc.data.phases (IP_PLATEAU_FRAC=0.9, MIN_PLATEAU_S=0.3s).

单位: kA/s.
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

from bc.common.constants import DATASET_ROOT, PCPF_NAMES
from bc.data.phases import (
    PHASE_NAMES,
    detect_phase_slices,
    phase_step_ids,
    phase_ids_per_step,
)


def _read_shots_txt(path: Path) -> list[int]:
    return [int(t) for t in Path(path).read_text().split() if t.strip().isdigit()]


def _collect_shot(
    shot_h5: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, slice]]:
    """Return (|dI/dt|_kAps (T-1, 12), step_mask (T-1, 12), step_phase_id (T-1,),
    phase_slices).

    |dI/dt|[k, c] = |PCPF_c[k+1] - PCPF_c[k]| / max(dt[k+1], 1e-6) 单位 kA/s.
    PCPF 在源 h5 是 Amperes; step_phase_id ∈ {0,1,2,-1} (详见 bc.phases).
    """
    with h5py.File(shot_h5, "r") as f:
        t = np.asarray(f["time"][:], dtype=np.float64)
        T = int(t.shape[0])
        Ip_A = np.asarray(f["PCRL01"][:], dtype=np.float64)  # already on ATIME
        if T < 2:
            return (
                np.empty((0, 12)),
                np.empty((0, 12), dtype=bool),
                np.empty((0,), dtype=np.int8),
                {n: slice(0, 0) for n in PHASE_NAMES},
            )
        actions_A = np.stack(
            [np.asarray(f[n][:], dtype=np.float64) for n in PCPF_NAMES], axis=1
        )
        masks = np.stack(
            [np.asarray(f[f"mask_{n}"][:], dtype=bool) for n in PCPF_NAMES], axis=1
        )

    dt = np.diff(t)
    dt_safe = np.maximum(dt, 1e-6)
    d_action = actions_A[1:] - actions_A[:-1]
    didt_kAps = np.abs(d_action / dt_safe[:, None]) / 1e3  # kA/s, (T-1, 12)
    step_mask = masks[1:] & masks[:-1]

    slices = detect_phase_slices(t, Ip_A)
    pid = phase_ids_per_step(t, Ip_A, valid_len=T)
    step_pid = phase_step_ids(pid)  # (T-1,)
    return didt_kAps, step_mask, step_pid, slices


_QUANTILES = (0.50, 0.90, 0.95, 0.99, 0.995, 0.999, 0.9999, 1.0)


def _quantile_row(name: str, x: np.ndarray) -> dict:
    row = {"PCPF": name, "n_steps": int(x.size)}
    if x.size == 0:
        for q in _QUANTILES:
            row[f"P{q*100:g}"] = float("nan")
        row["mean"] = float("nan")
        return row
    row["mean"] = float(x.mean())
    for q in _QUANTILES:
        row[f"P{q*100:g}"] = float(np.quantile(x, q))
    return row


def _write_stats_for_bucket(
    buckets_per_ch: list[np.ndarray],
    out_dir: Path,
    tag: str,
    title_suffix: str,
) -> dict:
    """Write per-channel quantiles CSV + histogram + CDF; return summary dict."""
    rows = [_quantile_row(n, buckets_per_ch[c]) for c, n in enumerate(PCPF_NAMES)]
    all_vals = (
        np.concatenate([x for x in buckets_per_ch if x.size])
        if any(x.size for x in buckets_per_ch)
        else np.empty(0)
    )
    rows.append(_quantile_row("ALL", all_vals))
    df = pd.DataFrame(rows)
    csv_path = out_dir / (f"per_channel_quantiles{('_' + tag) if tag else ''}.csv")
    df.to_csv(csv_path, index=False)

    # histogram
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=True)
    for c, name in enumerate(PCPF_NAMES):
        ax = axes[c // 4, c % 4]
        x = buckets_per_ch[c]
        if x.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name, fontsize=9)
            continue
        hi = min(30.0, float(np.quantile(x, 0.999)) * 1.05)
        ax.hist(x, bins=np.linspace(0, max(hi, 1.0), 80), log=True)
        p99 = float(np.quantile(x, 0.99))
        p999 = float(np.quantile(x, 0.999))
        mx = float(x.max())
        ax.axvline(p99, color="orange", lw=0.8, ls="--", label=f"P99={p99:.2f}")
        ax.axvline(p999, color="red", lw=0.8, ls="--", label=f"P99.9={p999:.2f}")
        ax.axvline(mx, color="purple", lw=0.8, ls="-", label=f"max={mx:.2f}")
        ax.axvline(4.0, color="k", lw=0.6, ls=":", label="4 kA/s (old)")
        ax.set_title(name, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    for ax in axes[-1, :]:
        ax.set_xlabel("|dI/dt| [kA/s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("count (log)")
    fig.suptitle(f"|dI/dt| histogram{title_suffix}")
    plt.tight_layout()
    fig.savefig(out_dir / (f"didt_histogram{('_' + tag) if tag else ''}.png"), dpi=110)
    plt.close(fig)

    # CDF
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.geomspace(0.01, 50.0, 200)
    for c, name in enumerate(PCPF_NAMES):
        x = buckets_per_ch[c]
        if x.size == 0:
            continue
        cdf = (x[:, None] <= xs[None, :]).mean(axis=0)
        ax.plot(xs, cdf, label=name, lw=0.9, alpha=0.85)
    ax.set_xscale("log")
    ax.axvline(4.0, color="k", ls=":", lw=0.7, label="4 kA/s (old)")
    ax.axhline(0.99, color="orange", ls="--", lw=0.7)
    ax.axhline(0.999, color="red", ls="--", lw=0.7)
    ax.set_xlabel("|dI/dt| [kA/s]")
    ax.set_ylabel("CDF")
    ax.set_title(f"Per-channel |dI/dt| CDF{title_suffix}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(out_dir / (f"didt_cdf{('_' + tag) if tag else ''}.png"), dpi=110)
    plt.close(fig)

    return {
        "csv": str(csv_path),
        "per_channel": {
            row["PCPF"]: {
                "mean": row["mean"],
                "P99": row["P99"],
                "P99.9": row["P99.9"],
                "P99.99": row["P99.99"],
                "max": row["P100"],
            }
            for row in rows
            if row["PCPF"] != "ALL" and not pd.isna(row["mean"])
        },
        "global": {
            "P99": df[df["PCPF"] == "ALL"]["P99"].iat[0],
            "P99.9": df[df["PCPF"] == "ALL"]["P99.9"].iat[0],
            "P99.99": df[df["PCPF"] == "ALL"]["P99.99"].iat[0],
            "max": df[df["PCPF"] == "ALL"]["P100"].iat[0],
        },
    }


def _plot_phase_demo(
    shots_demo: list[int],
    dataset_root: Path,
    out_dir: Path,
) -> None:
    """对 3 炮画 Ip(t) + phase 边界, 直观展示 detection."""
    fig, axes = plt.subplots(len(shots_demo), 1, figsize=(10, 2.4 * len(shots_demo)),
                             sharex=False)
    if len(shots_demo) == 1:
        axes = [axes]
    for ax, shot in zip(axes, shots_demo):
        fp = dataset_root / f"{shot}.h5"
        if not fp.exists():
            ax.set_title(f"shot {shot} missing")
            continue
        with h5py.File(fp, "r") as f:
            t = f["time"][:]
            Ip = f["PCRL01"][:] / 1e3  # kA
        slices = detect_phase_slices(t.astype(np.float64), Ip.astype(np.float64) * 1e3)
        ax.plot(t, Ip, color="k", lw=0.9, label="Ip [kA]")
        colors = {"ramp_up": "C0", "flat_top": "C2", "ramp_down": "C3"}
        for name, s in slices.items():
            if s.stop > s.start:
                ax.axvspan(t[s.start], t[min(s.stop - 1, len(t) - 1)],
                           color=colors[name], alpha=0.15, label=name)
        ax.set_ylabel("Ip [kA]")
        ax.set_title(f"shot {shot}")
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    plt.tight_layout()
    fig.savefig(out_dir / "phase_demo_ip_vs_time.png", dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots-file", type=Path, default=Path("meta/train_shots.txt"))
    ap.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    ap.add_argument("--out-dir", type=Path, default=Path("results/didt_stats"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--progress-every", type=int, default=5000)
    args = ap.parse_args()

    shots = _read_shots_txt(args.shots_file)
    if args.limit is not None:
        shots = shots[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"analyze |dI/dt| (phase-aware) on {len(shots)} shots -> {args.out_dir}")

    # overall + per-phase buckets  (12 channels each)
    overall: list[list[np.ndarray]] = [[] for _ in range(12)]
    per_phase: dict[str, list[list[np.ndarray]]] = {
        name: [[] for _ in range(12)] for name in PHASE_NAMES
    }
    phase_stats = {
        name: {"n_shots_with": 0, "n_steps": 0, "t_total_s": 0.0} for name in PHASE_NAMES
    }
    phase_stats["total_shots_ok"] = 0
    phase_stats["total_shots_err"] = 0

    t0 = time.monotonic()
    for i, shot in enumerate(shots, 1):
        try:
            abs_didt, sm, step_pid, slices = _collect_shot(args.dataset_root / f"{shot}.h5")
        except Exception as e:
            phase_stats["total_shots_err"] += 1
            if phase_stats["total_shots_err"] <= 5:
                print(f"  err shot {shot}: {type(e).__name__}: {e}")
            continue
        phase_stats["total_shots_ok"] += 1
        if abs_didt.size == 0:
            continue

        # overall
        for c in range(12):
            vals = abs_didt[:, c][sm[:, c]]
            if vals.size:
                overall[c].append(vals.astype(np.float32))

        # per phase
        for pid, name in enumerate(PHASE_NAMES):
            m_phase_step = step_pid == pid
            if not m_phase_step.any():
                continue
            phase_stats[name]["n_shots_with"] += 1
            for c in range(12):
                m = m_phase_step & sm[:, c]
                if m.any():
                    per_phase[name][c].append(abs_didt[:, c][m].astype(np.float32))
                    phase_stats[name]["n_steps"] += int(m.sum())
        # time in each phase (from slice widths)
        with h5py.File(args.dataset_root / f"{shot}.h5", "r") as f:
            t_arr = f["time"][:]
        for name, s in slices.items():
            if s.stop > s.start:
                phase_stats[name]["t_total_s"] += float(t_arr[s.stop - 1] - t_arr[s.start])

        if i % args.progress_every == 0:
            elapsed = time.monotonic() - t0
            eta = elapsed / i * (len(shots) - i)
            print(
                f"  {i}/{len(shots)}  ok={phase_stats['total_shots_ok']} "
                f"err={phase_stats['total_shots_err']}  "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s"
            )
    print(
        f"\nDone: ok={phase_stats['total_shots_ok']} "
        f"err={phase_stats['total_shots_err']}  "
        f"elapsed={time.monotonic()-t0:.0f}s"
    )

    # concat per bucket
    overall_concat = [
        np.concatenate(overall[c]) if overall[c] else np.empty(0, dtype=np.float32)
        for c in range(12)
    ]
    per_phase_concat = {
        name: [
            np.concatenate(per_phase[name][c]) if per_phase[name][c] else np.empty(0, dtype=np.float32)
            for c in range(12)
        ]
        for name in PHASE_NAMES
    }

    # ----- overall (no phase) -----
    overall_summary = _write_stats_for_bucket(
        overall_concat, args.out_dir, tag="", title_suffix=" (all steps)"
    )
    # keep legacy proposed_thresholds.json
    (args.out_dir / "proposed_thresholds.json").write_text(
        json.dumps(
            {
                "source": str(args.shots_file),
                "n_shots_ok": phase_stats["total_shots_ok"],
                "n_shots_err": phase_stats["total_shots_err"],
                "unit": "kA/s",
                "note": ("数据驱动阈值: <1% 违规率用 P99; <0.1% 用 P99.9; "
                         "几乎零违规用 P99.99 或 max"),
                "per_channel": overall_summary["per_channel"],
                "global": overall_summary["global"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    # ----- per-phase -----
    phase_summary = {}
    for name in PHASE_NAMES:
        tag = name
        phase_summary[name] = _write_stats_for_bucket(
            per_phase_concat[name], args.out_dir, tag=tag,
            title_suffix=f" ({name})",
        )
        phase_summary[name]["n_steps"] = phase_stats[name]["n_steps"]
        phase_summary[name]["n_shots_with"] = phase_stats[name]["n_shots_with"]
        phase_summary[name]["t_total_s"] = round(phase_stats[name]["t_total_s"], 2)

    # ----- proposed_phase_thresholds.json (primary output) -----
    proposed = {
        "source": str(args.shots_file),
        "n_shots_ok": phase_stats["total_shots_ok"],
        "n_shots_err": phase_stats["total_shots_err"],
        "unit": "kA/s",
        "phase_detection": {
            "ip_plateau_frac": 0.90,
            "min_plateau_s": 0.30,
            "ip_min_A": 100_000.0,
        },
        "phases": {
            name: {
                "n_shots_with": phase_summary[name]["n_shots_with"],
                "n_steps": phase_summary[name]["n_steps"],
                "t_total_s": phase_summary[name]["t_total_s"],
                "per_channel_max": {
                    n: phase_summary[name]["per_channel"][n]["max"]
                    for n in PCPF_NAMES
                    if n in phase_summary[name]["per_channel"]
                },
                "per_channel_P99_99": {
                    n: phase_summary[name]["per_channel"][n]["P99.99"]
                    for n in PCPF_NAMES
                    if n in phase_summary[name]["per_channel"]
                },
                "per_channel_P99_9": {
                    n: phase_summary[name]["per_channel"][n]["P99.9"]
                    for n in PCPF_NAMES
                    if n in phase_summary[name]["per_channel"]
                },
                "per_channel_P99": {
                    n: phase_summary[name]["per_channel"][n]["P99"]
                    for n in PCPF_NAMES
                    if n in phase_summary[name]["per_channel"]
                },
                "global": phase_summary[name]["global"],
            }
            for name in PHASE_NAMES
        },
    }
    (args.out_dir / "proposed_phase_thresholds.json").write_text(
        json.dumps(proposed, indent=2, ensure_ascii=False)
    )
    print(f"\nwrote {args.out_dir/'proposed_phase_thresholds.json'}")

    # phase stats
    (args.out_dir / "phase_stats.json").write_text(
        json.dumps(phase_stats, indent=2, ensure_ascii=False)
    )

    # phase detection demo
    try:
        demo_shots = [int(s) for s in shots[:1]] + [100082, 109100, 121041]
        demo_shots = [s for s in demo_shots if (args.dataset_root / f"{s}.h5").exists()]
        demo_shots = list(dict.fromkeys(demo_shots))[:3]
        _plot_phase_demo(demo_shots, args.dataset_root, args.out_dir)
        print(f"wrote {args.out_dir/'phase_demo_ip_vs_time.png'}")
    except Exception as e:
        print(f"phase demo plot failed: {e}")

    # summary.md
    lines = [
        "# |dI/dt| phase-aware thresholds",
        "",
        f"- source: `{args.shots_file}`",
        f"- shots ok / err: {phase_stats['total_shots_ok']} / {phase_stats['total_shots_err']}",
        f"- unit: kA/s",
        "",
        "## Phase breakdown",
        "",
        f"| phase | n_shots_with | n_steps | t_total [s] |",
        "|---|---:|---:|---:|",
    ]
    for name in PHASE_NAMES:
        lines.append(
            f"| {name} | {phase_stats[name]['n_shots_with']} | {phase_stats[name]['n_steps']:,} "
            f"| {phase_stats[name]['t_total_s']:,.1f} |"
        )
    lines += [
        "",
        "## per-channel `max |dI/dt|` by phase (kA/s)",
        "",
        f"| PCPF | ramp_up | flat_top | ramp_down |",
        "|---|---:|---:|---:|",
    ]
    for name in PCPF_NAMES:
        ru = proposed["phases"]["ramp_up"]["per_channel_max"].get(name, float("nan"))
        ft = proposed["phases"]["flat_top"]["per_channel_max"].get(name, float("nan"))
        rd = proposed["phases"]["ramp_down"]["per_channel_max"].get(name, float("nan"))
        lines.append(f"| {name} | {ru:.2f} | {ft:.2f} | {rd:.2f} |")
    lines += [
        "",
        "(也可用 P99.99 或 P99.9 作更稳健的 phase 阈值; 详见 proposed_phase_thresholds.json)",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(lines))
    print(f"wrote {args.out_dir/'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
