#!/usr/bin/env python3
"""Flat-top phase: Prediction only, single subplot.

Shows:
  - EFIT (R8,Z8)            dashed blue
  - FreeGSNKE (R8,Z8)       solid red (Prediction)

White background, legend on the right-middle with large font.

Usage::

    python scripts/plot_flattop_pred_only.py --shot 134925
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.environ.setdefault("NUMBA_CACHE_DIR", str(_REPO / ".numba_cache"))
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

from bc_notime.data.phases import PHASE_NAMES
from bc_notime.gs_forward.precursor_export import load_single_slice_npz

# ── colour / style constants ──────────────────────────────────────────
COL_EFIT = "#1E88E5"
COL_EFIT_EDGE = "#1565C0"
COL_PRED = "#E53935"
COL_PRED_EDGE = "#B71C1C"
BG_COLOR = "white"

LW_CURVE = 3.5
DOT_SIZE = 40
DOT_EDGE_LW = 1.0

LEGEND_FONT_SIZE = 22

PUB_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 14,
    "font.weight": "bold",
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.labelsize": 14,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.labelcolor": "black",
    "ytick.labelcolor": "black",
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "savefig.dpi": 300,
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
}


# ── helpers ────────────────────────────────────────────────────────────

def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _frame_info_from_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    picks: dict[str, dict[str, Any]] = {}
    for ph in PHASE_NAMES:
        entry = manifest.get(ph)
        picks[ph] = {"k": entry["k"], "time_s": entry["time_s"]} if entry else None
    return picks


# ── forward solve for one slice ────────────────────────────────────────

def _forward_solve_slice(
    slice_npz: Path,
    pf_xml: str | Path,
    wall_xml: str | Path,
    nx: int = 129,
    ny: int = 129,
    rtol: float = 1e-3,
    maxit: int = 80,
):
    from bc_notime.gs_forward.east_pf_mapping import pcpf12_to_pf14_amps
    from bc_notime.gs_forward.freegsnke_east_machine import (
        build_east_tokamak_freegsnke,
        currents_dict_to_vec,
        wall_outline_rz_from_xml,
    )
    from freegsnke.equilibrium_update import Equilibrium
    from freegsnke.jtor_update import ConstrainBetapIp, GeneralPprimeFFprime
    from freegsnke.GSstaticsolver import NKGSsolver

    sl = load_single_slice_npz(slice_npz)
    snap = sl["snapshot"]
    pcpf12 = np.asarray(sl["pcpf12"], dtype=np.float64).ravel()
    ip_sel = float(sl["Ip_A"])
    pf_amps = pcpf12_to_pf14_amps(pcpf12)

    lim_r, lim_z = wall_outline_rz_from_xml(wall_xml)
    Rmin, Rmax = float(np.nanmin(lim_r)), float(np.nanmax(lim_r))
    Zmin, Zmax = float(np.nanmin(lim_z)), float(np.nanmax(lim_z))

    tok = build_east_tokamak_freegsnke(pf_xml, wall_xml, quiet=True)
    vec = currents_dict_to_vec(tok, pf_amps)
    tok.set_all_coil_currents(vec)

    psi_n = np.linspace(0.0, 1.0, len(snap["pprime"]), dtype=np.float64)
    bcentr = float(snap.get("bcentr", 0))
    rmaxis = float(snap.get("rmaxis", Rmin + 0.5 * (Rmax - Rmin)))
    fvac = bcentr * rmaxis if bcentr > 0 else 1.0
    Ip = ip_sel

    def _make():
        e = Equilibrium(tok, Rmin, Rmax, Zmin, Zmax, nx=nx, ny=ny, order=4)
        tok.set_all_coil_currents(vec)
        try:
            e.adjust_psi_plasma()
        except Exception:
            pass
        return e

    def _solve(eq_obj, prof):
        nk = NKGSsolver(eq_obj, seed=42)
        nk.forward_solve(eq_obj, prof, target_relative_tolerance=rtol,
                         max_solving_iterations=maxit, suppress=True, verbose=False)
        return float(nk.relative_change)

    eq = _make()
    try:
        profiles = GeneralPprimeFFprime(
            eq, Ip, fvac, psi_n,
            pprime_data=np.asarray(snap["pprime"], dtype=np.float64),
            ffprime_data=np.asarray(snap["ffprim"], dtype=np.float64),
            p_data=None, f_data=None, Raxis=rmaxis, Ip_logic=True,
        )
        _solve(eq, profiles)
        return eq, snap
    except Exception:
        pass

    eq = _make()
    try:
        profiles = ConstrainBetapIp(
            eq, float(snap.get("betap", 0.5)), Ip, fvac, Raxis=rmaxis,
        )
        _solve(eq, profiles)
        return eq, snap
    except Exception:
        return None


# ── extract FreeGSNKE R8/Z8 from equilibrium ──────────────────────────

def _freegsnke_r8z8(eq):
    """Return (r8, z8) arrays from equilibrium separatrix, or None."""
    from scan_data.compat import shape_params
    try:
        sep = np.asarray(eq.separatrix(ntheta=240))
        if sep.size > 8:
            valid = ~(np.isnan(sep[:, 0]) | np.isnan(sep[:, 1]))
            sep_clean = sep[valid]
            if sep_clean.shape[0] >= 8:
                sp = shape_params(sep_clean[:, 0], sep_clean[:, 1])
                r8 = np.asarray(sp.R8, dtype=np.float64).ravel()
                z8 = np.asarray(sp.Z8, dtype=np.float64).ravel()
                if np.all(np.isfinite(r8)) and np.all(np.isfinite(z8)):
                    return r8, z8
    except Exception:
        pass
    return None


# ── render single subplot ──────────────────────────────────────────────

def _render_subplot(ax, eq, snap, *, color, edge_color, legend_label):
    """Draw EFIT R8/Z8 (dashed) + one FreeGSNKE R8/Z8 (solid) on *ax*."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # EFIT R8/Z8 (dashed blue)
    r8a = np.asarray(snap["r8"], dtype=np.float64).ravel()
    z8a = np.asarray(snap["z8"], dtype=np.float64).ravel()
    ax.plot(np.r_[r8a, r8a[0]], np.r_[z8a, z8a[0]],
            color=COL_EFIT, ls="--", lw=LW_CURVE, dashes=(6, 4), zorder=6)
    ax.scatter(r8a, z8a, s=DOT_SIZE, facecolors=COL_EFIT,
               edgecolors=COL_EFIT_EDGE, linewidths=DOT_EDGE_LW, zorder=7)

    # FreeGSNKE R8/Z8 (solid)
    r8_fwd = z8_fwd = None
    res = _freegsnke_r8z8(eq)
    if res is not None:
        r8_fwd, z8_fwd = res
        ax.plot(np.r_[r8_fwd, r8_fwd[0]], np.r_[z8_fwd, z8_fwd[0]],
                color=color, lw=LW_CURVE, zorder=5)
        ax.scatter(r8_fwd, z8_fwd, s=DOT_SIZE, facecolors=color,
                   edgecolors=edge_color, linewidths=DOT_EDGE_LW, zorder=7)

    # Axis range
    rs = np.concatenate([r8a, r8a])
    zs = np.concatenate([z8a, z8a])
    if r8_fwd is not None:
        rs = np.concatenate([rs, r8_fwd])
        zs = np.concatenate([zs, z8_fwd])
    mr = max(0.06, 0.03 * np.ptp(rs))
    mz = max(0.06, 0.03 * np.ptp(zs))
    ax.set_xlim(rs.min() - mr, rs.max() + mr)
    ax.set_ylim(zs.min() - mz, zs.max() + mz)

    # Cap lines at top of left spine and right of bottom spine
    from matplotlib.lines import Line2D
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    cap = 0.012 * (yt - yb)
    cap_x = 0.012 * (xr - xl)
    ax.add_line(Line2D([xl, xl], [yt - cap, yt], color="black", lw=1.5,
                        transform=ax.transData, clip_on=False))
    ax.add_line(Line2D([xr - cap_x, xr], [yb, yb], color="black", lw=1.5,
                        transform=ax.transData, clip_on=False))

    ax.set_xlabel(r"$\mathbf{R}$ (m)", fontsize=22, fontweight="bold")
    ax.set_ylabel(r"$\mathbf{Z}$ (m)", fontsize=22, fontweight="bold")
    ax.set_aspect("equal")

    # Legend on top of the plot, horizontal layout, large font, white background
    handles = [
        Line2D([], [], color=COL_EFIT, ls="--", lw=LW_CURVE, dashes=(6, 4),
               marker="o", markersize=8, markerfacecolor=COL_EFIT,
               markeredgecolor=COL_EFIT_EDGE, markeredgewidth=DOT_EDGE_LW,
               label=r"$\mathbf{EFIT}$ $\mathbf{(R_8,Z_8)}$"),
        Line2D([], [], color=color, lw=LW_CURVE,
               marker="o", markersize=8, markerfacecolor=color,
               markeredgecolor=edge_color, markeredgewidth=DOT_EDGE_LW,
               label=legend_label),
    ]
    leg = ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        fontsize=LEGEND_FONT_SIZE,
        facecolor="white",
        title=None,
        borderaxespad=0.5,
        prop={"weight": "bold", "size": LEGEND_FONT_SIZE},
    )


# ── main ───────────────────────────────────────────────────────────────

def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from bc_notime.gs_forward.freegsnke_east_machine import PF_XML_DEFAULT, WALL_XML_DEFAULT

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", type=int, required=True)
    ap.add_argument("--whole-shot-root", type=str,
                    default=str(_REPO / "results" / "freegsnke_whole_shot"))
    ap.add_argument("--out-dir", type=str,
                    default=str(_REPO / "results" / "bc_v1_notime" / "run1" / "paper_figures"))
    ap.add_argument("--pf-xml", type=str, default=str(PF_XML_DEFAULT))
    ap.add_argument("--wall-xml", type=str, default=str(WALL_XML_DEFAULT))
    args = ap.parse_args()

    shot = args.shot
    whole_root = Path(args.whole_shot_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    manifest_path = whole_root / "_montage" / f"montage_pcs_vs_pred_shot{shot}.json"
    manifest = _load_manifest(manifest_path)
    if manifest is None:
        print(f"[error] no manifest: {manifest_path}", file=sys.stderr)
        return 2
    picks = _frame_info_from_manifest(manifest)
    pick = picks.get("flat_top")
    if pick is None:
        print("[error] no flat_top entry in manifest", file=sys.stderr)
        return 2

    k, t_s = pick["k"], pick["time_s"]
    print(f"[info] flat-top: k={k}, t={t_s:.3f} s", flush=True)

    # Forward-solve prediction branch only
    branch = "bc_pred"
    file_tag = "bc"
    slice_npz = (whole_root / str(shot) / branch / "_slices"
                 / f"slice_whole_{file_tag}_k{k:05d}.npz")
    if not slice_npz.is_file():
        print(f"[error] missing {slice_npz}", file=sys.stderr)
        return 2
    print(f"[forward] {branch} k={k} t={t_s:.3f}s ...", flush=True)
    result = _forward_solve_slice(slice_npz, args.pf_xml, args.wall_xml)
    if result is None:
        print("[error] solve failed for bc_pred", file=sys.stderr)
        return 2

    eq, snap = result

    # ── single plot with room for legend on the right ──────────────────
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    legend_label = r"$\mathbf{FreeGSNKE}$ $\mathbf{(R_8,Z_8)}$"

    with plt.rc_context(PUB_RC):
        _render_subplot(ax, eq, snap,
                        color=COL_PRED, edge_color=COL_PRED_EDGE,
                        legend_label=legend_label)

    fig.subplots_adjust(top=0.85)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"flattop_pred_only_shot{shot}.png"
    pdf_path = out_dir / f"flattop_pred_only_shot{shot}.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white", edgecolor="none")
    fig.savefig(pdf_path, facecolor="white", edgecolor="none")
    plt.close(fig)

    print(f"[saved] {png_path}")
    print(f"[saved] {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
