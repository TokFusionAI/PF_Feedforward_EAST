#!/usr/bin/env python3
"""Prediction-only 1×3 montage + Ip(t) strip, with first-wall gap annotation.

Based on plot_montage_paper.py — only the Prediction row is kept.
Each subplot adds a red circle at the inner (leftmost) FreeGSNKE R8/Z8 point
and annotates its horizontal distance to the first wall (limiter).

Usage::

    python scripts_notime/plot_pred_only_wall_dist.py --shot 134925
"""

from __future__ import annotations

import argparse
import json
import math
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
from bc_notime.gs_forward.precursor_export import load_precursor_npz, load_single_slice_npz

_OVERLAY_COIL_NAMES = (
    "PF1", "PF2", "PF3", "PF4", "PF5", "PF6", "PF7", "PF8",
    "PF9", "PF10", "PF11", "PF12", "PF13", "PF14", "IC1", "IC2",
)

COL_PF_IC = "#40C4C4"
COL_VESSEL = "#1A630E"
COL_LIMITER = "magenta"
COL_TARGET = "#90E0EF"
COL_TARGET_EDGE = "#5BC0DE"
COL_PREDICTION = "#F08080"

PUB_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 11,
    "font.weight": "normal",
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "axes.linewidth": 1.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.labelcolor": "black",
    "ytick.labelcolor": "black",
    "legend.frameon": True,
    "legend.edgecolor": "0.85",
    "legend.facecolor": "1.0",
    "legend.framealpha": 1.0,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


# ── helpers ─────────────────────────────────────────────────────────────

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


def _interp_ip(atime, ip, t):
    m = np.isfinite(atime) & np.isfinite(ip)
    if not np.any(m):
        return float("nan")
    a = np.asarray(atime[m], dtype=np.float64)
    y = np.asarray(ip[m], dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    a, y = a[order], y[order]
    t = float(t)
    if t <= a[0]:
        return float(y[0])
    if t >= a[-1]:
        return float(y[-1])
    return float(np.interp(t, a, y))


def _read_shots_file(path: Path) -> list[int]:
    out: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip()
        if s.isdigit():
            out.append(int(s))
    return out


# ── forward solve ───────────────────────────────────────────────────────

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


# ── subplot renderer (from original, with wall-gap annotation added) ────

def _render_subplot(
    ax,
    eq,
    snap: dict[str, Any],
    *,
    pf_xml: str | Path,
    wall_xml: str | Path,
):
    import matplotlib.patches
    import matplotlib.patheffects as mpe
    from scan_data.compat import shape_params
    from bc_notime.gs_forward.freegsnke_east_machine import (
        active_coils_dict_from_pf_xml,
        pf_coils_rzwh_for_overlay,
        vessel_annular_rz_for_overlay,
    )

    tok = eq.tokamak
    lim = tok.limiter

    # limiter (first wall)
    lr = lz = None
    if lim is not None and len(getattr(lim, "R", ())):
        lr = np.asarray(lim.R, dtype=np.float64).ravel()
        lz = np.asarray(lim.Z, dtype=np.float64).ravel()
        ax.plot(np.r_[lr, lr[0]], np.r_[lz, lz[0]], color=COL_LIMITER, lw=1.0, zorder=2)

    # wall
    w = getattr(tok, "wall", None)
    if w is not None and len(getattr(w, "R", ())) and w is not lim:
        wr = np.asarray(w.R, dtype=np.float64).ravel()
        wz = np.asarray(w.Z, dtype=np.float64).ravel()
        ax.plot(wr, wz, color="0.5", ls=(0, (3, 2)), lw=0.8, zorder=1)

    # vessel
    inner_v, outer_v, _ = vessel_annular_rz_for_overlay(wall_xml)
    if inner_v is not None:
        ri, zi = inner_v
        ax.plot(np.r_[ri, ri[0]], np.r_[zi, zi[0]], color=COL_VESSEL, lw=1.5, zorder=2.4)
    if outer_v is not None:
        ro, zo = outer_v
        ax.plot(np.r_[ro, ro[0]], np.r_[zo, zo[0]], color=COL_VESSEL, lw=1.5, zorder=2.4)

    # PF coils
    active_rects, _ = pf_coils_rzwh_for_overlay(pf_xml)
    active_fb = {}
    if not active_rects:
        active_fb = active_coils_dict_from_pf_xml(pf_xml)
        for nm, d in active_fb.items():
            cx, cy = float(d["R"][0]), float(d["Z"][0])
            w0, h0 = float(d["dR"]), float(d["dZ"])
            wv, hv = w0 * 1.08, h0 * 1.08
            rect = matplotlib.patches.Rectangle(
                (cx - wv / 2, cy - hv / 2), wv, hv,
                fill=False, lw=0.8, edgecolor=COL_PF_IC, zorder=3)
            ax.add_patch(rect)
            ax.text(cx, cy, nm, ha="center", va="center", fontsize=7, color="black", zorder=22,
                    path_effects=[mpe.withStroke(linewidth=1.8, foreground="white")])
    else:
        for node in _OVERLAY_COIL_NAMES:
            t = active_rects.get(node)
            if t is None:
                continue
            cx, cy, w0, h0 = t
            nbr = [cx - .5*w0, cx + .5*w0, cx + .5*w0, cx - .5*w0, cx - .5*w0]
            nbz = [cy - .5*h0, cy - .5*h0, cy + .5*h0, cy + .5*h0, cy - .5*h0]
            ax.plot(nbr, nbz, color=COL_PF_IC, lw=0.7, zorder=3)
            ax.plot([nbr[0], nbr[2]], [nbz[0], nbz[2]], color=COL_PF_IC, lw=0.35, zorder=3)
            ax.plot([nbr[1], nbr[3]], [nbz[1], nbz[3]], color=COL_PF_IC, lw=0.35, zorder=3)
            ax.text(nbr[1], .5*(nbz[0]+nbz[3]), node, ha="left", va="center",
                    fontsize=6.5, color="black", zorder=22,
                    path_effects=[mpe.withStroke(linewidth=1.4, foreground="white")])

    # EFIT R8/Z8 (dashed light blue)
    r8a = np.asarray(snap["r8"], dtype=np.float64).ravel()
    z8a = np.asarray(snap["z8"], dtype=np.float64).ravel()
    ax.plot(np.r_[r8a, r8a[0]], np.r_[z8a, z8a[0]],
            color=COL_TARGET, ls="--", lw=1.5, dashes=(6, 4),
            label=r"EFIT $(R_i,Z_i)$", zorder=6)
    ax.scatter(r8a, z8a, s=15, facecolors=COL_TARGET, edgecolors=COL_TARGET_EDGE,
               linewidths=0.4, zorder=7)

    # FreeGSNKE LCFS → R8/Z8 via shape_params (solid coral)
    r8_fwd = z8_fwd = None
    try:
        sep = np.asarray(eq.separatrix(ntheta=240))
        if sep.size > 8:
            valid = ~(np.isnan(sep[:, 0]) | np.isnan(sep[:, 1]))
            sep_clean = sep[valid]
            if sep_clean.shape[0] >= 8:
                sp = shape_params(sep_clean[:, 0], sep_clean[:, 1])
                r8_fwd = np.asarray(sp.R8, dtype=np.float64).ravel()
                z8_fwd = np.asarray(sp.Z8, dtype=np.float64).ravel()
                if np.all(np.isfinite(r8_fwd)) and np.all(np.isfinite(z8_fwd)):
                    ax.plot(np.r_[r8_fwd, r8_fwd[0]], np.r_[z8_fwd, z8_fwd[0]],
                            color=COL_PREDICTION, lw=2.0,
                            label=r"FreeGSNKE $(R_i,Z_i)$", zorder=5)
                    ax.scatter(r8_fwd, z8_fwd, s=15, facecolors=COL_PREDICTION,
                               edgecolors="darkred", linewidths=0.4, zorder=7)
    except Exception:
        pass

    # magnetic axis
    if getattr(eq, "opt", None) is not None and len(eq.opt):
        ax.plot(float(eq.opt[0, 0]), float(eq.opt[0, 1]), "+",
                color="#CC0000", ms=7, mew=1.0, zorder=7)

    # ── wall-gap annotation ─────────────────────────────────────────
    if r8_fwd is not None and lr is not None and lr.size > 0:
        r_inner = float(r8_fwd[4])               # leftmost prediction point
        z_inner = float(z8_fwd[4])
        r_wall = float(np.nanmin(lr))             # inboard first wall R
        dist_cm = (r_inner - r_wall) * 100.0

        # DEBUG: print actual values for verification
        print(f"  [DEBUG] R8_fwd[4]={r_inner:.4f} m, Z8_fwd[4]={z_inner:.4f} m, "
              f"min(limiter_R)={r_wall:.4f} m, dist={dist_cm:.2f} cm",
              flush=True)

        # red circle on the inner prediction point
        ax.scatter([r_inner], [z_inner], s=160, facecolors="none",
                   edgecolors="#CC0000", linewidths=2.2, zorder=10)
        # arrow to blank area, black text, no border
        ax.annotate(
            f"{dist_cm:.1f} cm",
            xy=(r_inner, z_inner),
            xytext=(-35, -30), textcoords="offset points",
            fontsize=10, fontweight="bold", color="black",
            ha="center", va="top",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.0,
                            connectionstyle="arc3,rad=0.2"),
            zorder=11,
        )

    # axis range (same as original)
    rs, zs = [], []
    if lr is not None and lr.size:
        rs.extend(lr.tolist()); zs.extend(lz.tolist())
    if active_rects:
        for node in _OVERLAY_COIL_NAMES:
            t = active_rects.get(node)
            if t is None:
                continue
            cx, cy, w0, h0 = t
            rs.extend([cx - .5*w0, cx + .5*w0])
            zs.extend([cy - .5*h0, cy + .5*h0])
    else:
        for d in active_fb.values():
            cx, cy = float(d["R"][0]), float(d["Z"][0])
            w0, h0 = float(d["dR"]), float(d["dZ"])
            rs.extend([cx - w0*.54, cx + w0*.54])
            zs.extend([cy - h0*.54, cy + h0*.54])
    if inner_v is not None:
        rs.extend(inner_v[0].tolist()); zs.extend(inner_v[1].tolist())
    if outer_v is not None:
        rs.extend(outer_v[0].tolist()); zs.extend(outer_v[1].tolist())
    rs.extend(r8a.tolist()); zs.extend(z8a.tolist())
    if r8_fwd is not None:
        rs.extend(r8_fwd.tolist()); zs.extend(z8_fwd.tolist())
    if rs and zs:
        span_r = max(rs) - min(rs)
        span_z = max(zs) - min(zs)
        mr = max(0.06, 0.03 * span_r)
        mz = max(0.06, 0.03 * span_z)
        ax.set_xlim(min(rs) - mr, max(rs) + mr)
        ax.set_ylim(min(zs) - mz, max(zs) + mz)

    ax.set_xlabel("$R$ (m)", fontsize=11)
    ax.set_ylabel("$Z$ (m)", fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, color="0.88", lw=0.4, alpha=1.0)

    h, lab = ax.get_legend_handles_labels()
    uniq: dict[str, object] = {}
    for hi, li in zip(h, lab):
        if li and not str(li).startswith("_") and li not in uniq:
            uniq[li] = hi
    if uniq:
        ax.legend(uniq.values(), uniq.keys(), loc="upper right",
                  frameon=True, fancybox=False, edgecolor="0.8", fontsize=8)


# ── Ip(t) strip (same as original) ─────────────────────────────────────

def _render_ip_strip(ax, atime, ip, picks, shot):
    import matplotlib.patheffects as pe

    ip_ka = np.asarray(ip, dtype=np.float64) / 1000.0
    at_a = np.asarray(atime, dtype=np.float64)
    m = np.isfinite(at_a) & np.isfinite(ip_ka)
    ax.plot(at_a[m], ip_ka[m], color="black", lw=1.5, alpha=0.92, zorder=1)

    for ph in PHASE_NAMES:
        pick = picks.get(ph)
        if pick is None:
            continue
        ts = float(pick["time_s"])
        if not math.isfinite(ts):
            continue
        iy = _interp_ip(at_a, ip_ka, ts)
        if math.isfinite(iy):
            ax.scatter([ts], [iy], s=80, facecolors="none", edgecolors="red",
                       linewidths=1.8, zorder=5)
            dy = -8 if ph != "ramp_down" else 8
            va = "top" if ph != "ramp_down" else "bottom"
            ax.annotate(f"{ts:.2f} s", xy=(ts, iy), xytext=(6, dy),
                        textcoords="offset points", ha="left", va=va,
                        fontsize=9, color="black", fontweight="medium",
                        clip_on=False, zorder=6,
                        path_effects=[pe.Stroke(linewidth=2.8, foreground="white"),
                                      pe.Normal()])

    ax.set_xlabel(r"$t$ (s)", fontsize=11)
    ax.set_ylabel(r"$I_{\mathrm{p}}$ (kA)", fontsize=11)
    ax.grid(True, ls=":", alpha=0.35, color="0.70")
    ax.tick_params(labelsize=10)
    ax.margins(x=0.03, y=0.18)


# ── main ───────────────────────────────────────────────────────────────

def _process_shot(*, shot, precursor_npz, whole_shot_root, out_dir, pf_xml, wall_xml):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not precursor_npz.is_file():
        print(f"[skip] shot={shot} no precursor {precursor_npz}",
              file=sys.stderr, flush=True)
        return 2
    bundle = load_precursor_npz(precursor_npz)
    atime = np.asarray(bundle["ATIME"], dtype=np.float64).ravel()
    ipr = np.asarray(bundle["PCRL01"], dtype=np.float64).ravel()

    manifest_path = whole_shot_root / "_montage" / f"montage_pcs_vs_pred_shot{shot}.json"
    manifest = _load_manifest(manifest_path)
    if manifest is None:
        print(f"[skip] shot={shot} no manifest {manifest_path}",
              file=sys.stderr, flush=True)
        return 2
    picks = _frame_info_from_manifest(manifest)

    # ── layout: 1 row (Prediction × 3 phases) + Ip strip ──
    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(14.5, 7.0))
        gs = fig.add_gridspec(
            2, 3,
            height_ratios=[1, 0.35],
            hspace=0.22, wspace=0.0,
            left=0.05, right=0.98, top=0.90, bottom=0.10,
        )

    axes_row = [fig.add_subplot(gs[0, c]) for c in range(3)]
    ax_ip = fig.add_subplot(gs[1, :])

    branch = "bc_pred"
    file_tag = "bc"

    for col_idx, phase in enumerate(PHASE_NAMES):
        ax = axes_row[col_idx]
        pick = picks.get(phase)
        if pick is None:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="0.5")
            ax.set_aspect("equal")
            continue

        k = pick["k"]
        t_s = pick["time_s"]
        slice_npz = (whole_shot_root / str(shot) / branch / "_slices"
                     / f"slice_whole_{file_tag}_k{k:05d}.npz")
        if not slice_npz.is_file():
            ax.text(0.5, 0.5, f"missing\nk={k}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="0.5")
            ax.set_aspect("equal")
            continue

        print(f"[forward] Prediction {phase} k={k} t={t_s:.3f}s ...", flush=True)
        solved = _forward_solve_slice(slice_npz, pf_xml, wall_xml)
        if solved is None:
            ax.text(0.5, 0.5, "solve\nfailed", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="red")
            ax.set_aspect("equal")
            continue

        eq, snap = solved
        with plt.rc_context(PUB_RC):
            _render_subplot(ax, eq, snap, pf_xml=pf_xml, wall_xml=wall_xml)

    with plt.rc_context(PUB_RC):
        _render_ip_strip(ax_ip, atime, ipr, picks, shot)

    # pack top-row subplots: reduce gaps from set_aspect("equal") but keep readable spacing
    fig.canvas.draw()
    poss = [ax.get_position() for ax in axes_row]
    widths = [p.width for p in poss]
    gap = 0.018  # gap between subplots (figure coords)
    total_w = sum(widths) + gap * 2
    x_center = 0.5 * (poss[0].x0 + poss[-1].x1)
    x_start = x_center - total_w / 2
    for i, ax in enumerate(axes_row):
        p = poss[i]
        new_x = x_start + sum(widths[:i]) + gap * i
        ax.set_position([new_x, p.y0, p.width, p.height])

    # only first subplot shows y-axis label & tick labels
    for i, ax in enumerate(axes_row):
        if i > 0:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    # align Ip strip x-range to packed top-row subplots
    try:
        x_lo = axes_row[0].get_position().x0
        x_hi = axes_row[-1].get_position().x1
        ip_pos = ax_ip.get_position()
        ax_ip.set_position([x_lo, ip_pos.y0, x_hi - x_lo, ip_pos.height])
    except Exception:
        pass

    # column headers
    for col_idx, phase in enumerate(PHASE_NAMES):
        pos = axes_row[col_idx].get_position()
        pick = picks.get(phase)
        t_str = (f" @$\\mathbf{{t}}$={pick['time_s']:.2f} s") if pick else ""
        phase_label = {"ramp_up": "Ramp-up", "flat_top": "Flat-top", "ramp_down": "Ramp-down"}
        fig.text(0.5 * (pos.x0 + pos.x1), pos.y1 + 0.012,
                 phase_label[phase] + t_str,
                 ha="center", va="bottom", fontsize=11, fontweight="bold", color="black")

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"pred_wall_dist_shot{shot}.png"
    pdf_path = out_dir / f"pred_wall_dist_shot{shot}.pdf"

    fig.savefig(png_path, dpi=300, facecolor="white", edgecolor="none",
                bbox_inches="tight", pad_inches=0.15)
    fig.savefig(pdf_path, facecolor="white", edgecolor="none",
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    print(f"[saved] {png_path}", flush=True)
    print(f"[saved] {pdf_path}", flush=True)
    return 0


def main() -> int:
    from bc_notime.gs_forward.freegsnke_east_machine import PF_XML_DEFAULT, WALL_XML_DEFAULT

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", type=int, default=None)
    ap.add_argument("--shots-file", type=str, default=None)
    ap.add_argument("--precursor-root", type=str,
                    default=str(_REPO / "results" / "freegsnke_precursors"))
    ap.add_argument("--whole-shot-root", type=str,
                    default=str(_REPO / "results" / "freegsnke_whole_shot_notime"))
    ap.add_argument("--out-dir", type=str,
                    default=str(_REPO / "results" / "bc_v1_notime" / "run1" / "paper_figures"))
    ap.add_argument("--pf-xml", type=str, default=str(PF_XML_DEFAULT))
    ap.add_argument("--wall-xml", type=str, default=str(WALL_XML_DEFAULT))
    args = ap.parse_args()

    shots: list[int] = []
    if args.shots_file:
        p = Path(args.shots_file).resolve()
        if not p.is_file():
            raise SystemExit(f"--shots-file not found: {p}")
        shots.extend(_read_shots_file(p))
    if args.shot is not None:
        shots.append(int(args.shot))
    if not shots:
        raise SystemExit("specify --shot or --shots-file")

    prec_root = Path(args.precursor_root).resolve()
    whole_root = Path(args.whole_shot_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    rc_all = 0
    for shot in sorted(set(shots)):
        r = _process_shot(
            shot=shot,
            precursor_npz=prec_root / str(shot) / "precursor.npz",
            whole_shot_root=whole_root,
            out_dir=out_dir,
            pf_xml=args.pf_xml,
            wall_xml=args.wall_xml,
        )
        if r != 0:
            rc_all = max(rc_all, r)
    return min(rc_all, 255)


if __name__ == "__main__":
    raise SystemExit(main())
