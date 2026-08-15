"""EAST 静态前向 GS 验证（仅 freegsnke）：BC pred（或 PCS）+ EFIT 剖面 → NK forward_solve.

``--overlay-only`` 仅写出 ``07_overlay_coils_lcfs_r8z8.png`` 与 ``summary.json``（跳过 00–06）。

剖面与 PCS 可来自 ``bc/precursor_export`` 落盘的 ``--precursor-npz`` / ``--precursor-slice-npz``，
此时 EFIT 不再读 MDS/h5；单帧 slice 模式跳过 BC 推理。
``--precursor-slice-npz`` 为 ``slice_pred_<ramp_up|flat_top|ramp_down>_*.npz`` 时，输出目录
``{out_dir}/{shot}/<阶段>/`` 与图注、``summary.json`` 的 ``phase`` 默认从文件名推断，避免与 CLI ``--phase`` 不一致时覆盖结果；也可用 ``--out-phase`` 强制。

须使用已安装 ``freegsnke[freegs4e]`` 的 **torch** conda 环境::

    source .../conda.sh && conda activate .../envs/torch
    python -m bc.run_freegsnke_eval --shot 158413 --phase flat_top

环境变量 ``NUMBA_CACHE_DIR`` 默认指向仓库 ``.numba_cache/``，避免 numba 在
只读 site-packages 下缓存失败。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
os.environ.setdefault("NUMBA_CACHE_DIR", str(_REPO / ".numba_cache"))
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np


def _overlay_one_line_title(shot: int, out_phase: str, t_sel: float, coil_source_for_summary: str) -> str:
    """07 叠图单行标题：PCS/Pred, shot#…, 阶段@t=…s"""
    cs = str(coil_source_for_summary)
    if cs in ("slice_npz:bc_pred", "bc"):
        src = "Pred"
    elif cs in ("slice_npz:efit_pcs_same_row", "pcs"):
        src = "PCS"
    elif cs == "slice_npz:ad_hoc":
        src = ""
    else:
        src = cs
    phase_disp = str(out_phase).replace("_", " ")
    if src:
        return f"{src}, #{int(shot)}, {phase_disp} @ $t$={float(t_sel):.2f} s"
    return f"#{int(shot)}, {phase_disp} @ $t$={float(t_sel):.2f} s"


def _infer_phase_from_slice_npz_path(path: Path) -> str | None:
    """从 slice 文件名推断阶段（输出子目录与图注）。

    支持 ``slice_pred_<phase>_``（BC 预测电流 slice）与
    ``slice_efit_self_<phase>_``（EFIT 自洽 PCS+Ip+剖面 slice）。
    """
    for pat in (
        r"slice_pred_(ramp_up|flat_top|ramp_down)(?:_t\d+)?_kbc",  # 可选 _t01_ 等多时间片命名
        r"slice_pred_(ramp_up|flat_top|ramp_down)_",  # 旧式 slice_pred_phase_kbc...
        r"slice_efit_self_(ramp_up|flat_top|ramp_down)_",
    ):
        m = re.search(pat, path.name, flags=re.IGNORECASE)
        if m:
            return m.group(1).lower()
    return None


def _infer_phase_from_time_ip(t_efit: float, ip_A: float, precursor_npz_path: str | Path | None, slice_path: str | Path | None = None) -> str | None:
    """根据 t_efit 与整炮 precursor ATIME/PCRL01 推断放电阶段。

    找到 precursor 中最近时间步对应的 phase_ids。若无 precursor 则返回 None。
    """
    p = None
    if precursor_npz_path is not None:
        p = Path(precursor_npz_path)
    if p is None or not p.is_file():
        # Try standard location: results/freegsnke_precursors/{shot}/precursor.npz
        if slice_path is not None:
            sp = Path(slice_path).resolve()
            # Walk up to find shot directory
            parent = sp.parent
            for _ in range(5):
                # Check if parent name looks like a shot number
                if parent.name.isdigit():
                    candidate = parent.parent.parent / "freegsnke_precursors" / parent.name / "precursor.npz"
                    # Also try: walk up from whole_shot/shot -> precursors/shot
                    if not candidate.is_file():
                        # Try: results/freegsnke_precursors/{shot}/precursor.npz
                        candidate = parent.parent / "freegsnke_precursors" / parent.name / "precursor.npz"
                    if candidate.is_file():
                        p = candidate
                        break
                parent = parent.parent
    if p is None or not p.is_file():
        return None
    try:
        from .precursor_export import load_precursor_npz
        bundle = load_precursor_npz(p)
        atime = np.asarray(bundle["ATIME"], dtype=np.float64).ravel()
        ipr = np.asarray(bundle["PCRL01"], dtype=np.float64).ravel()
        from bc.data.phases import phase_ids_per_step
        pids = phase_ids_per_step(atime, ipr)
        k = int(np.argmin(np.abs(atime - t_efit)))
        if k < len(pids):
            return ["ramp_up", "flat_top", "ramp_down"][int(pids[k])]
    except Exception:
        pass
    return None


def _point_to_segment_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    t = np.clip((apx * abx + apy * aby) / (abx * abx + aby * aby + 1e-30), 0.0, 1.0)
    qx, qy = ax + t * abx, ay + t * aby
    return float(np.hypot(px - qx, py - qy))


def rmse_separatrix_to_r8z8(sep: np.ndarray, r8: np.ndarray, z8: np.ndarray) -> float:
    """sep: (N,2) 开放或闭合；R8/Z8 为 8 点 EFIT 多边形。"""
    r8 = np.asarray(r8, dtype=np.float64).ravel()
    z8 = np.asarray(z8, dtype=np.float64).ravel()
    if sep is None or len(sep) < 3:
        return float("nan")
    d2 = []
    for i in range(sep.shape[0]):
        px, py = float(sep[i, 0]), float(sep[i, 1])
        dm = 1e30
        for k in range(8):
            ax, ay = r8[k], z8[k]
            bx, by = r8[(k + 1) % 8], z8[(k + 1) % 8]
            dm = min(dm, _point_to_segment_dist(px, py, ax, ay, bx, by))
        d2.append(dm * dm)
    return float(np.sqrt(np.mean(d2)))


def _default_domain_from_wall(lim_r: np.ndarray, lim_z: np.ndarray) -> tuple[float, float, float, float]:
    lim_r = np.asarray(lim_r, dtype=np.float64).ravel()
    lim_z = np.asarray(lim_z, dtype=np.float64).ravel()
    margin_r, margin_z = 0.12, 0.15
    Rmin = max(0.22, float(np.min(lim_r)) - margin_r)
    Rmax = min(3.45, float(np.max(lim_r)) + margin_r)
    Zmin = float(np.min(lim_z)) - margin_z
    Zmax = float(np.max(lim_z)) + margin_z
    return Rmin, Rmax, Zmin, Zmax


def _fvac(snapshot: dict) -> float:
    bc = abs(float(snapshot["bcentr"]))
    rm = float(snapshot.get("rmaxis", float("nan")))
    if np.isfinite(rm) and rm > 0.15:
        return float(rm * bc)
    r8 = np.asarray(snapshot["r8"], dtype=np.float64)
    return float(np.mean(r8) * bc)


def _plot_profiles(snapshot: dict, out: Path, *, phase_label: str | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(snapshot["pprime"])
    psi_n = np.linspace(0.0, 1.0, n)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    ax[0].plot(psi_n, snapshot["pprime"], "b-", lw=1.0)
    ax[0].set_xlabel(r"$\psi_N$")
    ax[0].set_ylabel("p'")
    ax[0].grid(True, alpha=0.25)
    ax[1].plot(psi_n, snapshot["ffprim"], "C1-", lw=1.0)
    ax[1].set_xlabel(r"$\psi_N$")
    ax[1].set_ylabel("FF'")
    ax[1].grid(True, alpha=0.25)
    ph = f" · {phase_label}" if phase_label else ""
    fig.suptitle(
        f"EFIT profiles shot={snapshot['shot']}{ph} t≈{snapshot['t_efit']:.3f}s ({snapshot.get('source', '')})"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_ip_time(
    time_s: np.ndarray,
    ip_a: np.ndarray,
    t_pick: float,
    out: Path,
    shot: int,
    *,
    title: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(time_s, ip_a * 1e-3, lw=0.9)
    ax.axvline(t_pick, color="C3", ls="--", lw=1.0, label=f"t={t_pick:.3f}s")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$I_p$ [kA]")
    ax.set_title(title or f"shot {shot}: PCRL01 (from state)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_pcpf_bar(names: list[str], amps: np.ndarray, out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    x = np.arange(len(names))
    ax.bar(x, amps * 1e-3, color="C0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("kA")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_equilibrium(eq, out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    psi = eq.psi()
    fig, ax = plt.subplots(figsize=(6.0, 7.5))
    lev = np.linspace(float(np.nanmin(psi)), float(np.nanmax(psi)), 31)
    ax.contour(eq.R, eq.Z, psi, levels=lev, colors="k", linewidths=0.35, alpha=0.55)
    lim = eq.tokamak.limiter
    if lim is not None:
        ax.plot(lim.R, lim.Z, "k-", lw=1.0)
    try:
        sep = np.asarray(eq.separatrix(ntheta=180))
        if sep.size > 0:
            ax.plot(sep[:, 0], sep[:, 1], "m-", lw=1.4, label="separatrix")
    except Exception:
        pass
    if getattr(eq, "opt", None) is not None and len(eq.opt):
        ax.plot(float(eq.opt[0, 0]), float(eq.opt[0, 1]), "r+", ms=10, label="O-point")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_compare_lcfs(eq, r8: np.ndarray, z8: np.ndarray, out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 7.5))
    lim = eq.tokamak.limiter
    if lim is not None:
        ax.fill(np.r_[lim.R, lim.R[0]], np.r_[lim.Z, lim.Z[0]], color="0.88", zorder=0)
        ax.plot(lim.R, lim.Z, "k-", lw=0.8)
    r8 = np.asarray(r8, dtype=np.float64).ravel()
    z8 = np.asarray(z8, dtype=np.float64).ravel()
    ax.plot(np.r_[r8, r8[0]], np.r_[z8, z8[0]], "b-", lw=1.4, label="EFIT R8/Z8")
    try:
        sep = np.asarray(eq.separatrix(ntheta=240))
        ax.plot(sep[:, 0], sep[:, 1], "C3--", lw=1.5, label="freegsnke LCFS")
    except Exception:
        pass
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_overlay_coils_lcfs_r8z8(
    eq,
    r8: np.ndarray,
    z8: np.ndarray,
    out: Path,
    title: str,
    *,
    pf_xml: str | Path,
    wall_xml: str | Path,
) -> None:
    """整机视图：青 PF/IC 框、品红 limiter、深绿真空室双 D（不进图例）+ EFIT R8/Z8 + LCFS + 磁轴。

    图例条目与原先一致：``EFIT boundary (R8/Z8)``、``Forward LCFS``、``Magnetic axis``。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as mpe
    import matplotlib.pyplot as plt

    from .freegsnke_east_machine import (
        active_coils_dict_from_pf_xml,
        pf_coils_rzwh_for_overlay,
        vessel_annular_rz_for_overlay,
    )

    # notebook 风格 PF 框 + 参考图配色（青 PF、深绿真空室 D、品红 limiter、浅蓝 Target、浅红 Prediction）
    _OVERLAY_COIL_NAMES = (
        "PF1", "PF2", "PF3", "PF4", "PF5", "PF6", "PF7", "PF8", "PF9", "PF10", "PF11", "PF12", "PF13", "PF14", "IC1", "IC2"
    )
    col_pf_ic = "#40C4C4"  # 浅青绿 PF/IC 方框（对照参考图）
    col_vessel = "#1A630E"  # 深绿真空室内外壁
    col_limiter = "magenta"  # 内壁 limiter 折线
    col_target = "#90E0EF"  # Target：浅蓝
    col_target_edge = "#5BC0DE"
    col_prediction = "#F08080"  # Prediction：浅珊瑚红
    _col_coil_fb = "#40C4C4"  # 回退几何时与主配色一致

    # Nature/Science 类图常见：无衬线、细坐标轴、高分辨率导出
    pub_rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 9,
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "axes.linewidth": 1.15,
        "xtick.major.width": 0.85,
        "ytick.major.width": 0.85,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
        "legend.frameon": True,
        "legend.edgecolor": "0.85",
        "legend.facecolor": "1.0",
        "legend.framealpha": 1.0,
        "legend.fontsize": 8,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }

    with plt.rc_context(pub_rc):
        fig, ax = plt.subplots(figsize=(5.15, 6.35))
        tok = eq.tokamak
        lim = tok.limiter
        lr = lz = None
        if lim is not None and len(getattr(lim, "R", ())):
            lr = np.asarray(lim.R, dtype=np.float64).ravel()
            lz = np.asarray(lim.Z, dtype=np.float64).ravel()
            ax.plot(np.r_[lr, lr[0]], np.r_[lz, lz[0]], color=col_limiter, lw=1.5, label="_nolegend_", zorder=2)
        w = getattr(tok, "wall", None)
        if w is not None and len(getattr(w, "R", ())) and w is not lim:
            wr = np.asarray(w.R, dtype=np.float64).ravel()
            wz = np.asarray(w.Z, dtype=np.float64).ravel()
            ax.plot(wr, wz, color="0.5", ls=(0, (3, 2)), lw=1.2, label="_nolegend_", zorder=1)

        inner_v, outer_v, _ = vessel_annular_rz_for_overlay(wall_xml)
        if inner_v is not None:
            ri, zi = inner_v
            ax.plot(
                np.r_[ri, ri[0]],
                np.r_[zi, zi[0]],
                color=col_vessel,
                ls="-",
                lw=2.0,
                zorder=2.4,
                label="_nolegend_",
            )
        if outer_v is not None:
            ro, zo = outer_v
            ax.plot(
                np.r_[ro, ro[0]],
                np.r_[zo, zo[0]],
                color=col_vessel,
                ls="-",
                lw=2.0,
                zorder=2.4,
                label="_nolegend_",
            )

        active_rects, _pf_src = pf_coils_rzwh_for_overlay(pf_xml)
        if not active_rects:
            active_fb = active_coils_dict_from_pf_xml(pf_xml)
            for nm, d in active_fb.items():
                cx, cy = float(d["R"][0]), float(d["Z"][0])
                w0, h0 = float(d["dR"]), float(d["dZ"])
                wv, hv = w0 * 1.08, h0 * 1.08
                rect = plt.Rectangle(
                    (cx - wv / 2, cy - hv / 2),
                    wv,
                    hv,
                    fill=False,
                    lw=1.0,
                    edgecolor=_col_coil_fb,
                    zorder=3,
                )
                ax.add_patch(rect)
                ax.text(
                    cx,
                    cy,
                    nm,
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    color="black",
                    zorder=22,
                    path_effects=[mpe.withStroke(linewidth=2.2, foreground="white")],
                )
        else:
            for node in _OVERLAY_COIL_NAMES:
                t = active_rects.get(node)
                if t is None:
                    continue
                cx, cy, w0, h0 = t
                node_box_r = [
                    cx - 0.5 * w0,
                    cx + 0.5 * w0,
                    cx + 0.5 * w0,
                    cx - 0.5 * w0,
                    cx - 0.5 * w0,
                ]
                node_box_z = [
                    cy - 0.5 * h0,
                    cy - 0.5 * h0,
                    cy + 0.5 * h0,
                    cy + 0.5 * h0,
                    cy - 0.5 * h0,
                ]
                ax.plot(
                    node_box_r,
                    node_box_z,
                    color=col_pf_ic,
                    lw=1.0,
                    zorder=3,
                    label="_nolegend_",
                )
                ax.plot(
                    [node_box_r[0], node_box_r[2]],
                    [node_box_z[0], node_box_z[2]],
                    color=col_pf_ic,
                    lw=0.5,
                    zorder=3,
                )
                ax.plot(
                    [node_box_r[1], node_box_r[3]],
                    [node_box_z[1], node_box_z[3]],
                    color=col_pf_ic,
                    lw=0.5,
                    zorder=3,
                )
                ax.text(
                    node_box_r[1],
                    (node_box_z[0] + node_box_z[3]) * 0.5,
                    node,
                    ha="left",
                    va="center",
                    fontsize=4.5,
                    color="black",
                    zorder=22,
                    path_effects=[mpe.withStroke(linewidth=1.6, foreground="white")],
                )

        r8a = np.asarray(r8, dtype=np.float64).ravel()
        z8a = np.asarray(z8, dtype=np.float64).ravel()
        ax.plot(
            np.r_[r8a, r8a[0]],
            np.r_[z8a, z8a[0]],
            color=col_target,
            lw=3.0,
            solid_capstyle="round",
            label="EFIT boundary (R8/Z8)",
            zorder=6,
        )
        ax.scatter(
            r8a,
            z8a,
            s=30,
            facecolors=col_target,
            edgecolors=col_target_edge,
            linewidths=0.6,
            zorder=7,
            label=None,
        )

        try:
            sep = np.asarray(eq.separatrix(ntheta=240))
            if sep.size:
                nan_mask = np.isnan(sep[:, 0]) | np.isnan(sep[:, 1])
                if not nan_mask.any():
                    segments = [sep]
                else:
                    breaks = np.where(nan_mask)[0]
                    segments = np.split(sep, breaks)
                    segments = [s[~np.isnan(s[:, 0]) & ~np.isnan(s[:, 1])]
                                for s in segments if s.size]
                for i, seg in enumerate(segments):
                    if seg.shape[0] < 2:
                        continue
                    ax.plot(
                        seg[:, 0],
                        seg[:, 1],
                        color=col_prediction,
                        ls="-",
                        lw=2.5,
                        solid_capstyle="round",
                        solid_joinstyle="round",
                        label="Forward LCFS" if i == 0 else None,
                        zorder=5,
                    )
        except Exception:
            pass

        if getattr(eq, "opt", None) is not None and len(eq.opt):
            ax.plot(
                float(eq.opt[0, 0]),
                float(eq.opt[0, 1]),
                "+",
                color="#CC0000",
                ms=9,
                mew=1.15,
                label="Magnetic axis",
                zorder=8,
            )

        rs: list[float] = []
        zs: list[float] = []
        if lr is not None and lr.size:
            rs.extend(lr.tolist())
            zs.extend(lz.tolist())
        if active_rects:
            for node in _OVERLAY_COIL_NAMES:
                t = active_rects.get(node)
                if t is None:
                    continue
                cx, cy, w0, h0 = t
                rs.extend([cx - 0.5 * w0, cx + 0.5 * w0])
                zs.extend([cy - 0.5 * h0, cy + 0.5 * h0])
        else:
            for d in active_fb.values():
                cx, cy = float(d["R"][0]), float(d["Z"][0])
                w0, h0 = float(d["dR"]), float(d["dZ"])
                wv, hv = w0 * 1.08, h0 * 1.08
                rs.extend([cx - wv / 2, cx + wv / 2])
                zs.extend([cy - hv / 2, cy + hv / 2])
        if inner_v is not None:
            rs.extend(inner_v[0].tolist())
            zs.extend(inner_v[1].tolist())
        if outer_v is not None:
            rs.extend(outer_v[0].tolist())
            zs.extend(outer_v[1].tolist())
        rs.extend(r8a.tolist())
        zs.extend(z8a.tolist())
        try:
            sep2 = np.asarray(eq.separatrix(ntheta=120))
            if sep2.size:
                rs.extend(sep2[:, 0].tolist())
                zs.extend(sep2[:, 1].tolist())
        except Exception:
            pass
        if rs and zs:
            span_r = max(rs) - min(rs)
            span_z = max(zs) - min(zs)
            mr = max(0.14, 0.07 * span_r)
            mz = max(0.14, 0.07 * span_z)
            ax.set_xlim(min(rs) - mr, max(rs) + mr)
            ax.set_ylim(min(zs) - mz, max(zs) + mz)

        ax.set_xlabel("$R$ (m)")
        ax.set_ylabel("$Z$ (m)")
        ax.set_title(title, pad=10)
        ax.set_aspect("equal")
        h, lab = ax.get_legend_handles_labels()
        uniq: dict[str, object] = {}
        for hi, li in zip(h, lab):
            if li and not str(li).startswith("_") and li not in uniq:
                uniq[li] = hi
        ax.legend(
            uniq.values(),
            uniq.keys(),
            loc="upper right",
            frameon=True,
            fancybox=False,
        )
        ax.grid(True, color="0.88", lw=0.45, ls="-", alpha=1.0)
        fig.tight_layout()
        fig.savefig(out, dpi=300)
        plt.close(fig)


def _plot_overlay_coils_lcfs_r8z8_shape_params(
    eq,
    r8: np.ndarray,
    z8: np.ndarray,
    out: Path,
    title: str,
    *,
    pf_xml: str | Path,
    wall_xml: str | Path,
) -> dict[str, list[float]] | None:
    """同 _plot_overlay_coils_lcfs_r8z8，但用 shape_params 将 Forward LCFS 转为 R8/Z8 八角形（点线）。

    返回 ``{"r8": [...], "z8": [...]}`` 或 *None*（计算失败时）。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as mpe
    import matplotlib.pyplot as plt

    from scan_data.compat import shape_params
    from .freegsnke_east_machine import (
        active_coils_dict_from_pf_xml,
        pf_coils_rzwh_for_overlay,
        vessel_annular_rz_for_overlay,
    )

    _OVERLAY_COIL_NAMES = (
        "PF1", "PF2", "PF3", "PF4", "PF5", "PF6", "PF7", "PF8", "PF9", "PF10", "PF11", "PF12", "PF13", "PF14", "IC1", "IC2"
    )
    col_pf_ic = "#40C4C4"
    col_vessel = "#1A630E"
    col_limiter = "magenta"
    col_target = "#90E0EF"
    col_target_edge = "#5BC0DE"
    col_prediction = "#F08080"
    _col_coil_fb = "#40C4C4"

    pub_rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 9,
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "axes.linewidth": 1.15,
        "xtick.major.width": 0.85,
        "ytick.major.width": 0.85,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
        "legend.frameon": True,
        "legend.edgecolor": "0.85",
        "legend.facecolor": "1.0",
        "legend.framealpha": 1.0,
        "legend.fontsize": 8,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }

    r8_fwd: np.ndarray | None = None
    z8_fwd: np.ndarray | None = None

    with plt.rc_context(pub_rc):
        fig, ax = plt.subplots(figsize=(5.15, 6.35))
        tok = eq.tokamak
        lim = tok.limiter
        lr = lz = None
        if lim is not None and len(getattr(lim, "R", ())):
            lr = np.asarray(lim.R, dtype=np.float64).ravel()
            lz = np.asarray(lim.Z, dtype=np.float64).ravel()
            ax.plot(np.r_[lr, lr[0]], np.r_[lz, lz[0]], color=col_limiter, lw=1.5, label="_nolegend_", zorder=2)
        w = getattr(tok, "wall", None)
        if w is not None and len(getattr(w, "R", ())) and w is not lim:
            wr = np.asarray(w.R, dtype=np.float64).ravel()
            wz = np.asarray(w.Z, dtype=np.float64).ravel()
            ax.plot(wr, wz, color="0.5", ls=(0, (3, 2)), lw=1.2, label="_nolegend_", zorder=1)

        inner_v, outer_v, _ = vessel_annular_rz_for_overlay(wall_xml)
        if inner_v is not None:
            ri, zi = inner_v
            ax.plot(np.r_[ri, ri[0]], np.r_[zi, zi[0]], color=col_vessel, ls="-", lw=2.0, zorder=2.4, label="_nolegend_")
        if outer_v is not None:
            ro, zo = outer_v
            ax.plot(np.r_[ro, ro[0]], np.r_[zo, zo[0]], color=col_vessel, ls="-", lw=2.0, zorder=2.4, label="_nolegend_")

        active_rects, _pf_src = pf_coils_rzwh_for_overlay(pf_xml)
        if not active_rects:
            active_fb = active_coils_dict_from_pf_xml(pf_xml)
            for nm, d in active_fb.items():
                cx, cy = float(d["R"][0]), float(d["Z"][0])
                w0, h0 = float(d["dR"]), float(d["dZ"])
                wv, hv = w0 * 1.08, h0 * 1.08
                rect = plt.Rectangle((cx - wv / 2, cy - hv / 2), wv, hv, fill=False, lw=1.0, edgecolor=_col_coil_fb, zorder=3)
                ax.add_patch(rect)
                ax.text(cx, cy, nm, ha="center", va="center", fontsize=5.2, color="black", zorder=22,
                        path_effects=[mpe.withStroke(linewidth=2.2, foreground="white")])
        else:
            for node in _OVERLAY_COIL_NAMES:
                t = active_rects.get(node)
                if t is None:
                    continue
                cx, cy, w0, h0 = t
                node_box_r = [cx - 0.5 * w0, cx + 0.5 * w0, cx + 0.5 * w0, cx - 0.5 * w0, cx - 0.5 * w0]
                node_box_z = [cy - 0.5 * h0, cy - 0.5 * h0, cy + 0.5 * h0, cy + 0.5 * h0, cy - 0.5 * h0]
                ax.plot(node_box_r, node_box_z, color=col_pf_ic, lw=1.0, zorder=3, label="_nolegend_")
                ax.plot([node_box_r[0], node_box_r[2]], [node_box_z[0], node_box_z[2]], color=col_pf_ic, lw=0.5, zorder=3)
                ax.plot([node_box_r[1], node_box_r[3]], [node_box_z[1], node_box_z[3]], color=col_pf_ic, lw=0.5, zorder=3)
                ax.text(node_box_r[1], (node_box_z[0] + node_box_z[3]) * 0.5, node, ha="left", va="center",
                        fontsize=4.5, color="black", zorder=22,
                        path_effects=[mpe.withStroke(linewidth=1.6, foreground="white")])

        # --- EFIT R8/Z8 (solid light blue) ---
        r8a = np.asarray(r8, dtype=np.float64).ravel()
        z8a = np.asarray(z8, dtype=np.float64).ravel()
        ax.plot(np.r_[r8a, r8a[0]], np.r_[z8a, z8a[0]], color=col_target, ls="--", lw=2.0,
                dashes=(8, 5), label=r"EFIT $(R_8,\,Z_8)$", zorder=6)
        ax.scatter(r8a, z8a, s=30, facecolors=col_target, edgecolors=col_target_edge,
                   linewidths=0.6, zorder=7, label=None)

        # --- Forward LCFS → R8/Z8 via shape_params (dotted coral red) ---
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
                                color=col_prediction, lw=2.5,
                                label=r"FreeGSNKE $(R_8,\,Z_8)$", zorder=5)
                        ax.scatter(r8_fwd, z8_fwd, s=30, facecolors=col_prediction,
                                   edgecolors="darkred", linewidths=0.6, zorder=7, label=None)
        except Exception:
            pass

        # --- Magnetic axis ---
        if getattr(eq, "opt", None) is not None and len(eq.opt):
            ax.plot(float(eq.opt[0, 0]), float(eq.opt[0, 1]), "+",
                    color="#CC0000", ms=9, mew=1.15, label="Magnetic axis", zorder=7)

        # --- Axis range ---
        rs: list[float] = []
        zs: list[float] = []
        if lr is not None and lr.size:
            rs.extend(lr.tolist())
            zs.extend(lz.tolist())
        if active_rects:
            for node in _OVERLAY_COIL_NAMES:
                t = active_rects.get(node)
                if t is None:
                    continue
                cx, cy, w0, h0 = t
                rs.extend([cx - 0.5 * w0, cx + 0.5 * w0])
                zs.extend([cy - 0.5 * h0, cy + 0.5 * h0])
        else:
            for d in active_fb.values():
                cx, cy = float(d["R"][0]), float(d["Z"][0])
                w0, h0 = float(d["dR"]), float(d["dZ"])
                wv, hv = w0 * 1.08, h0 * 1.08
                rs.extend([cx - wv / 2, cx + wv / 2])
                zs.extend([cy - hv / 2, cy + hv / 2])
        if inner_v is not None:
            rs.extend(inner_v[0].tolist())
            zs.extend(inner_v[1].tolist())
        if outer_v is not None:
            rs.extend(outer_v[0].tolist())
            zs.extend(outer_v[1].tolist())
        rs.extend(r8a.tolist())
        zs.extend(z8a.tolist())
        if r8_fwd is not None:
            rs.extend(r8_fwd.tolist())
            zs.extend(z8_fwd.tolist())
        if rs and zs:
            span_r = max(rs) - min(rs)
            span_z = max(zs) - min(zs)
            mr = max(0.14, 0.07 * span_r)
            mz = max(0.14, 0.07 * span_z)
            ax.set_xlim(min(rs) - mr, max(rs) + mr)
            ax.set_ylim(min(zs) - mz, max(zs) + mz)

        ax.set_xlabel("$R$ (m)")
        ax.set_ylabel("$Z$ (m)")
        ax.set_title(title, pad=10)
        ax.set_aspect("equal")
        h, lab = ax.get_legend_handles_labels()
        uniq: dict[str, object] = {}
        for hi, li in zip(h, lab):
            if li and not str(li).startswith("_") and li not in uniq:
                uniq[li] = hi
        ax.legend(uniq.values(), uniq.keys(), loc="upper right", frameon=True, fancybox=False, edgecolor="0.8")
        ax.grid(True, color="0.88", lw=0.45, ls="-", alpha=1.0)
        fig.tight_layout()
        fig.savefig(out, dpi=300)
        plt.close(fig)

    if r8_fwd is not None and z8_fwd is not None:
        return {"r8": r8_fwd.tolist(), "z8": z8_fwd.tolist()}
    return None


def _plot_env_check(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import freegsnke

    try:
        import torch

        tv = torch.__version__
    except Exception as exc:  # noqa: BLE001 — login 节点常缺 libgalaxyhip
        tv = f"(unavailable: {exc})"

    lines = [
        f"freegsnke {getattr(freegsnke, '__version__', '?')}",
        f"torch {tv}",
        f"NUMBA_CACHE_DIR={os.environ.get('NUMBA_CACHE_DIR','')}",
    ]
    fig, ax = plt.subplots(figsize=(5.0, 1.2))
    ax.axis("off")
    ax.text(0.02, 0.5, "\n".join(lines), va="center", family="monospace", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _forward_solve_profile_fixed(eq, profiles, *, rtol: float, maxit: int) -> float:
    from freegsnke.GSstaticsolver import NKGSsolver

    nk = NKGSsolver(eq, seed=42)
    nk.forward_solve(
        eq,
        profiles,
        target_relative_tolerance=rtol,
        max_solving_iterations=maxit,
        suppress=True,
        verbose=False,
    )
    return float(nk.relative_change)


def main() -> int:
    from .east_pf_mapping import PCPF_NAMES, pcpf12_to_pf14_amps
    from .freegsnke_east_machine import (
        PF_XML_DEFAULT,
        WALL_XML_DEFAULT,
        build_east_tokamak_freegsnke,
        currents_dict_to_vec,
        plot_machine_rz,
        pf_coils_rzwh_for_overlay,
        vessel_annular_rz_for_overlay,
        wall_outline_rz_from_xml,
    )
    from .infer_one_shot import infer_one_shot, pick_representative_steps
    from .mds_efit_snapshot import EFIT_DB_DEFAULT, load_efit_snapshot
    from .precursor_export import (
        efit_snapshot_row,
        load_precursor_npz,
        load_single_slice_npz,
        nearest_row,
    )

    ap = argparse.ArgumentParser(description="freegsnke static forward GS eval (EAST)")
    ap.add_argument("--shot", type=int, default=158413)
    ap.add_argument("--ckpt", type=str, default="results/bc_v1/run1/checkpoints/best_val.pt")
    ap.add_argument("--out-dir", type=str, default="results/freegsnke_eval")
    ap.add_argument("--phase", type=str, default="flat_top", choices=("ramp_up", "flat_top", "ramp_down"))
    ap.add_argument("--efit-dir", type=str, default=str(EFIT_DB_DEFAULT))
    ap.add_argument(
        "--efit-source",
        type=str,
        default="auto",
        choices=("auto", "h5", "mds"),
        help="auto: 有 EFIT h5 则用 h5，否则 MDS（与 notebook/05 §5 一致）",
    )
    ap.add_argument("--prefer-mds", action="store_true", help="优先 MDS，失败再回退 h5")
    ap.add_argument(
        "--mds-server",
        type=str,
        default=None,
        help="覆盖 MDS_HOSTNAME（notebook 05: efit_east_path / pcs_east_path）",
    )
    ap.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="覆盖 ckpt 内 dataset_root（仍缺 {shot}.h5 时用 MDS 建样本）",
    )
    ap.add_argument("--pf-xml", type=str, default=str(PF_XML_DEFAULT))
    ap.add_argument("--wall-xml", type=str, default=str(WALL_XML_DEFAULT))
    ap.add_argument("--nx", type=int, default=129)
    ap.add_argument("--ny", type=int, default=129)
    ap.add_argument("--rtol", type=float, default=8e-3)
    ap.add_argument("--maxit", type=int, default=80)
    ap.add_argument("--device", type=str, default=None, help="torch device for BC infer (default cuda:0 or cpu)")
    ap.add_argument(
        "--precursor-npz",
        type=str,
        default=None,
        help="全时间序列 precursor（bc/precursor_export 生成）；EFIT 剖面从文件取帧，不再读 MDS/h5",
    )
    ap.add_argument(
        "--precursor-k",
        type=int,
        default=None,
        help="与 --precursor-npz 连用：强制使用第 k 帧（否则按 BC 代表时刻对 ATIME 最近邻）",
    )
    ap.add_argument(
        "--precursor-slice-npz",
        type=str,
        default=None,
        help="单帧 slice（export_slice_npz）；剖面+PCS 电流+Ip 全离线，跳过 BC 推理与数据库",
    )
    ap.add_argument(
        "--out-phase",
        type=str,
        default=None,
        choices=("ramp_up", "flat_top", "ramp_down"),
        help="写入 {out_dir}/{shot}/<本参数>/ 并在图注、summary 中用该阶段名；"
        "slice 模式默认从文件名 slice_pred_<ramp_up|flat_top|ramp_down>_*.npz 自动推断，避免与 --phase 不一致时互相覆盖",
    )
    ap.add_argument(
        "--out-root-exact",
        type=str,
        default=None,
        help="若指定：所有输出直接写入该路径（不再拼接 {out_dir}/{shot}/{phase}）。"
        "用于批量 t1..tN 等目录结构；与 --out-dir 互斥于路径拼接（本参数优先）。",
    )
    ap.add_argument(
        "--coil-source",
        type=str,
        default="bc",
        choices=("bc", "pcs"),
        help="bc: PCPF 用 BC 预测；pcs: 与 EFIT 同帧的 PCS（需 precursor 中有 PCPF）",
    )
    ap.add_argument(
        "--overlay-only",
        action="store_true",
        help="仅生成 07_overlay_coils_lcfs_r8z8.png 与 summary.json；跳过 00_env_check–06_compare_lcfs",
    )
    ap.add_argument(
        "--shape-params-overlay",
        action="store_true",
        help="将 Forward LCFS 替换为 shape_params 计算的 R8/Z8 八角形点线",
    )
    args = ap.parse_args()

    if args.precursor_npz and args.precursor_slice_npz:
        raise SystemExit("不可同时指定 --precursor-npz 与 --precursor-slice-npz")
    if args.coil_source == "pcs" and not (args.precursor_npz or args.precursor_slice_npz):
        raise SystemExit("--coil-source pcs 需要 precursor 文件")
    if args.out_root_exact and not (args.precursor_slice_npz):
        raise SystemExit("--out-root-exact 仅适用于单帧 slice 模式（与 --precursor-slice-npz 连用）")

    slice_mode = args.precursor_slice_npz is not None
    bundle_mode = args.precursor_npz is not None
    slice_p = Path(args.precursor_slice_npz).resolve() if slice_mode and args.precursor_slice_npz else None

    if args.out_phase is not None:
        out_phase = args.out_phase
    elif slice_p is not None:
        inferred = _infer_phase_from_slice_npz_path(slice_p)
        if inferred is not None:
            out_phase = inferred
            if inferred != args.phase:
                print(
                    f"[run_freegsnke_eval] 输出目录与标注使用 slice 文件名推断阶段 {inferred!r} "
                    f"（--phase={args.phase!r}）；显式指定请用 --out-phase",
                    flush=True,
                )
        else:
            # Filename has no phase; try auto-detect from precursor data
            out_phase = args.phase  # fallback
    else:
        out_phase = args.phase

    if args.out_root_exact:
        out_root = Path(args.out_root_exact).resolve()
    else:
        out_root = Path(args.out_dir) / str(args.shot) / out_phase
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.overlay_only:
        _plot_env_check(out_root / "00_env_check.png")

    tok = build_east_tokamak_freegsnke(args.pf_xml, args.wall_xml, quiet=True)
    if not args.overlay_only:
        plot_machine_rz(
            tok,
            out_root / "01_machine_rz.png",
            pf_xml=args.pf_xml,
            title=f"shot {args.shot} · {out_phase} · EAST coils + limiter",
        )

    def _default_device() -> str:
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    device = args.device or _default_device()

    if slice_mode:
        sl = load_single_slice_npz(Path(args.precursor_slice_npz))
        snap = sl["snapshot"]
        t_sel = float(snap["t_efit"])
        k = int(snap["k_efit"])
        pcpf12 = np.asarray(sl["pcpf12"], dtype=np.float64).ravel()
        ip_sel = float(sl["Ip_A"])
        # Auto-detect phase from precursor if not explicitly set and filename has no phase
        if args.out_phase is None and _infer_phase_from_slice_npz_path(slice_p) is None:
            detected = _infer_phase_from_time_ip(t_sel, ip_sel, None, slice_path=slice_p)
            if detected is not None:
                out_phase = detected
                print(f"[run_freegsnke_eval] 自动推断阶段: {detected!r}", flush=True)
        if not args.overlay_only:
            _plot_ip_time(
                np.array([t_sel], dtype=np.float64),
                np.array([ip_sel], dtype=np.float64),
                t_sel,
                out_root / "02_ip_time.png",
                args.shot,
                title=f"{args.shot}, {out_phase}, Ip (slice) t={t_sel:.4f}s",
            )
    else:
        inf = infer_one_shot(
            args.shot,
            args.ckpt,
            device=device,
            dataset_root=args.dataset_root,
            mds_server=args.mds_server,
        )
        picks = pick_representative_steps(inf["phase_slices"])
        if args.phase not in picks:
            raise SystemExit(f"phase {args.phase!r} not available for shot {args.shot}; have {list(picks)}")
        k = int(picks[args.phase])
        t_sel = float(inf["time"][k])
        ip_sel = float(inf["Ip_A"][k])

        if not args.overlay_only:
            _plot_ip_time(
                inf["time"],
                inf["Ip_A"],
                t_sel,
                out_root / "02_ip_time.png",
                args.shot,
                title=f"{args.shot}, {out_phase}, PCRL01 @ t={t_sel:.4f}s",
            )

        if bundle_mode:
            bundle = load_precursor_npz(Path(args.precursor_npz))
            k_snap = int(args.precursor_k) if args.precursor_k is not None else nearest_row(bundle["ATIME"], t_sel)
            if args.precursor_k is None:
                t_row = float(np.asarray(bundle["ATIME"], dtype=np.float64).ravel()[k_snap])
                print(
                    f"[precursor] 最近邻行 k={k_snap} ATIME={t_row:.6f}s "
                    f"(BC 代表步 t={t_sel:.6f}s, Δt={t_row - t_sel:+.6e}s)",
                    flush=True,
                )
            snap = efit_snapshot_row(bundle, k_snap, shot=args.shot)
        else:
            snap = load_efit_snapshot(
                args.shot,
                t_sel,
                efit_dir=args.efit_dir,
                prefer_mds=args.prefer_mds,
                mds_server=args.mds_server,
                efit_source=args.efit_source,
            )

        if bundle_mode and args.coil_source == "pcs":
            pcpf12 = np.asarray(bundle["PCPF"][k_snap], dtype=np.float64).ravel()
            ip_sel = float(np.asarray(bundle["PCRL01"], dtype=np.float64).ravel()[k_snap])
        else:
            pcpf12 = inf["pred_A"][k]

    if not args.overlay_only:
        _plot_profiles(snap, out_root / "03_profiles.png", phase_label=out_phase)

    if slice_mode and slice_p is not None:
        sn = slice_p.name.lower()
        if "slice_pred_" in sn:
            coil_tag = "BC pred (slice npz)"
        elif sn.startswith("slice_whole_bc_"):
            coil_tag = "BC pred (whole-shot slice)"
        elif sn.startswith("slice_whole_pcs_") or "slice_efit_self_" in sn or sn.startswith("slice_k"):
            coil_tag = "PCS EFIT row (slice npz)"
        else:
            coil_tag = "PCPF in slice npz"
    else:
        coil_tag = "PCS (precursor)" if (bundle_mode and args.coil_source == "pcs") else "BC pred"

    if slice_mode and slice_p is not None:
        sn = slice_p.name.lower()
        if "slice_pred_" in sn:
            coil_source_for_summary = "slice_npz:bc_pred"
        elif sn.startswith("slice_whole_bc_"):
            coil_source_for_summary = "slice_npz:bc_pred"
        elif sn.startswith("slice_whole_pcs_") or "slice_efit_self_" in sn or sn.startswith("slice_k"):
            coil_source_for_summary = "slice_npz:efit_pcs_same_row"
        else:
            coil_source_for_summary = "slice_npz:ad_hoc"
    else:
        coil_source_for_summary = args.coil_source

    if not args.overlay_only:
        _plot_pcpf_bar(
            list(PCPF_NAMES),
            pcpf12,
            out_root / "04_pcpf_bar.png",
            f"{args.shot}, {out_phase}, {coil_tag} @ t={t_sel:.3f}s",
        )
    pf_amps = pcpf12_to_pf14_amps(pcpf12)

    lim_r, lim_z = wall_outline_rz_from_xml(args.wall_xml)
    Rmin, Rmax, Zmin, Zmax = _default_domain_from_wall(lim_r, lim_z)

    from freegsnke.equilibrium_update import Equilibrium
    from freegsnke.jtor_update import ConstrainBetapIp, GeneralPprimeFFprime

    vec = currents_dict_to_vec(tok, pf_amps)

    fvac = _fvac(snap)
    Ip = float(ip_sel)
    psi_n = np.linspace(0.0, 1.0, len(snap["pprime"]), dtype=np.float64)

    def _make_fresh_equilibrium():
        e = Equilibrium(tok, Rmin, Rmax, Zmin, Zmax, nx=args.nx, ny=args.ny, order=4)
        tok.set_all_coil_currents(vec)
        try:
            e.adjust_psi_plasma()
        except Exception:
            pass
        return e

    rel_final = 1e9
    profile_method = "failed"
    profiles = None
    eq = _make_fresh_equilibrium()
    try:
        profiles = GeneralPprimeFFprime(
            eq,
            Ip,
            fvac,
            psi_n,
            pprime_data=np.asarray(snap["pprime"], dtype=np.float64),
            ffprime_data=np.asarray(snap["ffprim"], dtype=np.float64),
            p_data=None,
            f_data=None,
            Raxis=float(snap["rmaxis"]),
            Ip_logic=True,
        )
        rel_final = _forward_solve_profile_fixed(eq, profiles, rtol=args.rtol, maxit=args.maxit)
        profile_method = "pprime"
    except Exception:
        eq = _make_fresh_equilibrium()
        profiles = ConstrainBetapIp(
            eq,
            float(snap["betap"]),
            Ip,
            fvac,
            Raxis=float(snap["rmaxis"]),
        )
        rel_final = _forward_solve_profile_fixed(eq, profiles, rtol=args.rtol, maxit=args.maxit)
        profile_method = "betap"

    converged = rel_final <= float(args.rtol)
    sep = None
    lcfs_rmse = float("nan")
    try:
        sep = np.asarray(eq.separatrix(ntheta=240))
        lcfs_rmse = rmse_separatrix_to_r8z8(sep, snap["r8"], snap["z8"])
    except Exception:
        pass

    if not args.overlay_only:
        _plot_equilibrium(
            eq,
            out_root / "05_equilibrium_psi.png",
            f"shot {args.shot} {out_phase} freegsnke ({profile_method}) rel={rel_final:.2e}",
        )
        _plot_compare_lcfs(
            eq,
            snap["r8"],
            snap["z8"],
            out_root / "06_compare_lcfs.png",
            f"{args.shot}, {out_phase}, t≈{t_sel:.3f}s",
        )
    sp_r8z8_result = None
    if args.shape_params_overlay:
        sp_r8z8_result = _plot_overlay_coils_lcfs_r8z8_shape_params(
            eq,
            snap["r8"],
            snap["z8"],
            out_root / "07_overlay_coils_lcfs_r8z8.png",
            _overlay_one_line_title(args.shot, out_phase, t_sel, coil_source_for_summary),
            pf_xml=args.pf_xml,
            wall_xml=args.wall_xml,
        )
    else:
        _plot_overlay_coils_lcfs_r8z8(
            eq,
            snap["r8"],
            snap["z8"],
            out_root / "07_overlay_coils_lcfs_r8z8.png",
            _overlay_one_line_title(args.shot, out_phase, t_sel, coil_source_for_summary),
            pf_xml=args.pf_xml,
            wall_xml=args.wall_xml,
        )

    r_mag = float("nan")
    z_mag = float("nan")
    if getattr(eq, "opt", None) is not None and len(eq.opt):
        r_mag = float(eq.opt[0, 0])
        z_mag = float(eq.opt[0, 1])

    _, _, wall_annular_src = vessel_annular_rz_for_overlay(args.wall_xml)
    _, pf_overlay_src = pf_coils_rzwh_for_overlay(args.pf_xml)

    summary = {
        "shot": args.shot,
        "phase": out_phase,
        "phase_cli": args.phase,
        "overlay_only": bool(args.overlay_only),
        "step_index": k,
        "time_s": t_sel,
        "k_efit_profile": int(snap.get("k_efit", -1)),
        "Ip_input_A": Ip,
        "profile_method": profile_method,
        "profile_source": snap.get("source", "h5"),
        "precursor_slice_npz": args.precursor_slice_npz,
        "precursor_npz": args.precursor_npz,
        "precursor_k": args.precursor_k,
        "coil_source": coil_source_for_summary,
        "forward_rel_residual": rel_final,
        "forward_converged": converged,
        "rtol_requested": float(args.rtol),
        "betap_efit": float(snap["betap"]),
        "bcentr_efit": float(snap["bcentr"]),
        "fvac_used": fvac,
        "Rmag": r_mag,
        "Zmag": z_mag,
        "Ip_eq_A": float(getattr(eq, "_current", float("nan"))),
        "lcfs_rmsep_to_r8z8_m": lcfs_rmse,
        "out_dir": str(out_root.resolve()),
        "wall_xml_machine": str(Path(args.wall_xml).resolve()),
        "wall_xml_annular_overlay": str(wall_annular_src.resolve()) if wall_annular_src else "",
        "pf_xml_machine": str(Path(args.pf_xml).resolve()),
        "pf_xml_overlay": str(pf_overlay_src.resolve()) if pf_overlay_src else "",
    }
    if args.out_root_exact:
        summary["out_root_exact"] = True
    if args.shape_params_overlay:
        summary["shape_params_overlay"] = True
        if sp_r8z8_result is not None:
            summary["r8_calc_from_lcfs"] = sp_r8z8_result["r8"]
            summary["z8_calc_from_lcfs"] = sp_r8z8_result["z8"]
    if slice_p is not None:
        summary["precursor_slice_basename"] = slice_p.name
        m_kbc = re.search(r"kbc(\d+)", slice_p.name, flags=re.IGNORECASE)
        m_kef = re.search(r"kef(\d+)", slice_p.name, flags=re.IGNORECASE)
        if m_kbc:
            summary["k_bc_step_from_slice_filename"] = int(m_kbc.group(1))
        if m_kef:
            summary["k_efit_step_from_slice_filename"] = int(m_kef.group(1))
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
