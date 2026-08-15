#!/usr/bin/env python3
"""每炮每阶段取 PCS 与 bc_pred「LCFS RMSE 之和最优」的同一时刻帧，拼 2×3 对比图。

布局 (2 行 × 3 列 + 底部 Ip)：
    上行 Pred:   ramp_up   flat_top   ramp_down
    下行 PCS:    ramp_up   flat_top   ramp_down
    底部共享:    Ip(t) 时间演化，宽度与上方严格对齐

每阶段在两个 branch（pcs / bc_pred）共有的时间步中，取
``lcfs_rmsep_to_r8z8_m`` 之和最小的一帧，保证上行与下行使用完全
相同的时间切片。

输出::

    {out_dir}/montage_pcs_vs_pred_shot{shot}.png

示例::

    python -m bc.batch_freegsnke.montage_pcs_vs_pred --shot 134925
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from bc.data.phases import PHASE_NAMES, phase_ids_per_step
from bc.gs_forward.precursor_export import load_precursor_npz

_REPO = Path(__file__).resolve().parent.parent.parent
_DIR_K_RE = re.compile(r"^k(\d+)_t", re.IGNORECASE)
OVERLAY_NAME = "07_overlay_coils_lcfs_r8z8.png"


def _k_from_dirname(d: Path) -> int | None:
    m = _DIR_K_RE.match(d.name)
    return int(m.group(1), 10) if m else None


def _load_frame_metrics(by_t_efit: Path) -> dict[int, dict[str, Any]]:
    """扫描 by_t_efit/*/summary.json，返回 {k: metrics_dict}。"""
    rows: dict[int, dict[str, Any]] = {}
    if not by_t_efit.is_dir():
        return rows
    for d in sorted(by_t_efit.iterdir()):
        if not d.is_dir():
            continue
        sj = d / "summary.json"
        if not sj.is_file():
            continue
        try:
            data = json.loads(sj.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        k = int(data.get("k_efit_profile", -1))
        if k < 0:
            kk = _k_from_dirname(d)
            if kk is None:
                continue
            k = kk
        rmsep = data.get("lcfs_rmsep_to_r8z8_m")
        if rmsep is None:
            rmsep = float("nan")
        else:
            rmsep = float(rmsep)
        rows[k] = {
            "eval_dir": str(d.resolve()),
            "k_efit": k,
            "time_s": float(data.get("time_s", float("nan"))),
            "lcfs_rmsep_to_r8z8_m": rmsep,
            "forward_converged": bool(data.get("forward_converged", False)),
            "overlay_png": str((d / OVERLAY_NAME).resolve()),
        }
    return rows


def _best_shared_k(
    pcs_rows: dict[int, dict[str, Any]],
    bc_rows: dict[int, dict[str, Any]],
    phase_ids: np.ndarray,
    phase: str,
    *,
    converged_only: bool,
) -> tuple[int, dict[str, Any], dict[str, Any]] | None:
    """在两个 branch 共有的 k 中，找 LCFS RMSE 之和最小的。"""
    pid = PHASE_NAMES.index(phase)
    common_ks = set(pcs_rows.keys()) & set(bc_rows.keys())
    best: tuple[float, int, dict, dict] | None = None
    for k in common_ks:
        if k < 0 or k >= phase_ids.size:
            continue
        if int(phase_ids[k]) != pid:
            continue
        pr = pcs_rows[k]
        br = bc_rows[k]
        if converged_only:
            if not pr["forward_converged"] or not br["forward_converged"]:
                continue
        pv = float(pr["lcfs_rmsep_to_r8z8_m"])
        bv = float(br["lcfs_rmsep_to_r8z8_m"])
        if not (math.isfinite(pv) and math.isfinite(bv)):
            continue
        s = pv + bv
        if best is None or s < best[0]:
            best = (s, k, pr, br)
    if best is None:
        return None
    _, k, pr, br = best
    return (k, pr, br)


def _interp_ip(atime: np.ndarray, ip: np.ndarray, t: float) -> float:
    m = np.isfinite(atime) & np.isfinite(ip)
    if not np.any(m):
        return float("nan")
    a = np.asarray(atime[m], dtype=np.float64)
    y = np.asarray(ip[m], dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    a = a[order]
    y = y[order]
    t = float(t)
    if t <= a[0]:
        return float(y[0])
    if t >= a[-1]:
        return float(y[-1])
    return float(np.interp(t, a, y))


_PHASE_COLORS = {
    "ramp_up": "#0ea5e9",
    "flat_top": "#14b8a6",
    "ramp_down": "#e11d48",
}


def _render_ip_strip(
    *,
    total_w_px: int,
    plot_h_px: int,
    dpi: int,
    shot: int,
    atime: np.ndarray,
    ip: np.ndarray,
    picks: dict[str, tuple[int, dict, dict] | None],
    resample: int,
) -> Any:
    """Matplotlib 光栅化 Ip(t)，宽度与上方拼接图严格对齐。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    w_in = total_w_px / dpi
    h_in = plot_h_px / dpi
    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=dpi)
    ip_a = np.asarray(ip, dtype=np.float64)
    ip_ka = ip_a / 1000.0
    at_a = np.asarray(atime, dtype=np.float64)
    m = np.isfinite(at_a) & np.isfinite(ip_a)
    ax.plot(at_a[m], ip_ka[m], color="black", lw=1.5, alpha=0.92, zorder=1)

    for phase in PHASE_NAMES:
        pick = picks.get(phase)
        if pick is None:
            continue
        _k, pcs_info, _bc_info = pick
        ts = float(pcs_info["time_s"])
        if not math.isfinite(ts):
            continue
        iy = _interp_ip(at_a, ip_ka, ts)
        if math.isfinite(iy):
            ax.scatter([ts], [iy], s=80, facecolors="none", edgecolors="red",
                       linewidths=1.8, zorder=5)
            dy_px = -8 if phase != "ramp_down" else 8
            va = "top" if phase != "ramp_down" else "bottom"
            ax.annotate(
                f"{ts:.2f} s",
                xy=(ts, iy),
                xytext=(6, dy_px),
                textcoords="offset points",
                ha="left", va=va,
                fontsize=8, color="black", fontweight="medium",
                clip_on=False, zorder=6,
                path_effects=[
                    pe.Stroke(linewidth=2.8, foreground="white"),
                    pe.Normal(),
                ],
            )

    ax.set_xlabel(r"$t$ (s)", fontsize=9)
    ax.set_ylabel(r"$I_{\mathrm{p}}$ (kA)", fontsize=9)
    ax.grid(True, axis="y", ls=":", alpha=0.35, color="0.70")
    ax.tick_params(labelsize=8)
    ax.margins(x=0.03, y=0.18)
    fig.tight_layout(pad=0.5)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    im = Image.open(buf).convert("RGB")
    if im.size[0] != total_w_px or im.size[1] != plot_h_px:
        im = im.resize((total_w_px, plot_h_px), resample)
    return im


def _render_figure(
    *,
    shot: int,
    picks: dict[str, tuple[int, dict, dict] | None],
    atime: np.ndarray,
    ip: np.ndarray,
    out_png: Path,
    dpi: int = 300,
) -> None:
    """PIL 像素级拼接：上行 Pred、下行 PCS（3 列），底部 Ip(t) 宽度对齐。"""
    from PIL import Image

    try:
        _resample = Image.Resampling.LANCZOS
    except AttributeError:
        _resample = Image.LANCZOS

    gap_px = 4
    vgap_px = 4
    rows_order = ["bc_pred", "pcs"]  # 上行 Prediction，下行 PCS
    row_tags = ["Pred", "PCS"]

    # --- 收集 6 张 overlay ---
    loaded: dict[str, dict[str, Image.Image | None]] = {}
    for tag in rows_order:
        loaded[tag] = {}
    for phase in PHASE_NAMES:
        pick = picks.get(phase)
        if pick is None:
            for tag in rows_order:
                loaded[tag][phase] = None
            continue
        _k, pcs_info, bc_info = pick
        info_map = {"pcs": pcs_info, "bc_pred": bc_info}
        for tag in rows_order:
            png = Path(info_map[tag]["overlay_png"])
            loaded[tag][phase] = Image.open(png).convert("RGBA") if png.is_file() else None

    # --- 统一高度 ---
    all_ims = [im for tag in rows_order for im in loaded[tag].values() if im is not None]
    if not all_ims:
        return
    max_h = max(im.size[1] for im in all_ims)

    def _scale(im: Image.Image) -> Image.Image:
        if im.size[1] == max_h:
            return im
        new_w = max(1, int(round(im.size[0] * max_h / im.size[1])))
        return im.resize((new_w, max_h), _resample)

    scaled: dict[str, dict[str, Image.Image]] = {}
    widths_ok: list[int] = []
    for tag in rows_order:
        scaled[tag] = {}
        for phase in PHASE_NAMES:
            im = loaded[tag].get(phase)
            if im is not None:
                s = _scale(im)
                scaled[tag][phase] = s
                widths_ok.append(s.size[0])
            else:
                scaled[tag][phase] = Image.new("RGBA", (400, max_h), (240, 240, 240, 255))
                widths_ok.append(400)
    ph_w = int(np.median(widths_ok)) if widths_ok else 400

    # 统一每列宽度为 3 列中最大宽度
    col_widths: list[int] = []
    for phase in PHASE_NAMES:
        cw = max(scaled[tag][phase].size[0] for tag in rows_order)
        col_widths.append(cw)
    total_w = sum(col_widths) + gap_px * (len(PHASE_NAMES) - 1)

    # --- 拼接上、下两行 ---
    row_images: list[Image.Image] = []
    for tag in rows_order:
        panels: list[Image.Image] = []
        for i, phase in enumerate(PHASE_NAMES):
            im = scaled[tag][phase]
            cw = col_widths[i]
            if im.size[0] != cw:
                new_im = Image.new("RGBA", (cw, max_h), (255, 255, 255, 255))
                x_off = (cw - im.size[0]) // 2
                new_im.paste(im, (x_off, 0), im)
                im = new_im
            panels.append(im)
        row_w = sum(p.size[0] for p in panels) + gap_px * (len(panels) - 1)
        row_im = Image.new("RGBA", (row_w, max_h), (255, 255, 255, 255))
        x = 0
        for j, p in enumerate(panels):
            row_im.paste(p, (x, 0), p)
            x += p.size[0] + (gap_px if j < len(panels) - 1 else 0)
        row_images.append(row_im)

    total_h_top = len(row_images) * max_h + vgap_px * (len(row_images) - 1)
    top_rgb = Image.new("RGB", (total_w, total_h_top), (255, 255, 255))
    y = 0
    for idx, rim in enumerate(row_images):
        # 居中
        x_off = (total_w - rim.size[0]) // 2
        top_rgb.paste(rim, (x_off, y))
        y += max_h + vgap_px

    # --- 底部 Ip(t) ---
    plot_h = int(max(320, min(900, round(max_h * 0.45))))
    bottom = _render_ip_strip(
        total_w_px=total_w,
        plot_h_px=plot_h,
        dpi=dpi,
        shot=shot,
        atime=atime,
        ip=ip,
        picks=picks,
        resample=_resample,
    )

    # --- 上下拼接 ---
    margin_top = 10
    final_h = margin_top + top_rgb.size[1] + vgap_px + bottom.size[1]
    final = Image.new("RGB", (total_w, final_h), (255, 255, 255))
    final.paste(top_rgb, (0, margin_top))
    final.paste(bottom, (0, margin_top + top_rgb.size[1] + vgap_px))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    final.save(out_png, dpi=(dpi, dpi), optimize=True)
    print(f"[montage] {out_png}", flush=True)


def _process_shot(
    *,
    shot: int,
    precursor_npz: Path,
    pcs_by_t_efit: Path,
    bc_by_t_efit: Path,
    out_dir: Path,
    converged_only: bool,
) -> int:
    if not precursor_npz.is_file():
        print(f"[skip] shot={shot} 无 precursor {precursor_npz}", file=sys.stderr, flush=True)
        return 2

    bundle = load_precursor_npz(precursor_npz)
    atime = np.asarray(bundle["ATIME"], dtype=np.float64).ravel()
    ipr = np.asarray(bundle["PCRL01"], dtype=np.float64).ravel()
    if ipr.size != atime.size:
        print(f"[skip] shot={shot} ATIME/PCRL01 长度不一致", file=sys.stderr, flush=True)
        return 2
    phase_ids = phase_ids_per_step(atime, ipr)

    pcs_rows = _load_frame_metrics(pcs_by_t_efit)
    bc_rows = _load_frame_metrics(bc_by_t_efit)
    if not pcs_rows or not bc_rows:
        print(f"[skip] shot={shot} pcs 或 bc_pred 无 summary", file=sys.stderr, flush=True)
        return 2

    picks: dict[str, tuple[int, dict, dict] | None] = {}
    manifest: dict[str, Any] = {"shot": shot, "converged_only": converged_only}
    for ph in PHASE_NAMES:
        result = _best_shared_k(pcs_rows, bc_rows, phase_ids, ph, converged_only=converged_only)
        picks[ph] = result
        if result is not None:
            k, pr, br = result
            manifest[ph] = {
                "k": k,
                "time_s": pr["time_s"],
                "pcs_lcfs_rmsep_m": pr["lcfs_rmsep_to_r8z8_m"],
                "bc_lcfs_rmsep_m": br["lcfs_rmsep_to_r8z8_m"],
                "sum_rmsep_m": pr["lcfs_rmsep_to_r8z8_m"] + br["lcfs_rmsep_to_r8z8_m"],
            }
        else:
            manifest[ph] = None

    out_png = out_dir / f"montage_pcs_vs_pred_shot{shot}.png"
    _render_figure(
        shot=shot,
        picks=picks,
        atime=atime,
        ip=ipr,
        out_png=out_png,
    )

    meta_path = out_dir / f"montage_pcs_vs_pred_shot{shot}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[manifest] {meta_path}", flush=True)
    return 0


def _read_shots_file(path: Path) -> list[int]:
    out: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip()
        if s.isdigit():
            out.append(int(s))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", type=int, default=None)
    ap.add_argument("--shots-file", type=str, default=None, help="每行一炮号")
    ap.add_argument(
        "--precursor-root",
        type=str,
        default=str(_REPO / "results" / "freegsnke_precursors"),
    )
    ap.add_argument(
        "--whole-shot-root",
        type=str,
        default=str(_REPO / "results" / "freegsnke_whole_shot"),
    )
    ap.add_argument("--by-t-efit-subdir", type=str, default="by_t_efit")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--converged-only", action="store_true")
    args = ap.parse_args()

    shots: list[int] = []
    if args.shots_file:
        p = Path(args.shots_file).resolve()
        if not p.is_file():
            raise SystemExit(f"--shots-file 不存在：{p}")
        shots.extend(_read_shots_file(p))
    if args.shot is not None:
        shots.append(int(args.shot))
    if not shots:
        raise SystemExit("请指定 --shot 或 --shots-file")

    prec_root = Path(args.precursor_root).resolve()
    whole_root = Path(args.whole_shot_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (whole_root / "_montage")

    rc_all = 0
    for shot in sorted(set(shots)):
        prec = prec_root / str(shot) / "precursor.npz"
        pcs_bt = whole_root / str(shot) / "pcs" / args.by_t_efit_subdir
        bc_bt = whole_root / str(shot) / "bc_pred" / args.by_t_efit_subdir
        r = _process_shot(
            shot=shot,
            precursor_npz=prec,
            pcs_by_t_efit=pcs_bt,
            bc_by_t_efit=bc_bt,
            out_dir=out_dir,
            converged_only=bool(args.converged_only),
        )
        if r != 0:
            rc_all = max(rc_all, r)
    return min(rc_all, 255)


if __name__ == "__main__":
    raise SystemExit(main())
