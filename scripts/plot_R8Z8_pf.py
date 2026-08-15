#!/usr/bin/env python3
"""EAST PF 模仿学习项目的输入/输出示意图（R-Z 截面）。

**输入（state）**：EFIT/PF_ATIME 数据库中的 R8、Z8（8 控制点）与磁轴 (lmsr, lmsz)。
**输出（action）**：14 路 PF 线圈（PCPF1–8、PCPF11–14 共 12 路指令对应其中 12 个线圈）。

运行::

    python3 scripts/plot_R8Z8_pf.py --shot 150268
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bc.common.constants import DATASET_ROOT, PCPF_NAMES
from bc.gs_forward.mds_efit_snapshot import EFIT_DB_DEFAULT

try:
    from scan_data.compat import _fetch_node_data
except ImportError:  # pragma: no cover
    _fetch_node_data = None

EAST_CONFIG = REPO_ROOT / "scan_data" / "unused" / "DataBase" / "EAST_config"
OUTPUT_DIR = REPO_ROOT / "scripts" / "figures"
DEFAULT_EFIT_SHOT = 150268
PF_NAMES = [f"PF{i}" for i in range(1, 15)]

CONTOUR_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
QUIVER_SCALE = 0.08
LIMITER_CLIP_MARGIN = 0.005  # [m] 偏滤器腿与第一壁的最小间隙（约 5 mm）

# 参考图配色（像素级提取）
PALETTE_INPUT = "#FF0000"           # R8/Z8 控制点散点（红色，参考图无此元素）
PALETTE_PF_COIL = "#00BFBF"         # PF 线圈（青色）
PF_COIL_COLOR = PALETTE_PF_COIL
PF_LABEL_COLOR = "#000000"          # PF 标签（黑色）
INPUT_R8Z8_COLOR = PALETTE_INPUT
R8Z8_TICK_COLOR = "#FF0000"         # R8/Z8 径向标记线（红色）
MAGNETIC_AXIS_COLOR = "#FF0000"     # 磁轴标记（红色）
COIL_EDGE_COLOR = "#7F7F7F"         # IC 线圈（灰色）
R8Z8_RADIAL_TICK_HALF_LEN = 0.075  # [m] 磁轴–控制点径向线上的短标记半长
LCFS_COLOR = "#000000"              # LCFS（黑色）
WALL_COLOR = "#FF69B4"              # 第一壁轮廓（粉红）
VESSEL_COLOR = "#008000"            # 真空室轮廓（绿色）

OUTBOARD_PF = frozenset({"PF11", "PF12", "PF13", "PF14"})

# compare_r8z8_control_points.py 仍使用
R8Z8_COLOR = "#4895EF"
R8Z8_DB_COLOR = "#FFD166"
PLASMA_CENTER_COLOR = MAGNETIC_AXIS_COLOR
R8Z8_SHAPE_CONTROL_LABEL = "Shape control (8-pt)"

LABEL_LCFS = "LCFS"
LABEL_R8Z8 = r"$(R_i, Z_i)$"
LABEL_MAGNETIC_AXIS = r"$(R_\mathrm{c}, Z_\mathrm{c})$"
LABEL_PF_COILS = "PF coils"

# 向后兼容 compare 脚本
LABEL_INPUT_R8Z8 = LABEL_R8Z8
LABEL_PF_OUTPUT = LABEL_PF_COILS


def _parse_rz_text(r_el: ET.Element | None, z_el: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    if r_el is None or z_el is None or r_el.text is None or z_el.text is None:
        raise ValueError("缺少 r/z 文本")
    rs = [float(x) for x in "".join(r_el.text.split()).split(",") if x.strip()]
    zs = [float(x) for x in "".join(z_el.text.split()).split(",") if x.strip()]
    if len(rs) != len(zs):
        raise ValueError(f"r/z 点数不一致: {len(rs)} vs {len(zs)}")
    return np.asarray(rs, dtype=float), np.asarray(zs, dtype=float)


def close_polyline(r: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if r.size == 0:
        return r, z
    if r[0] != r[-1] or z[0] != z[-1]:
        return np.r_[r, r[0]], np.r_[z, z[0]]
    return r, z


def _dist_point_to_polyline(px: float, py: float, xr: np.ndarray, yr: np.ndarray) -> float:
    dmin = np.inf
    for i in range(len(xr) - 1):
        x1, y1, x2, y2 = float(xr[i]), float(yr[i]), float(xr[i + 1]), float(yr[i + 1])
        dx, dy = x2 - x1, y2 - y1
        denom = dx * dx + dy * dy
        if denom == 0.0:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denom))
        d = math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))
        dmin = min(dmin, d)
    return float(dmin)


def _clip_separatrix_before_limiter(
    r: np.ndarray,
    z: np.ndarray,
    limiter_r: np.ndarray,
    limiter_z: np.ndarray,
    *,
    margin: float = LIMITER_CLIP_MARGIN,
) -> tuple[np.ndarray, np.ndarray]:
    """截断偏滤器腿两端，使其在触及第一壁前停止。"""
    r = np.asarray(r, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    n = r.size
    if n < 3:
        return r, z

    lim_path = MplPath(np.c_[limiter_r, limiter_z])
    inside = lim_path.contains_points(np.c_[r, z])
    dists = np.array(
        [_dist_point_to_polyline(float(r[i]), float(z[i]), limiter_r, limiter_z) for i in range(n)]
    )

    def ok(i: int) -> bool:
        return bool(inside[i] and dists[i] >= margin)

    i0 = 0
    while i0 < n - 1 and not ok(i0):
        i0 += 1
    i1 = n - 1
    while i1 > i0 and not ok(i1):
        i1 -= 1
    if i0 >= i1:
        return r, z
    return r[i0 : i1 + 1].copy(), z[i0 : i1 + 1].copy()


def load_wall(config_dir: Path) -> dict[str, np.ndarray]:
    root = ET.parse(config_dir / "wall.xml").getroot()
    lim = root.findall(".//outline")[0]
    limiter_r, limiter_z = _parse_rz_text(lim.find("r"), lim.find("z"))
    inn = root.findall(".//outline_inner")[0]
    vessel_inner_r, vessel_inner_z = _parse_rz_text(inn.find("r"), inn.find("z"))
    out = root.findall(".//outline_outer")[0]
    vessel_outer_r, vessel_outer_z = _parse_rz_text(out.find("r"), out.find("z"))
    return {
        "limiter_r": limiter_r,
        "limiter_z": limiter_z,
        "vessel_inner_r": vessel_inner_r,
        "vessel_inner_z": vessel_inner_z,
        "vessel_outer_r": vessel_outer_r,
        "vessel_outer_z": vessel_outer_z,
    }


def load_pf_coils(config_dir: Path) -> dict[str, tuple[float, float, float, float]]:
    root = ET.parse(config_dir / "pf_active.xml").getroot()
    want = {f"PF{i}" for i in range(1, 15)} | {"IC1", "IC2"}
    out: dict[str, tuple[float, float, float, float]] = {}
    for coil in root.findall(".//coil"):
        ne = coil.find("name")
        if ne is None or not (ne.text and ne.text.strip()):
            continue
        name = ne.text.strip()
        if name not in want:
            continue
        rect = coil.find(".//rectangle")
        if rect is None:
            continue
        r_el, z_el = rect.find("r"), rect.find("z")
        w_el, h_el = rect.find("width"), rect.find("height")
        if any(x is None or x.text is None for x in (r_el, z_el, w_el, h_el)):
            continue
        out[name] = (
            float(r_el.text),
            float(z_el.text),
            float(w_el.text),
            float(h_el.text),
        )
    missing = want - set(out)
    if missing:
        raise ValueError(f"pf_active.xml 缺少线圈: {sorted(missing)}")
    return out


def load_magnetics(config_dir: Path) -> tuple[list[dict], list[dict]]:
    root = ET.parse(config_dir / "magnetics.xml").getroot()
    probes: list[dict] = []
    for tag in ("b_field_pol_probe", "b_field_tor_probe"):
        for el in root.findall(tag):
            ne = el.find("name")
            pos = el.find("position")
            ang = el.find("poloidal_angle")
            if ne is None or pos is None or ang is None:
                continue
            r_el, z_el = pos.find("r"), pos.find("z")
            if r_el is None or z_el is None or r_el.text is None or z_el.text is None:
                continue
            probes.append(
                {
                    "name": ne.text.strip(),
                    "r": float(r_el.text),
                    "z": float(z_el.text),
                    "angle_deg": float(ang.text),
                }
            )
    flux_loops: list[dict] = []
    for el in root.findall("flux_loop"):
        ne = el.find("name")
        pos = el.find("position")
        if ne is None or pos is None:
            continue
        r_el, z_el = pos.find("r"), pos.find("z")
        if r_el is None or z_el is None or r_el.text is None or z_el.text is None:
            continue
        flux_loops.append(
            {
                "name": ne.text.strip(),
                "r": float(r_el.text),
                "z": float(z_el.text),
            }
        )
    return probes, flux_loops


def coil_box(r: float, z: float, w: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    half_w, half_h = 0.5 * w, 0.5 * h
    rr = np.array([r - half_w, r + half_w, r + half_w, r - half_w, r - half_w])
    zz = np.array([z - half_h, z - half_h, z + half_h, z + half_h, z - half_h])
    return rr, zz


def _draw_coil_box_with_x(
    ax,
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    color: str = COIL_EDGE_COLOR,
    linewidth: float = 1.0,
    label: str | None = None,
    zorder: int = 2,
) -> None:
    """线圈矩形框 + 中心对角 X（无填充）。"""
    rr, zz = coil_box(cx, cy, w, h)
    ax.plot(rr, zz, color=color, linewidth=linewidth, label=label, zorder=zorder)
    hw, hh = 0.5 * w, 0.5 * h
    ax.plot(
        [cx - hw, cx + hw],
        [cy - hh, cy + hh],
        color=color,
        linewidth=linewidth * 0.55,
        zorder=zorder,
    )
    ax.plot(
        [cx - hw, cx + hw],
        [cy + hh, cy - hh],
        color=color,
        linewidth=linewidth * 0.55,
        zorder=zorder,
    )


def _configure_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )


def _r8z8_radial_tick(
    pr: float,
    pz: float,
    rcx: float,
    rcz: float,
    *,
    half_len: float = R8Z8_RADIAL_TICK_HALF_LEN,
) -> tuple[list[float], list[float]]:
    """磁轴–控制点径向线上、以控制点为中心的短标记（各段互不连接）。"""
    dr, dz = pr - rcx, pz - rcz
    norm = math.hypot(dr, dz)
    if norm < 1e-9:
        return [pr, pr], [pz, pz]
    ux, uy = dr / norm, dz / norm
    return [pr - half_len * ux, pr + half_len * ux], [pz - half_len * uy, pz + half_len * uy]


def coil_label_position(
    r: float,
    z: float,
    w: float,
    h: float,
    *,
    name: str | None = None,
) -> tuple[float, float, str]:
    """线圈标签位置；外侧 PF11–14 标签放在线圈左侧，避免出界。"""
    if name in OUTBOARD_PF:
        return r - 0.5 * w - 0.04, z, "right"
    return r + 0.5 * w + 0.03, z, "left"


def synthetic_flux_contour(
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    *,
    r0: float = 1.85,
    z0: float = 0.0,
    a: float = 0.42,
    b: float = 0.55,
    kappa: float = 1.65,
) -> np.ndarray:
    """示意性归一化极向磁通，用于绘制等值线背景。"""
    rr, zz = np.meshgrid(r_grid, z_grid)
    r_minor = np.sqrt((rr - r0) ** 2 + (zz - z0) ** 2)
    theta = np.arctan2(zz - z0, rr - r0)
    r_surf = a * (1.0 + 0.08 * np.cos(theta))
    z_surf = b * np.sin(theta) * kappa
    psi = (rr - r0) ** 2 / r_surf**2 + (zz - z0) ** 2 / z_surf**2
    return psi


def _pick_efit_frame_index(
    atime: np.ndarray,
    time_target: float | None,
    f: h5py.File | None = None,
) -> int:
    atime = np.asarray(atime, dtype=np.float64).ravel()
    if time_target is not None:
        return int(np.argmin(np.abs(atime - float(time_target))))
    if f is not None:
        for key in ("WMHD", "AREA", "EFIT_MFILE:CPASMA"):
            if key not in f:
                continue
            vals = np.asarray(f[key][:], dtype=np.float64).ravel()
            if vals.size != atime.size:
                continue
            valid = atime >= 0.0
            k = int(np.argmax(np.where(valid, vals, -np.inf)))
            if np.isfinite(vals[k]) and vals[k] > 0:
                return k
    return len(atime) // 2


def _lcfs_rz_from_bdry_slice(bdry: np.ndarray, nb: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(nb)
    if n < 3:
        raise ValueError(f"EFIT LCFS 点数过少: NBDRY={n}")
    pts = np.asarray(bdry[:n], dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"EFIT BDRY 形状异常: {pts.shape}")
    return pts[:, 0].copy(), pts[:, 1].copy()


def _contour_polylines_at_level(
    psirz: np.ndarray,
    rgrid: np.ndarray,
    zgrid: np.ndarray,
    level: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """在 PSIRZ 网格上提取 psi=level 的等值线段。"""
    rr, zz = np.meshgrid(rgrid, zgrid)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contour(rr, zz, psirz, levels=[level])
    plt.close(fig)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for seg in cs.allsegs[0]:
        v = np.asarray(seg, dtype=np.float64)
        if v.shape[0] >= 3:
            out.append((v[:, 0].copy(), v[:, 1].copy()))
    return out


def _select_main_separatrix_polylines(
    polylines: list[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """保留主分离面段（含偏滤器腿），丢弃网格边角伪影。"""
    candidates = [(r, z) for r, z in polylines if len(r) >= 50]
    if not candidates:
        candidates = polylines
    if not candidates:
        raise ValueError("PSIRZ@SSIBRY 等值线为空")
    divertor = [(r, z) for r, z in candidates if float(np.min(z)) <= -0.5]
    if divertor:
        return [max(divertor, key=lambda p: len(p[0]))]
    return [max(candidates, key=lambda p: len(p[0]))]


def _lcfs_polylines_from_psirz(
    ssibry: float,
    psirz: np.ndarray,
    rgrid: np.ndarray,
    zgrid: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    polylines = _contour_polylines_at_level(psirz, rgrid, zgrid, ssibry)
    return _select_main_separatrix_polylines(polylines)


def _efit_meta(
    shot: int,
    atime: np.ndarray,
    k: int,
    time_target: float | None,
    *,
    source: str,
    npts: int,
) -> dict[str, Any]:
    return {
        "shot": int(shot),
        "t_target": float(time_target if time_target is not None else atime[k]),
        "t_efit": float(atime[k]),
        "k_efit": k,
        "npts": npts,
        "source": source,
    }


def load_efit_lcfs_polylines(
    shot: int = DEFAULT_EFIT_SHOT,
    time_target: float | None = None,
    *,
    efit_dir: Path = EFIT_DB_DEFAULT,
    efit_source: str = "auto",
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """从 EFIT 读取分离面折线：优先 PSIRZ@SSIBRY（含偏滤器腿），否则回退 BDRY。"""
    fp = Path(efit_dir) / f"{shot}.h5"
    if efit_source in ("auto", "h5") and fp.is_file():
        with h5py.File(fp) as f:
            atime = np.asarray(f["ATIME"][:], dtype=np.float64)
            k = _pick_efit_frame_index(atime, time_target, f)
            if all(key in f for key in ("PSIRZ", "SSIBRY", "R", "Z")):
                ssibry = float(f["SSIBRY"][k])
                psirz = np.asarray(f["PSIRZ"][k], dtype=np.float64)
                rgrid = np.asarray(f["R"][:], dtype=np.float64).ravel()
                zgrid = np.asarray(f["Z"][:], dtype=np.float64).ravel()
                polylines = _lcfs_polylines_from_psirz(ssibry, psirz, rgrid, zgrid)
                npts = sum(len(r) for r, _ in polylines)
                return polylines, _efit_meta(
                    shot, atime, k, time_target, source="h5:PSIRZ@SSIBRY", npts=npts
                )
            if "BDRY" in f and "NBDRY" in f:
                nb = int(f["NBDRY"][k])
                r, z = _lcfs_rz_from_bdry_slice(f["BDRY"][k], nb)
                return [(r, z)], _efit_meta(
                    shot, atime, k, time_target, source="h5:BDRY", npts=nb
                )
            raise KeyError(f"{fp} 缺少 PSIRZ/SSIBRY 或 BDRY/NBDRY")

    if efit_source == "h5":
        raise FileNotFoundError(f"缺少 EFIT h5: {fp}")

    if _fetch_node_data is None:
        raise FileNotFoundError(f"无本地 EFIT h5 且无法 import scan_data.compat")

    psirz_raw, t_axis = _fetch_node_data(int(shot), "efit_east", "PSIRZ")
    ssibry_raw, _ = _fetch_node_data(int(shot), "efit_east", "SSIBRY")
    r_raw, _ = _fetch_node_data(int(shot), "efit_east", "R")
    z_raw, _ = _fetch_node_data(int(shot), "efit_east", "Z")
    if t_axis is not None and psirz_raw is not None and ssibry_raw is not None:
        atime = np.asarray(t_axis, dtype=np.float64).ravel()
        psirz = np.asarray(psirz_raw, dtype=np.float64)
        ssibry = np.asarray(ssibry_raw, dtype=np.float64).ravel()
        rgrid = np.asarray(r_raw, dtype=np.float64).ravel()
        zgrid = np.asarray(z_raw, dtype=np.float64).ravel()
        if psirz.ndim == 3 and psirz.shape[0] == atime.size:
            k = _pick_efit_frame_index(atime, time_target)
            polylines = _lcfs_polylines_from_psirz(float(ssibry[k]), psirz[k], rgrid, zgrid)
            npts = sum(len(r) for r, _ in polylines)
            return polylines, _efit_meta(
                shot, atime, k, time_target, source="mds:PSIRZ@SSIBRY", npts=npts
            )

    bdry_raw, t_axis = _fetch_node_data(int(shot), "efit_east", "BDRY")
    nbdy_raw, _ = _fetch_node_data(int(shot), "efit_east", "NBDRY")
    bdry = np.asarray(bdry_raw, dtype=np.float64)
    nbdy = np.asarray(nbdy_raw, dtype=np.float64).ravel()
    if t_axis is None:
        raise ValueError("MDS BDRY 缺少时间轴")
    atime = np.asarray(t_axis, dtype=np.float64).ravel()
    if bdry.ndim != 3 or bdry.shape[0] != atime.size:
        raise ValueError(f"MDS BDRY 形状异常: {bdry.shape}, T={atime.size}")
    if nbdy.size != atime.size:
        raise ValueError(f"MDS NBDRY 长度异常: {nbdy.size}, T={atime.size}")

    k = _pick_efit_frame_index(atime, time_target)
    nb = int(nbdy[k])
    r, z = _lcfs_rz_from_bdry_slice(bdry[k], nb)
    return [(r, z)], _efit_meta(shot, atime, k, time_target, source="mds:BDRY", npts=nb)


def _plot_efit_lcfs(
    ax,
    polylines: list[tuple[np.ndarray, np.ndarray]],
    efit_meta: dict[str, Any],
    wall: dict[str, np.ndarray] | None = None,
    *,
    label: str = LABEL_LCFS,
) -> None:
    """绘制 EFIT 分离面；PSIRZ@SSIBRY 不强制闭合，BDRY 回退时闭合。"""
    src = str(efit_meta["source"])
    for i, (r, z) in enumerate(polylines):
        if "BDRY" in src:
            r, z = close_polyline(r, z)
        elif wall is not None:
            r, z = _clip_separatrix_before_limiter(
                r, z, wall["limiter_r"], wall["limiter_z"]
            )
        ax.plot(
            r,
            z,
            color=LCFS_COLOR,
            linewidth=2.5,
            label=label if i == 0 else None,
            zorder=4,
        )


def _boundary_arc_indices(i0: int, i1: int, n: int, forward: bool) -> list[int]:
    idx: list[int] = []
    i = i0
    for _ in range(n + 1):
        idx.append(i)
        if i == i1:
            break
        i = (i + 1) % n if forward else (i - 1) % n
    return idx


def _pick_boundary_arc(
    r: np.ndarray,
    z: np.ndarray,
    rc: float,
    zc: float,
    i0: int,
    i1: int,
    target_angle: float,
) -> list[int]:
    """在两条弧段中选与目标极角更一致的一条。"""
    n = int(r.size)
    fwd = _boundary_arc_indices(i0, i1, n, True)
    bwd = _boundary_arc_indices(i0, i1, n, False)
    theta = np.arctan2(z - zc, r - rc)

    def arc_mean_angle(idx: list[int]) -> float:
        return float(np.mean(theta[idx]))

    d_fwd = abs(np.angle(np.exp(1j * (arc_mean_angle(fwd) - target_angle))))
    d_bwd = abs(np.angle(np.exp(1j * (arc_mean_angle(bwd) - target_angle))))
    return fwd if d_fwd <= d_bwd else bwd


def _ray_segment_hit(
    rc: float,
    zc: float,
    angle: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[float, float, float] | None:
    dx, dy = math.cos(angle), math.sin(angle)
    sx, sy = x2 - x1, y2 - y1
    denom = dx * sy - dy * sx
    if abs(denom) < 1e-14:
        return None
    t = ((x1 - rc) * sy - (y1 - zc) * sx) / denom
    u = ((x1 - rc) * dy - (y1 - zc) * dx) / denom
    if t >= 0.0 and 0.0 <= u <= 1.0:
        return float(t), float(rc + t * dx), float(zc + t * dy)
    return None


def _squareness_point_on_arc(
    r: np.ndarray,
    z: np.ndarray,
    rc: float,
    zc: float,
    i0: int,
    i1: int,
    target_angle: float,
) -> tuple[float, float]:
    """在象限弧段上沿 target_angle 方向求方度点（约 45/135/225/315°）。"""
    arc = _pick_boundary_arc(r, z, rc, zc, i0, i1, target_angle)
    best: tuple[float, float, float] | None = None
    for j in range(len(arc) - 1):
        a, b = arc[j], arc[j + 1]
        hit = _ray_segment_hit(rc, zc, target_angle, float(r[a]), float(z[a]), float(r[b]), float(z[b]))
        if hit is not None and (best is None or hit[0] < best[0]):
            best = hit
    if best is not None:
        return best[1], best[2]

    theta = np.arctan2(z - zc, r - rc)
    deltas = [abs(np.angle(np.exp(1j * (float(theta[i]) - target_angle)))) for i in arc]
    k = arc[int(np.argmin(deltas))]
    return float(r[k]), float(z[k])


def compute_r8z8_from_boundary(
    r: np.ndarray,
    z: np.ndarray,
    rc: float,
    zc: float,
) -> tuple[np.ndarray, np.ndarray]:
    """8-pt shape control from a closed boundary: extrema + squareness @ 45°."""
    r = np.asarray(r, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    if r.size < 8 or r.size != z.size:
        raise ValueError(f"边界点数不足: {r.size}")

    io = int(np.argmax(r))
    iu = int(np.argmax(z))
    ii = int(np.argmin(r))
    il = int(np.argmin(z))

    r8 = np.zeros(8, dtype=np.float64)
    z8 = np.zeros(8, dtype=np.float64)
    r8[0], z8[0] = r[io], z[io]
    r8[2], z8[2] = r[iu], z[iu]
    r8[4], z8[4] = r[ii], z[ii]
    r8[6], z8[6] = r[il], z[il]

    for idx, i0, i1, ang in (
        (1, io, iu, math.pi / 4),
        (3, iu, ii, 3 * math.pi / 4),
        (5, ii, il, 5 * math.pi / 4),
        (7, il, io, 7 * math.pi / 4),
    ):
        sr, sz = _squareness_point_on_arc(r, z, rc, zc, i0, i1, ang)
        r8[idx], z8[idx] = sr, sz
    return r8, z8


def _load_efit_bdry_rz(
    shot: int,
    k: int,
    *,
    efit_dir: Path = EFIT_DB_DEFAULT,
    efit_source: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    fp = Path(efit_dir) / f"{int(shot)}.h5"
    if efit_source in ("auto", "h5") and fp.is_file():
        with h5py.File(fp) as f:
            if "BDRY" in f and "NBDRY" in f:
                nb = int(f["NBDRY"][k])
                return _lcfs_rz_from_bdry_slice(f["BDRY"][k], nb)

    if _fetch_node_data is None:
        raise KeyError(f"无法读取炮 {shot} 的 BDRY")

    bdry_raw, t_axis = _fetch_node_data(int(shot), "efit_east", "BDRY")
    nbdy_raw, _ = _fetch_node_data(int(shot), "efit_east", "NBDRY")
    if bdry_raw is None or nbdy_raw is None:
        raise KeyError(f"MDS 缺少炮 {shot} 的 BDRY/NBDRY")
    bdry = np.asarray(bdry_raw, dtype=np.float64)
    nbdy = np.asarray(nbdy_raw, dtype=np.float64).ravel()
    return _lcfs_rz_from_bdry_slice(bdry[k], int(nbdy[k]))


def _load_efit_db_r8z8(
    shot: int,
    k: int,
    *,
    efit_dir: Path = EFIT_DB_DEFAULT,
    efit_source: str = "auto",
) -> tuple[np.ndarray, np.ndarray] | None:
    """从 EFIT 数据库读取存储的 R8/Z8；不可用则返回 None。"""
    fp = Path(efit_dir) / f"{int(shot)}.h5"
    if efit_source in ("auto", "h5") and fp.is_file():
        with h5py.File(fp) as f:
            if "R8" in f and "Z8" in f:
                r8 = np.asarray(f["R8"][k], dtype=np.float64).ravel()
                z8 = np.asarray(f["Z8"][k], dtype=np.float64).ravel()
                if r8.size == 8 and z8.size == 8:
                    return r8, z8

    if _fetch_node_data is None:
        return None

    r8_raw, _ = _fetch_node_data(int(shot), "efit_east", "R8")
    z8_raw, _ = _fetch_node_data(int(shot), "efit_east", "Z8")
    if r8_raw is None or z8_raw is None:
        return None
    r8_arr = np.asarray(r8_raw, dtype=np.float64)
    z8_arr = np.asarray(z8_raw, dtype=np.float64)
    if r8_arr.ndim == 2 and r8_arr.shape[1] == 8:
        return r8_arr[k].ravel(), z8_arr[k].ravel()
    if r8_arr.ndim == 2 and r8_arr.shape[0] == 8:
        return r8_arr[:, k].ravel(), z8_arr[:, k].ravel()
    return None


def _load_lmsr_lmsz_at_time(
    shot: int,
    t_efit: float,
    *,
    dataset_root: Path = DATASET_ROOT,
) -> tuple[float, float]:
    """从 PF_ATIME 数据集读取与 EFIT 时刻对齐的 lmsr/lmsz。"""
    pf_fp = Path(dataset_root) / f"{int(shot)}.h5"
    if pf_fp.is_file():
        with h5py.File(pf_fp) as f:
            if "time" not in f or "lmsr" not in f or "lmsz" not in f:
                raise KeyError(f"{pf_fp} 缺少 time/lmsr/lmsz")
            time = np.asarray(f["time"][:], dtype=np.float64)
            k = int(np.argmin(np.abs(time - float(t_efit))))
            lmsr = float(f["lmsr"][k])
            lmsz = float(f["lmsz"][k])
            if "mask_lmsr" in f and not bool(f["mask_lmsr"][k]):
                lmsr = float("nan")
            if "mask_lmsz" in f and not bool(f["mask_lmsz"][k]):
                lmsz = float("nan")
            return lmsr, lmsz

    if _fetch_node_data is None:
        return float("nan"), float("nan")

    atime_raw, _ = _fetch_node_data(int(shot), "efit_east", "ATIME")
    atime = np.asarray(atime_raw, dtype=np.float64).ravel()
    k = int(np.argmin(np.abs(atime - float(t_efit))))
    t_axis = float(atime[k])
    out: dict[str, float] = {}
    for node in ("lmsr", "lmsz"):
        y_raw, t_raw = _fetch_node_data(int(shot), "pcs_east", node)
        if y_raw is None or t_raw is None:
            out[node] = float("nan")
            continue
        y = np.asarray(y_raw, dtype=np.float64).ravel()
        t = np.asarray(t_raw, dtype=np.float64).ravel()
        if y.size != t.size or y.size == 0:
            out[node] = float("nan")
            continue
        out[node] = float(np.interp(t_axis, t, y))
    return out["lmsr"], out["lmsz"]


def _load_r8z8_at_time(
    shot: int,
    t_efit: float,
    k_efit: int,
    *,
    efit_dir: Path = EFIT_DB_DEFAULT,
    efit_source: str = "auto",
    dataset_root: Path = DATASET_ROOT,
) -> tuple[np.ndarray, np.ndarray, str]:
    """从 PF_ATIME 或 EFIT 数据库读取 R8/Z8（不做边界重算）。"""
    pf_fp = Path(dataset_root) / f"{int(shot)}.h5"
    if pf_fp.is_file():
        with h5py.File(pf_fp) as f:
            if all(key in f for key in ("time", "R8", "Z8")):
                time = np.asarray(f["time"][:], dtype=np.float64)
                k = int(np.argmin(np.abs(time - float(t_efit))))
                r8 = np.asarray(f["R8"][k], dtype=np.float64).ravel()
                z8 = np.asarray(f["Z8"][k], dtype=np.float64).ravel()
                if r8.size == 8 and z8.size == 8:
                    return r8, z8, "PF_ATIME"

    db_r8z8 = _load_efit_db_r8z8(shot, k_efit, efit_dir=efit_dir, efit_source=efit_source)
    if db_r8z8 is not None:
        return db_r8z8[0], db_r8z8[1], "EFIT"
    raise KeyError(f"炮 {shot} 无法从 PF_ATIME 或 EFIT 读取 R8/Z8")


def load_io_state(
    shot: int,
    efit_meta: dict[str, Any],
    *,
    efit_dir: Path = EFIT_DB_DEFAULT,
    efit_source: str = "auto",
    dataset_root: Path = DATASET_ROOT,
) -> dict[str, Any]:
    """读取模仿学习输入：数据库 R8/Z8 + (lmsr, lmsz)。"""
    k = int(efit_meta["k_efit"])
    t_efit = float(efit_meta["t_efit"])
    r8, z8, r8_source = _load_r8z8_at_time(
        shot,
        t_efit,
        k,
        efit_dir=efit_dir,
        efit_source=efit_source,
        dataset_root=dataset_root,
    )
    lmsr, lmsz = _load_lmsr_lmsz_at_time(shot, t_efit, dataset_root=dataset_root)
    return {
        "r8": r8,
        "z8": z8,
        "lmsr": lmsr,
        "lmsz": lmsz,
        "t_efit": t_efit,
        "r8_source": r8_source,
        "shot": int(shot),
    }


def _load_pcpf_at_time(
    shot: int,
    t_efit: float,
    *,
    dataset_root: Path = DATASET_ROOT,
) -> dict[str, float]:
    pf_fp = Path(dataset_root) / f"{int(shot)}.h5"
    if not pf_fp.is_file():
        return {}
    out: dict[str, float] = {}
    with h5py.File(pf_fp) as f:
        if "time" not in f:
            return {}
        time = np.asarray(f["time"][:], dtype=np.float64)
        k = int(np.argmin(np.abs(time - float(t_efit))))
        for name in PCPF_NAMES:
            if name not in f:
                continue
            val = float(f[name][k])
            mask_key = f"mask_{name}"
            if mask_key in f and not bool(f[mask_key][k]):
                val = float("nan")
            out[name] = val
    return out


def load_boundary_overlay(
    shot: int,
    efit_meta: dict[str, Any],
    *,
    efit_dir: Path = EFIT_DB_DEFAULT,
    efit_source: str = "auto",
    dataset_root: Path = DATASET_ROOT,
) -> dict[str, Any]:
    """从 BDRY 重算 R8/Z8，并读取 PCS 等离子体中心 lmsr/lmsz。"""
    k = int(efit_meta["k_efit"])
    t_efit = float(efit_meta["t_efit"])
    bdry_r, bdry_z = _load_efit_bdry_rz(
        shot, k, efit_dir=efit_dir, efit_source=efit_source
    )
    lmsr, lmsz = _load_lmsr_lmsz_at_time(shot, t_efit, dataset_root=dataset_root)
    if np.isfinite(lmsr) and np.isfinite(lmsz):
        rc, zc = float(lmsr), float(lmsz)
    else:
        rc = float(0.5 * (bdry_r.max() + bdry_r.min()))
        zc = float(0.5 * (bdry_z.max() + bdry_z.min()))
    r8, z8 = compute_r8z8_from_boundary(bdry_r, bdry_z, rc, zc)
    db_r8z8 = _load_efit_db_r8z8(shot, k, efit_dir=efit_dir, efit_source=efit_source)
    out: dict[str, Any] = {
        "r8": r8,
        "z8": z8,
        "lmsr": lmsr,
        "lmsz": lmsz,
        "bdry_r": bdry_r,
        "bdry_z": bdry_z,
        "center_r": rc,
        "center_z": zc,
        "t_efit": t_efit,
    }
    if db_r8z8 is not None:
        out["r8_db"], out["z8_db"] = db_r8z8
    return out


def probe_quiver_components(angle_deg: float, length: float = QUIVER_SCALE) -> tuple[float, float]:
    rad = math.radians(angle_deg)
    return length * math.cos(rad), length * math.sin(rad)


def _draw_wall(ax, wall: dict[str, np.ndarray]) -> None:
    for key in ("vessel_inner", "vessel_outer"):
        rk, zk = f"{key}_r", f"{key}_z"
        rr, zz = close_polyline(wall[rk], wall[zk])
        ax.plot(rr, zz, color=VESSEL_COLOR, linewidth=2.0, zorder=1)
    rr, zz = close_polyline(wall["limiter_r"], wall["limiter_z"])
    ax.plot(rr, zz, color=WALL_COLOR, linewidth=1.4, zorder=1)


def _draw_pf_outputs(
    ax,
    pf: dict[str, tuple[float, float, float, float]],
) -> None:
    """14 路 PF 线圈：空心框 + 中心 X。"""
    for i, name in enumerate(PF_NAMES):
        cx, cy, w, h = pf[name]
        _draw_coil_box_with_x(
            ax,
            cx,
            cy,
            w,
            h,
            color=PF_COIL_COLOR,
            linewidth=1.8,
            label=LABEL_PF_COILS if i == 0 else None,
            zorder=2,
        )
        lx, ly, ha = coil_label_position(cx, cy, w, h, name=name)
        ax.text(lx, ly, name, fontsize=8.5, va="center", ha=ha, color=PF_LABEL_COLOR, zorder=3)


def _draw_ic_coils(
    ax,
    pf: dict[str, tuple[float, float, float, float]],
) -> None:
    for name in ("IC1", "IC2"):
        cx, cy, w, h = pf[name]
        _draw_coil_box_with_x(ax, cx, cy, w, h, color=PF_COIL_COLOR, linewidth=1.2, zorder=2)
        lx, ly, ha = coil_label_position(cx, cy, w, h, name=name)
        ax.text(lx, ly, name, fontsize=8, va="center", ha=ha, color="#000000", zorder=3)


def _draw_coils_and_wall(
    ax,
    wall: dict[str, np.ndarray],
    pf: dict[str, tuple[float, float, float, float]],
) -> None:
    _draw_wall(ax, wall)
    _draw_ic_coils(ax, pf)
    _draw_pf_outputs(ax, pf)


def _plot_io_inputs(ax, state: dict[str, Any]) -> None:
    """R8/Z8 控制点 + 径向短标记；磁轴 (lmsr, lmsz)。"""
    r8 = np.asarray(state["r8"], dtype=np.float64).ravel()
    z8 = np.asarray(state["z8"], dtype=np.float64).ravel()
    lmsr = float(state["lmsr"])
    lmsz = float(state["lmsz"])
    if np.isfinite(lmsr) and np.isfinite(lmsz):
        rcx, rcz = lmsr, lmsz
    else:
        rcx = float(np.mean(r8))
        rcz = float(np.mean(z8))

    for i, (pr, pz) in enumerate(zip(r8, z8)):
        seg_r, seg_z = _r8z8_radial_tick(pr, pz, rcx, rcz)
        ax.plot(
            seg_r,
            seg_z,
            color=R8Z8_TICK_COLOR,
            linewidth=2.5,
            solid_capstyle="round",
            label=LABEL_R8Z8 if i == 0 else None,
            zorder=10,
        )

    ax.scatter(
        r8,
        z8,
        s=44,
        facecolors=INPUT_R8Z8_COLOR,
        edgecolors="white",
        linewidths=1.0,
        zorder=11,
    )

    if np.isfinite(lmsr) and np.isfinite(lmsz):
        ax.plot(
            lmsr,
            lmsz,
            linestyle="none",
            marker="+",
            color=MAGNETIC_AXIS_COLOR,
            markersize=11,
            markeredgewidth=2.4,
            label=LABEL_MAGNETIC_AXIS,
            zorder=9,
        )


def _style_axes(ax) -> None:
    ax.set_aspect("equal")
    ax.set_xlabel(r"$R$ (m)", fontsize=11)
    ax.set_ylabel(r"$Z$ (m)", fontsize=11)
    ax.tick_params(labelsize=9, width=0.8, length=3.5)
    ax.grid(False)


def _save_figure(fig, out_path: Path | None, default_name: str, *, show: bool) -> Path:
    if out_path is None:
        out_path = OUTPUT_DIR / default_name
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"saved: {out_path}")
    print(f"saved: {pdf_path}")
    return out_path


def create_pf_io_figure(
    config_dir: Path = EAST_CONFIG,
    out_path: Path | None = None,
    *,
    shot: int = DEFAULT_EFIT_SHOT,
    time_target: float | None = None,
    efit_dir: Path = EFIT_DB_DEFAULT,
    efit_source: str = "auto",
    dataset_root: Path = DATASET_ROOT,
    show: bool = False,
) -> Path:
    """绘制项目输入 (R8/Z8, lmsr/lmsz) 与输出 (14×PF) 示意图。"""
    _configure_figure_style()
    if not (config_dir / "wall.xml").is_file():
        raise FileNotFoundError(f"未找到 EAST 配置目录: {config_dir}")

    wall = load_wall(config_dir)
    pf = load_pf_coils(config_dir)
    lcfs_polylines, efit_meta = load_efit_lcfs_polylines(
        shot, time_target, efit_dir=efit_dir, efit_source=efit_source
    )
    io_state = load_io_state(
        shot,
        efit_meta,
        efit_dir=efit_dir,
        efit_source=efit_source,
        dataset_root=dataset_root,
    )

    fig, ax = plt.subplots(figsize=(6.8, 8.6), facecolor="white")
    ax.set_facecolor("white")
    _style_axes(ax)
    _draw_coils_and_wall(ax, wall, pf)
    _plot_efit_lcfs(ax, lcfs_polylines, efit_meta, wall=wall)
    _plot_io_inputs(ax, io_state)

    t_efit = float(io_state["t_efit"])
    ax.set_title(f"Shot #{shot},  $t = {t_efit:.2f}$ s", fontsize=11, pad=6)
    ax.legend(
        loc="upper right",
        fontsize=8.5,
        frameon=True,
        framealpha=0.97,
        edgecolor="0.75",
        facecolor="white",
        borderpad=0.4,
        labelspacing=0.35,
        handlelength=1.8,
    )
    fig.subplots_adjust(left=0.11, right=0.96, top=0.98, bottom=0.07)
    ax.margins(x=0.015, y=0.015)

    print(
        f"IO figure: shot={shot} t={io_state['t_efit']:.3f}s "
        f"R8/Z8={io_state['r8_source']} lmsr={io_state['lmsr']:.4f} "
        f"lmsz={io_state['lmsz']:.4f}\n"
        f"  palette: input={INPUT_R8Z8_COLOR}  PF={PF_COIL_COLOR}  PF-label={PF_LABEL_COLOR}"
    )
    return _save_figure(fig, out_path, "R8Z8_PF.png", show=show)


create_machine_figure = create_pf_io_figure


def create_diagnostics_figure(
    config_dir: Path = EAST_CONFIG,
    out_path: Path | None = None,
    *,
    shot: int = DEFAULT_EFIT_SHOT,
    time_target: float | None = None,
    efit_dir: Path = EFIT_DB_DEFAULT,
    efit_source: str = "auto",
    show: bool = False,
) -> Path:
    """完整诊断布局图（含磁探针、磁通环、磁通等值线）。"""
    if not (config_dir / "wall.xml").is_file():
        raise FileNotFoundError(f"未找到 EAST 配置目录: {config_dir}")

    wall = load_wall(config_dir)
    pf = load_pf_coils(config_dir)
    probes, flux_loops = load_magnetics(config_dir)
    lcfs_polylines, efit_meta = load_efit_lcfs_polylines(
        shot, time_target, efit_dir=efit_dir, efit_source=efit_source
    )

    fig, ax = plt.subplots(figsize=(5.75, 10.31), facecolor="white")
    ax.set_facecolor("white")
    _style_axes(ax)

    r_grid = np.linspace(0.55, 3.45, 280)
    z_grid = np.linspace(-2.35, 2.35, 320)
    psi = synthetic_flux_contour(r_grid, z_grid)
    ax.contour(r_grid, z_grid, psi, levels=CONTOUR_LEVELS, colors="k", linewidths=1.2)

    _draw_coils_and_wall(ax, wall, pf)

    seen: set[tuple[float, float]] = set()
    qr, qz, qu, qv = [], [], [], []
    for p in probes:
        key = (round(p["r"], 4), round(p["z"], 4))
        if key in seen:
            continue
        seen.add(key)
        u, v = probe_quiver_components(p["angle_deg"])
        qr.append(p["r"])
        qz.append(p["z"])
        qu.append(u)
        qv.append(v)
    ax.quiver(qr, qz, qu, qv, angles="xy", scale_units="xy", scale=1.0, color="b", width=0.0025)

    fl_upper_r = [f["r"] for f in flux_loops if f["z"] >= 0.0]
    fl_upper_z = [f["z"] for f in flux_loops if f["z"] >= 0.0]
    fl_lower_r = [f["r"] for f in flux_loops if f["z"] < 0.0]
    fl_lower_z = [f["z"] for f in flux_loops if f["z"] < 0.0]
    ax.plot(fl_upper_r, fl_upper_z, linestyle="none", marker="s", color="b", markersize=4)
    ax.plot(fl_lower_r, fl_lower_z, linestyle="none", marker="o", color="r", markersize=4)

    _plot_efit_lcfs(ax, lcfs_polylines, efit_meta, wall=wall)

    lr, lz = close_polyline(wall["limiter_r"], wall["limiter_z"])
    ax.plot(lr, lz, color="k", linewidth=1.0)

    ax.annotate("", xy=(0.53, 0.70), xytext=(0.57, 0.80), xycoords="figure fraction",
                arrowprops=dict(arrowstyle="->", lw=2, color="k"))
    ax.text(0.58, 0.83, "Magnetic\nProbe", transform=fig.transFigure, fontsize=16,
            fontweight="bold", ha="center", va="center")
    ax.annotate("", xy=(0.57, 0.33), xytext=(0.58, 0.25), xycoords="figure fraction",
                arrowprops=dict(arrowstyle="->", lw=2, color="k"))
    ax.text(0.59, 0.23, "Flux Loop", transform=fig.transFigure, fontsize=16,
            fontweight="bold", ha="center", va="center")
    ax.text(0.20, 0.95, "(a)", transform=fig.transFigure, fontsize=18, fontweight="bold",
            ha="left", va="top", bbox=dict(facecolor="white", edgecolor="white", pad=2))

    fig.tight_layout()
    return _save_figure(fig, out_path, "createfigure_east_diagnostics.png", show=show)


def main() -> None:
    parser = argparse.ArgumentParser(description="EAST PF 控制输入/输出示意图")
    parser.add_argument(
        "--mode",
        choices=("io", "diagnostics"),
        default="io",
        help="io=输入输出图；diagnostics=完整诊断图",
    )
    parser.add_argument("--shot", type=int, default=DEFAULT_EFIT_SHOT, help="EFIT 炮号")
    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="EFIT 目标时刻 [s]；省略则取 WMHD/AREA 最大时刻",
    )
    parser.add_argument(
        "--efit-dir",
        type=Path,
        default=EFIT_DB_DEFAULT,
        help="EFIT h5 数据库目录",
    )
    parser.add_argument(
        "--efit-source",
        choices=("auto", "h5", "mds"),
        default="auto",
        help="EFIT 读取来源",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="PF_ATIME 数据集目录",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="输出文件路径 (默认 scripts/figures/R8Z8_PF.png)",
    )
    args = parser.parse_args()
    kwargs = dict(
        shot=args.shot,
        time_target=args.time,
        efit_dir=args.efit_dir,
        efit_source=args.efit_source,
        dataset_root=args.dataset_root,
    )
    if args.mode == "io":
        create_pf_io_figure(out_path=args.output, show=False, **kwargs)
    else:
        create_diagnostics_figure(out_path=args.output, show=False, **kwargs)


if __name__ == "__main__":
    main()
