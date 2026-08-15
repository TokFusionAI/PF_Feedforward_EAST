"""EAST pf_active.xml + wall.xml → freegsnke ``tokamak()`` 机器描述.

不 import ``freegs``；仅使用 ``freegsnke.build_machine.tokamak``。
线圈用单丝近似矩形截面（几何中心 + dR=宽、dZ=高），导体电流由
``Machine.set_all_coil_currents`` 按与 FreeGS 一致的匝数逻辑注入。
"""

from __future__ import annotations

import contextlib
import io
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

PF_XML_DEFAULT = Path(__file__).resolve().parents[2] / "scan_data" / "machine" / "EAST_config" / "pf_active.xml"
WALL_XML_DEFAULT = Path(__file__).resolve().parents[2] / "scan_data" / "machine" / "EAST_config" / "wall.xml"

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_EAST_CONFIG_WALL_IDS = _REPO_ROOT / "scan_data" / "machine" / "EAST_config" / "wall.xml"
_EAST_CONFIG_PF_IDS = _REPO_ROOT / "scan_data" / "machine" / "EAST_config" / "pf_active.xml"


def vessel_annular_outlines_rz_from_wall_xml(
    wall_xml: str | Path = WALL_XML_DEFAULT,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, tuple[np.ndarray, np.ndarray] | None]:
    """解析 IDS ``wall.xml`` 中 ``vessel/annular/outline_inner``、``outline_outer``（剖面上两条 D 形折线）。

    与 ``scan_data/unused/plot_east_vessel_green_rz.ipynb`` 一致：图中所画绿色大 D 来自此处，
    而非 ``tf.xml``（后者通常为环向线圈电路元数据，不含此类轮廓）。
    """
    wall_xml = Path(wall_xml)
    if not wall_xml.is_file():
        return None, None

    def _parse_closed_rz(container: ET.Element | None) -> tuple[np.ndarray, np.ndarray] | None:
        if container is None:
            return None
        r_el, z_el = container.find("r"), container.find("z")
        if r_el is None or z_el is None or r_el.text is None or z_el.text is None:
            return None
        rs = [float(x) for x in r_el.text.replace("\n", ",").split(",") if x.strip()]
        zs = [float(x) for x in z_el.text.replace("\n", ",").split(",") if x.strip()]
        if len(rs) != len(zs) or len(rs) < 3:
            return None
        return np.asarray(rs, dtype=np.float64), np.asarray(zs, dtype=np.float64)

    tree = ET.parse(wall_xml)
    root = tree.getroot()
    inner_el = root.find(".//outline_inner")
    outer_el = root.find(".//outline_outer")
    inner = _parse_closed_rz(inner_el)
    outer = _parse_closed_rz(outer_el)
    return inner, outer


def vessel_annular_rz_for_overlay(
    wall_xml: str | Path = WALL_XML_DEFAULT,
) -> tuple[
    tuple[np.ndarray, np.ndarray] | None,
    tuple[np.ndarray, np.ndarray] | None,
    Path | None,
]:
    """叠图专用：先读 ``wall_xml``；若无 ``vessel/annular`` 内外轮廓再试仓库内 IDS ``EAST_config/wall.xml``。

    默认 ``WALL_XML_DEFAULT``（freegs 自带）往往只有 limiter、**不含** IDS annular，叠图会没有绿线；
    回退文件与 ``plot_east_vessel_green_rz.ipynb`` 一致。建机仍用调用方传入的 ``wall_xml``，不受回退影响。
    """
    wall_xml = Path(wall_xml)
    inner, outer = vessel_annular_outlines_rz_from_wall_xml(wall_xml)
    if inner is not None or outer is not None:
        return inner, outer, wall_xml
    if _EAST_CONFIG_WALL_IDS.is_file() and wall_xml.resolve() != _EAST_CONFIG_WALL_IDS.resolve():
        inn2, out2 = vessel_annular_outlines_rz_from_wall_xml(_EAST_CONFIG_WALL_IDS)
        if inn2 is not None or out2 is not None:
            return inn2, out2, _EAST_CONFIG_WALL_IDS
    return inner, outer, None


def _parse_one_pf_xml_rects_rzwh(
    pf_path: Path,
) -> dict[str, tuple[float, float, float, float]]:
    """单文件：IDS ``rectangle (r,z,width,height)`` 或 freegs 风格 ``center+width+height`` → 名称 → (R,Z,dR,dZ)。"""
    if not pf_path.is_file():
        return {}
    tree = ET.parse(pf_path)
    out: dict[str, tuple[float, float, float, float]] = {}
    for coil in tree.findall(".//coil"):
        ne = coil.find("name")
        if ne is None or not (ne.text and ne.text.strip()):
            continue
        name = ne.text.strip()
        geom = coil.find("element/geometry")
        if geom is None:
            continue
        rect = geom.find("rectangle")
        if rect is not None:
            r_e, z_e, w_e, h_e = rect.find("r"), rect.find("z"), rect.find("width"), rect.find("height")
            if any(x is None or x.text is None for x in (r_e, z_e, w_e, h_e)):
                continue
            out[name] = (
                float(r_e.text.strip().replace(",", " ").split()[0]),
                float(z_e.text.strip().replace(",", " ").split()[0]),
                float(w_e.text.strip().replace(",", " ").split()[0]),
                float(h_e.text.strip().replace(",", " ").split()[0]),
            )
            continue
        c_el, w_el, h_el = geom.find("center"), geom.find("width"), geom.find("height")
        if c_el is None or w_el is None or h_el is None or c_el.text is None:
            continue
        part = c_el.text.split(",")
        if len(part) < 2:
            continue
        out[name] = (float(part[0]), float(part[1]), float(w_el.text), float(h_el.text))
    return out


def pf_coils_rzwh_for_overlay(
    pf_xml: str | Path = PF_XML_DEFAULT,
) -> tuple[dict[str, tuple[float, float, float, float]], Path | None]:
    """叠图用 PF/IC 几何：与 ``plot_east_vessel_green_rz`` 一致，**优先**仓库 ``EAST_config/pf_active.xml``（IDS 矩形）。

    建机仍用 ``pf_xml``；此函数仅影响 ``07`` 叠图轮廓。若两路径均未解析出任何线圈，返回空字典与 None。
    """
    pf_path = Path(pf_xml)
    if _EAST_CONFIG_PF_IDS.is_file():
        d = _parse_one_pf_xml_rects_rzwh(_EAST_CONFIG_PF_IDS)
        if d:
            return d, _EAST_CONFIG_PF_IDS
    if pf_path.is_file():
        d2 = _parse_one_pf_xml_rects_rzwh(pf_path)
        if d2:
            return d2, pf_path
    return {}, None


def wall_outline_rz_from_xml(wall_xml: str | Path = WALL_XML_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    wall_xml = Path(wall_xml)
    tree_wall = ET.parse(wall_xml)
    p_text = tree_wall.find(".//limiter/unit/outline/points").text
    if p_text is None:
        raise ValueError(f"no limiter outline points in {wall_xml}")
    p_text = p_text.strip()
    vals = [float(x) for x in p_text.replace("\n", ",").split(",") if x.strip()]
    r = np.asarray(vals[0::2], dtype=np.float64)
    z = np.asarray(vals[1::2], dtype=np.float64)
    return r, z


def limiter_wall_dicts_from_xml(
    wall_xml: str | Path = WALL_XML_DEFAULT,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """freegsnke 需要的 list[{\"R\":r,\"Z\":z}, ...]。"""
    rw, zw = wall_outline_rz_from_xml(wall_xml)
    lim = [{"R": float(rw[i]), "Z": float(zw[i])} for i in range(rw.size)]
    # 壁与 limiter 同轮廓（与 build_machine 默认 wall=limiter 一致，显式双份供绘图）
    return lim, list(lim)


def active_coils_dict_from_pf_xml(pf_xml: str | Path = PF_XML_DEFAULT) -> dict[str, Any]:
    """解析 pf_active.xml → freegsnke ``active_coils_data`` 字典。"""
    pf_xml = Path(pf_xml)
    tree = ET.parse(pf_xml)
    out: dict[str, Any] = {}
    for coil in tree.findall("coil"):
        name_el = coil.find("name")
        if name_el is None or not name_el.text:
            continue
        name = name_el.text.strip()
        geom = coil.find("element/geometry")
        if geom is None:
            continue
        cx, cy = map(float, geom.find("center").text.split(","))
        w = float(geom.find("width").text)
        h = float(geom.find("height").text)
        tw = int(float(coil.find("element/turns_with_sign").text))
        polarity = 1.0 if tw >= 0 else -1.0
        mult = float(abs(tw))
        # 单丝 + 等效矩形截面（与 MultiCoil 面积电流模型一致的量级）
        out[name] = {
            "R": np.array([cx], dtype=np.float64),
            "Z": np.array([cy], dtype=np.float64),
            "dR": float(w),
            "dZ": float(h),
            "resistivity": 2.0e-8,
            "polarity": polarity,
            "multiplier": mult,
        }
    return out


def build_east_tokamak_freegsnke(
    pf_xml: str | Path = PF_XML_DEFAULT,
    wall_xml: str | Path = WALL_XML_DEFAULT,
    *,
    quiet: bool = True,
):
    """返回 freegsnke ``Machine``（带 limiter）。"""
    from freegsnke.build_machine import tokamak

    active = active_coils_dict_from_pf_xml(pf_xml)
    limiter_data, wall_data = limiter_wall_dicts_from_xml(wall_xml)
    buf = io.StringIO()
    ctx = contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()
    with ctx:
        mac = tokamak(
            active_coils_data=active,
            passive_coils_data=[],
            limiter_data=limiter_data,
            wall_data=wall_data,
        )
    return mac


def currents_dict_to_vec(tokamak, amps_by_name: dict[str, float]) -> np.ndarray:
    """按 ``tokamak`` 电路顺序构造 ``current_vec``（导体电流 A）。"""
    names = [label for label, _ in tokamak.coils]
    vec = np.zeros(len(names), dtype=np.float64)
    for i, nm in enumerate(names):
        if nm in amps_by_name:
            vec[i] = float(amps_by_name[nm])
    return vec


def plot_machine_rz(
    tokamak,
    out_path: str | Path,
    *,
    pf_xml: str | Path = PF_XML_DEFAULT,
    title: str = "EAST machine (freegsnke)",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 8.0))
    lim = tokamak.limiter
    if lim is not None and len(lim.R):
        ax.plot(lim.R, lim.Z, "k-", lw=1.2, label="limiter")
        ax.plot(np.r_[lim.R, lim.R[0]], np.r_[lim.Z, lim.Z[0]], "k-", lw=1.2)
    w = getattr(tokamak, "wall", None)
    if w is not None and len(w.R) and w is not lim:
        ax.plot(w.R, w.Z, color="0.45", ls="--", lw=0.9, label="wall")
    # 线圈矩形（由 XML 几何重画，不依赖内部 MultiCoil 细丝）
    active = active_coils_dict_from_pf_xml(pf_xml)
    for nm, d in active.items():
        cx, cy = float(d["R"][0]), float(d["Z"][0])
        w0, h0 = d["dR"], d["dZ"]
        rect = plt.Rectangle(
            (cx - w0 / 2, cy - h0 / 2),
            w0,
            h0,
            fill=False,
            lw=0.8,
            edgecolor="C0",
        )
        ax.add_patch(rect)
        ax.text(cx, cy, nm, ha="center", va="center", fontsize=5, color="C0")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
