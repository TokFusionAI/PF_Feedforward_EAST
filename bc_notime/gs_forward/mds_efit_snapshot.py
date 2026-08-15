"""EFIT 时刻快照：本地 ``{shot}.h5`` 与 MDS ``efit_east`` 二选一 / 自动回退.

读取方式与 `notebook/05_stable_shot_pred_slices_gs_export.ipynb` §5 一致：
使用 `scan_data.compat._fetch_node_data`（内部 ``Tree(..., mode='NORMAL')``；避 EAST 上 ``TreeOpen($,$,1)`` 的 TdiEXTRA_ARG）+
``efit_east_path``）。

供 freegsnke 前向 GS 使用；不依赖 ``freegs``。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from scan_data.compat import _DEFAULT_MDS_SERVER, _fetch_node_data

EFIT_DB_DEFAULT = Path("/data/EFIT")
MDS_TREE_DEFAULT = "efit_east"


def _interp_1d(t_axis: np.ndarray, y: np.ndarray, t_query: float) -> np.ndarray | float:
    t_axis = np.asarray(t_axis, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        return float(np.interp(t_query, t_axis, y))
    if y.shape[0] != t_axis.size and y.shape[1] == t_axis.size:
        y = y.T
    if y.shape[0] != t_axis.size:
        raise ValueError(f"time len {t_axis.size} vs y.shape[0]={y.shape[0]}")
    out = np.empty(y.shape[1:], dtype=np.float64)
    flat = y.reshape(y.shape[0], -1)
    out_flat = out.reshape(-1)
    for j in range(flat.shape[1]):
        out_flat[j] = float(np.interp(t_query, t_axis, flat[:, j]))
    return out.reshape(y.shape[1:])


def load_efit_snapshot_h5(
    shot: int,
    time_target: float,
    efit_dir: str | Path = EFIT_DB_DEFAULT,
) -> dict[str, Any]:
    """从 EFIT h5 取最接近 ATIME 的一帧（与 ``freegs_adapter.load_efit_snapshot`` 字段对齐）。"""
    fp = Path(efit_dir) / f"{shot}.h5"
    with h5py.File(fp, "r") as f:
        atime = np.asarray(f["ATIME"][:], dtype=np.float64)
        k = int(np.argmin(np.abs(atime - float(time_target))))
        out: dict[str, Any] = {
            "shot": int(shot),
            "t_target": float(time_target),
            "t_efit": float(atime[k]),
            "k_efit": k,
            "bcentr": float(f["BCENTR"][k]),
            "betap": float(f["BETAP"][k]),
            "pprime": np.asarray(f["PPRIME"][k], dtype=np.float64),
            "ffprim": np.asarray(f["FFPRIM"][k], dtype=np.float64),
            "fpol": np.asarray(f["FPOL"][k], dtype=np.float64),
            "r8": np.asarray(f["R8"][k], dtype=np.float64),
            "z8": np.asarray(f["Z8"][k], dtype=np.float64),
            "rmaxis": float(f["RMAXIS"][k]),
            "zmaxis": float(f["ZMAXIS"][k]),
            "source": "h5",
        }
    return out


def load_efit_snapshot_mds_interpolated(
    shot: int,
    time_query: float,
    *,
    tree: str = MDS_TREE_DEFAULT,
    mds_server: str | None = None,
) -> dict[str, Any] | None:
    """与 notebook 05 ``read_scalar_from_mds_efit`` 同一插值思想；支持 (T,) 与 (T,npsi)。"""
    srv = (mds_server or os.environ.get("MDS_HOSTNAME", _DEFAULT_MDS_SERVER)).strip()

    def read_interp(path: str) -> Any:
        y, t = _fetch_node_data(int(shot), tree, path, mds_server=srv)
        y = np.asarray(y, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).ravel() if t is not None else None
        if t is None or t.size != y.reshape(y.shape[0], -1).shape[0]:
            raise ValueError(f"{path}: invalid time axis")
        y2 = y if y.ndim >= 1 else y.reshape(1)
        if y2.ndim == 1 and y2.size != t.size:
            raise ValueError(f"{path}: len mismatch")
        return _interp_1d(t, y2, float(time_query))

    try:
        out: dict[str, Any] = {
            "shot": int(shot),
            "t_target": float(time_query),
            "t_efit": float(time_query),
            "k_efit": -1,
            "source": "mds",
            "bcentr": float(read_interp("BCENTR")),
            "betap": float(read_interp("BETAP")),
            "rmaxis": float(read_interp("RMAXIS")),
            "zmaxis": float(read_interp("ZMAXIS")),
            "pprime": np.asarray(read_interp("PPRIME"), dtype=np.float64).ravel(),
            "ffprim": np.asarray(read_interp("FFPRIM"), dtype=np.float64).ravel(),
            "fpol": np.asarray(read_interp("FPOL"), dtype=np.float64).ravel(),
            "r8": np.asarray(read_interp("R8"), dtype=np.float64).ravel(),
            "z8": np.asarray(read_interp("Z8"), dtype=np.float64).ravel(),
        }
    except Exception:
        return None
    if out["r8"].size != 8 or out["z8"].size != 8:
        return None
    return out


def load_efit_snapshot(
    shot: int,
    time_target: float,
    *,
    efit_dir: str | Path = EFIT_DB_DEFAULT,
    prefer_mds: bool = False,
    mds_tree: str = MDS_TREE_DEFAULT,
    mds_server: str | None = None,
    efit_source: str = "auto",
) -> dict[str, Any]:
    """``efit_source``: ``h5`` | ``mds`` | ``auto``（无 h5 文件则用 MDS）。

    ``prefer_mds=True`` 时优先尝试 MDS，失败再 h5。
    """
    h5p = Path(efit_dir) / f"{shot}.h5"
    srv = mds_server

    def from_mds() -> dict[str, Any] | None:
        return load_efit_snapshot_mds_interpolated(shot, time_target, tree=mds_tree, mds_server=srv)

    if prefer_mds:
        m = from_mds()
        if m is not None:
            return m

    if efit_source == "mds":
        m = from_mds()
        if m is None:
            raise FileNotFoundError(f"MDS {mds_tree} shot={shot} 读取失败")
        return m

    if efit_source == "h5":
        if not h5p.is_file():
            raise FileNotFoundError(f"missing EFIT h5: {h5p}")
        return load_efit_snapshot_h5(shot, time_target, efit_dir=efit_dir)

    # auto
    if h5p.is_file():
        return load_efit_snapshot_h5(shot, time_target, efit_dir=efit_dir)
    m = from_mds()
    if m is not None:
        return m
    raise FileNotFoundError(
        f"EFIT: 无本地文件 {h5p} 且 MDS 读取失败；请检查 MDS_HOSTNAME / 网络或改用 --efit-source mds"
    )
