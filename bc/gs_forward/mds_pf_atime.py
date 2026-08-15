"""当本地 ``{shot}.h5`` 不存在时，从 MDSplus 重建 ATIME 对齐样本（与 ``build_sample_arrays`` 同形）.

实现与 `notebook/05_stable_shot_pred_slices_gs_export.ipynb` §5 及 `scan_data.compat._fetch_node_data`
一致：通过 ``efit_east`` / ``pcs_east`` 树读节点，环境变量 ``{tree}_path = MDS_HOSTNAME::``。

- **R8 时间维长度** 作为 EFIT 重建步数 ``T`` 的权威来源；**ATIME** 与 ``T`` 对齐（``ATIME.data``
  或 ``dim_of`` 与 ``T`` 等长时取为物理时间轴）。
- **R8, Z8**: ``efit_east``。
- **PCPF*, PCRL01, lmsr, lmsz**: ``pcs_east``，``np.interp`` 到 EFIT 时间栅格。

环境变量 ``MDS_HOSTNAME``（默认 ``mds.ipp.ac.cn``）可由 CLI ``--mds-server`` 覆盖传入。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from bc.common.constants import D_ACTION, D_STATE, PCPF_NAMES, T_MAX
from bc.data.dataset import _safe_diff_dt
from scan_data.compat import _fetch_node_data
from scan_data.mds_bootstrap import _DEFAULT_MDS_SERVER, ensure_default_mds_connection


def configure_mds_paths(mds_server: str) -> None:
    """与 ``scan_data/notebook/scan_data_rtefit_test.ipynb`` / notebook 05 一致：双树 path + 大小写双写 + 同步主机名。

    notebook 在 ``openTree`` 前设置 ``{tree}_path`` 与 ``{TREE.upper()}_path`` 均为 ``{mds_server}::``；
    此处对 ``efit_east`` / ``pcs_east`` 同样处理，并把解析后的主机写入 ``MDS_HOSTNAME`` / ``MDS_HOST``，
    供 ``mdsthin`` 的 ``ensure_default_mds_connection`` 与 TDI 使用。
    """
    host = (mds_server or os.environ.get("MDS_HOSTNAME", _DEFAULT_MDS_SERVER)).strip()
    val = f"{host}::"
    for tree in ("efit_east", "pcs_east"):
        os.environ[f"{tree}_path"] = val
        os.environ[f"{tree.upper()}_path"] = val
    os.environ["MDS_HOSTNAME"] = host
    os.environ.setdefault("MDS_HOST", host)
    # 与 notebook：先 Connection(server) 再 openTree；mdsthin.Tree 依赖默认 Connection
    ensure_default_mds_connection(host)


def _interp_y_on_x(xq: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    xq = np.asarray(xq, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        return np.interp(xq, x, y).astype(np.float32)
    out = np.empty((xq.size, y.shape[1]), dtype=np.float64)
    for j in range(y.shape[1]):
        out[:, j] = np.interp(xq, x, y[:, j])
    return out.astype(np.float32)


def _as_T8_r8z8(arr: np.ndarray, n_time: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected 2D R8/Z8 array, got shape {a.shape}")
    if a.shape[0] == n_time and a.shape[1] == 8:
        return a.astype(np.float32)
    if a.shape[1] == n_time and a.shape[0] == 8:
        return a.T.astype(np.float32)
    raise ValueError(f"cannot reshape R8/Z8 with shape {a.shape} to (T={n_time}, 8)")


def _resolve_atime_axis(shot: int, T: int, mds_server: str) -> np.ndarray:
    """与 R8 第一维长度 ``T`` 对齐的一维 ATIME（秒）。"""
    at_d, at_t = _fetch_node_data(shot, "efit_east", "ATIME", mds_server=mds_server)
    at_d = np.asarray(at_d, dtype=np.float64).ravel()
    at_t = None if at_t is None else np.asarray(at_t, dtype=np.float64).ravel()
    if at_d.size == T:
        return at_d
    if at_t is not None and at_t.size == T:
        return at_t
    raise RuntimeError(
        f"efit_east ATIME: data len={at_d.size}, dim len={at_t.size if at_t is not None else None}, "
        f"expected T={T} (from R8). 参见 notebook/05 §5。"
    )


def _pcs_series(shot: int, node: str, atime: np.ndarray, mds_server: str) -> np.ndarray:
    """读 pcs_east 标量序列并插值到 ``atime``（与 notebook 05 对 PCRL01 的约定一致）。"""
    y, t = _fetch_node_data(shot, "pcs_east", node, mds_server=mds_server)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    t = np.asarray(t, dtype=np.float64).ravel() if t is not None else None
    if t is None or t.size != y.ravel().shape[0]:
        raise RuntimeError(
            f"pcs_east {node}: 需要与 notebook 05 相同的一维时间轴 (dim_of 与 data 等长)，"
            f"got t={None if t is None else t.shape}, y.shape={y.shape}"
        )
    y1 = y.ravel() if y.ndim == 1 else y
    if y1.ndim == 1:
        return _interp_y_on_x(atime, t, y1).ravel().astype(np.float32)
    return _interp_y_on_x(atime, t, y1)


def build_sample_arrays_from_mds(
    shot: int,
    state_mean: np.ndarray | None = None,
    state_std: np.ndarray | None = None,
    action_mean: np.ndarray | None = None,
    action_std: np.ndarray | None = None,
    T_max: int = T_MAX,
    *,
    mds_server: str | None = None,
) -> dict[str, np.ndarray | int]:
    """等价于 ``build_sample_arrays(h5_path, ...)`` 的返回 dict（numpy）。"""
    srv = (mds_server or os.environ.get("MDS_HOSTNAME", _DEFAULT_MDS_SERVER)).strip()
    configure_mds_paths(srv)

    r8_raw, _ = _fetch_node_data(shot, "efit_east", "R8", mds_server=srv)
    r8_raw = np.asarray(r8_raw, dtype=np.float64)
    if r8_raw.ndim != 2:
        raise RuntimeError(f"efit_east R8: expected 2D array, got shape {r8_raw.shape}")
    if r8_raw.shape[1] == 8:
        T = int(r8_raw.shape[0])
    elif r8_raw.shape[0] == 8:
        r8_raw = r8_raw.T
        T = int(r8_raw.shape[0])
    else:
        raise RuntimeError(f"efit_east R8: cannot infer (T,8) from shape {r8_raw.shape}")

    atime_full = _resolve_atime_axis(shot, T, srv)
    if atime_full.size != T:
        raise RuntimeError(f"ATIME length {atime_full.size} != R8 T={T}")

    z8_raw, _ = _fetch_node_data(shot, "efit_east", "Z8", mds_server=srv)
    R8f = _as_T8_r8z8(r8_raw, T)
    Z8f = _as_T8_r8z8(np.asarray(z8_raw, dtype=np.float64), T)

    m = atime_full >= 0.0
    atime = atime_full[m]
    R8 = R8f[m]
    Z8 = Z8f[m]
    if atime.size < 3:
        raise RuntimeError(f"efit_east ATIME too short after t>=0 filter: {atime.size}")

    time_phys = atime.astype(np.float32)
    Tlen = int(time_phys.shape[0])

    pcrl = _pcs_series(shot, "PCRL01", atime, srv)
    lmsr = _pcs_series(shot, "lmsr", atime, srv)
    lmsz = _pcs_series(shot, "lmsz", atime, srv)

    pcpf_cols: list[np.ndarray] = []
    for name in PCPF_NAMES:
        yi = _pcs_series(shot, name, atime, srv)
        pcpf_cols.append(np.asarray(yi, dtype=np.float32).ravel())
    action = np.stack(pcpf_cols, axis=1)

    dt_phys = _safe_diff_dt(time_phys)

    state = np.concatenate(
        [
            R8,
            Z8,
            lmsr[:, None],
            lmsz[:, None],
            pcrl[:, None],
            time_phys[:, None],
            dt_phys[:, None],
        ],
        axis=1,
    )

    state = np.nan_to_num(state, nan=0.0, copy=False)
    action = np.nan_to_num(action, nan=0.0, copy=False)

    if state_mean is not None and state_std is not None:
        state = (state - state_mean.astype(np.float32)) / (state_std.astype(np.float32) + 1e-8)
    if action_mean is not None and action_std is not None:
        action = (action - action_mean.astype(np.float32)) / (action_std.astype(np.float32) + 1e-8)

    action_mask = np.ones((Tlen, D_ACTION), dtype=bool)

    T_eff = min(Tlen, T_max)
    out_state = np.zeros((T_max, D_STATE), dtype=np.float32)
    out_action = np.zeros((T_max, D_ACTION), dtype=np.float32)
    out_amask = np.zeros((T_max, D_ACTION), dtype=bool)
    out_time = np.zeros((T_max,), dtype=np.float32)
    out_dt = np.zeros((T_max,), dtype=np.float32)
    token_mask = np.zeros((T_max,), dtype=bool)

    out_state[:T_eff] = state[:T_eff]
    out_action[:T_eff] = action[:T_eff]
    out_amask[:T_eff] = action_mask[:T_eff]
    out_time[:T_eff] = time_phys[:T_eff]
    out_dt[:T_eff] = dt_phys[:T_eff]
    token_mask[:T_eff] = True

    return {
        "state": out_state,
        "action": out_action,
        "action_mask": out_amask,
        "token_mask": token_mask,
        "time_phys": out_time,
        "dt_phys": out_dt,
        "T": T_eff,
    }


def pf_atime_h5_exists(dataset_root: str | Path, shot: int) -> bool:
    return (Path(dataset_root) / f"{int(shot)}.h5").is_file()
