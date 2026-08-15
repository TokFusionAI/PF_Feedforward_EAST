"""按炮号 EFIT ATIME 栅格对齐：从 MDS（及可选 EFIT h5）导出 PF/PCS 与剖面，落盘供验证阶段离线读取.

不含装置几何（线圈 XML、壁等仍用本地文件）。与 ``mds_pf_atime.build_sample_arrays_from_mds``
及 ``mds_efit_snapshot`` 的字段约定一致。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from bc.common.constants import PCPF_NAMES

from .mds_efit_snapshot import EFIT_DB_DEFAULT
from .mds_pf_atime import (
    _as_T8_r8z8,
    _pcs_series,
    _resolve_atime_axis,
    configure_mds_paths,
)
from scan_data.compat import _fetch_node_data
from scan_data.mds_bootstrap import _DEFAULT_MDS_SERVER, bootstrap_mdsplus, mdsplus_available

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class PrecursorUnavailableError(RuntimeError):
    """炮号在库中不可用或数据无法对齐时抛出（消息面向用户，中文）。"""


def check_shot_available(shot: int, mds_server: str | None = None) -> tuple[bool, str]:
    """检查 efit_east 是否可读（作为「库中有此炮」的判据之一）。

    若返回 False 且说明中含「未安装 MDSplus」，请先换带 MDS 客户端的环境，再解读为炮号问题。
    """
    m_ok, m_msg = mdsplus_available()
    if not m_ok:
        return False, m_msg

    srv = (mds_server or os.environ.get("MDS_HOSTNAME", _DEFAULT_MDS_SERVER)).strip()
    try:
        configure_mds_paths(srv)
        r8_raw, _ = _fetch_node_data(int(shot), "efit_east", "R8", mds_server=srv)
        r8_raw = np.asarray(r8_raw, dtype=np.float64)
        if r8_raw.ndim != 2 or r8_raw.size == 0:
            return False, f"efit_east R8 数据异常 shape={getattr(r8_raw, 'shape', None)}"
    except Exception as exc:  # noqa: BLE001
        return False, f"读取 efit_east R8 失败：{exc}"
    return True, ""


def _efit_profile_TN(data: np.ndarray, T: int) -> np.ndarray:
    """将 PPRIME/FFPRIM/FPOL 等整理为 (T, Npsi)。"""
    a = np.asarray(data, dtype=np.float64)
    if a.ndim == 1:
        if a.size == T:
            return a.reshape(T, 1)
        raise ValueError(f"expected 1D len T={T} or 2D, got shape {a.shape}")
    if a.shape[0] == T:
        return a
    if a.shape[1] == T:
        return a.T
    raise ValueError(f"cannot align profile of shape {a.shape} to T={T}")


def _efit_scalar_T(data: np.ndarray, T: int) -> np.ndarray:
    a = np.asarray(data, dtype=np.float64).ravel()
    if a.size == T:
        return a
    if a.ndim == 2 and a.shape[0] == T and a.shape[1] == 1:
        return a[:, 0]
    if a.ndim == 2 and a.shape[1] == T and a.shape[0] == 1:
        return a[0, :]
    raise ValueError(f"scalar EFIT node shape {a.shape} != ({T},)")


def _fetch_efit_series(shot: int, node: str, T: int, mds_server: str) -> np.ndarray:
    y, _t = _fetch_node_data(int(shot), "efit_east", node, mds_server=mds_server)
    y = np.asarray(y, dtype=np.float64)
    if node in ("PPRIME", "FFPRIM", "FPOL"):
        return _efit_profile_TN(y, T)
    return _efit_scalar_T(y, T)


def fetch_precursor_series_mds(shot: int, mds_server: str | None = None) -> dict[str, Any]:
    """从 MDS 拉取与 ATIME（t>=0 段）对齐的完整时间序列。"""
    srv = (mds_server or os.environ.get("MDS_HOSTNAME", _DEFAULT_MDS_SERVER)).strip()
    ok, msg = check_shot_available(shot, srv)
    if not ok:
        if msg.startswith("当前环境无法加载") or "mdsthin" in msg.lower():
            raise PrecursorUnavailableError(msg)
        raise PrecursorUnavailableError(
            f"数据库中无此炮号或 efit_east 无法读取（shot={shot}）。{msg}"
        )
    configure_mds_paths(srv)

    r8_raw, _ = _fetch_node_data(int(shot), "efit_east", "R8", mds_server=srv)
    r8_raw = np.asarray(r8_raw, dtype=np.float64)
    if r8_raw.ndim != 2:
        raise PrecursorUnavailableError(f"efit_east R8 维数错误 shape={r8_raw.shape}")
    if r8_raw.shape[1] == 8:
        T = int(r8_raw.shape[0])
    elif r8_raw.shape[0] == 8:
        r8_raw = r8_raw.T
        T = int(r8_raw.shape[0])
    else:
        raise PrecursorUnavailableError(f"efit_east R8 形状无法解释为 (T,8)：{r8_raw.shape}")

    atime_full = _resolve_atime_axis(int(shot), T, srv)
    if atime_full.size != T:
        raise PrecursorUnavailableError(
            f"ATIME 长度 {atime_full.size} 与 R8 时间维 T={T} 不一致。"
        )

    z8_raw, _ = _fetch_node_data(int(shot), "efit_east", "Z8", mds_server=srv)
    R8f = _as_T8_r8z8(r8_raw, T)
    Z8f = _as_T8_r8z8(np.asarray(z8_raw, dtype=np.float64), T)

    m = atime_full >= 0.0
    atime_u = np.asarray(atime_full[m], dtype=np.float64)
    R8u = np.asarray(R8f[m], dtype=np.float64)
    Z8u = np.asarray(Z8f[m], dtype=np.float64)
    Tu = int(atime_u.shape[0])
    if Tu < 3:
        raise PrecursorUnavailableError(f"t>=0 后有效步数过少：{Tu}")

    def take_mask(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float64)
        if a.shape[0] == T:
            return a[m]
        if a.ndim == 2 and a.shape[1] == T:
            return a[:, m].T
        raise PrecursorUnavailableError(f"节点维数与 R8 不一致：shape={a.shape}, T={T}")

    pprime = take_mask(_fetch_efit_series(int(shot), "PPRIME", T, srv))
    ffprim = take_mask(_fetch_efit_series(int(shot), "FFPRIM", T, srv))
    fpol = take_mask(_fetch_efit_series(int(shot), "FPOL", T, srv))
    bcentr = take_mask(_fetch_efit_series(int(shot), "BCENTR", T, srv))
    betap = take_mask(_fetch_efit_series(int(shot), "BETAP", T, srv))
    rmaxis = take_mask(_fetch_efit_series(int(shot), "RMAXIS", T, srv))
    zmaxis = take_mask(_fetch_efit_series(int(shot), "ZMAXIS", T, srv))

    if pprime.ndim == 1:
        pprime = pprime.reshape(Tu, -1)
    if ffprim.ndim == 1:
        ffprim = ffprim.reshape(Tu, -1)
    if fpol.ndim == 1:
        fpol = fpol.reshape(Tu, -1)

    pcrl = np.asarray(_pcs_series(int(shot), "PCRL01", atime_u.astype(np.float32), srv), dtype=np.float64).ravel()
    lmsr = np.asarray(_pcs_series(int(shot), "lmsr", atime_u.astype(np.float32), srv), dtype=np.float64).ravel()
    lmsz = np.asarray(_pcs_series(int(shot), "lmsz", atime_u.astype(np.float32), srv), dtype=np.float64).ravel()

    pcpf_cols: list[np.ndarray] = []
    for name in PCPF_NAMES:
        yi = _pcs_series(int(shot), name, atime_u.astype(np.float32), srv)
        pcpf_cols.append(np.asarray(yi, dtype=np.float64).ravel())
    pcpf = np.stack(pcpf_cols, axis=1)

    return {
        "shot": int(shot),
        "mds_server": srv,
        "source": "mds",
        "ATIME": atime_u,
        "R8": R8u,
        "Z8": Z8u,
        "PPRIME": pprime.astype(np.float64),
        "FFPRIM": ffprim.astype(np.float64),
        "FPOL": fpol.astype(np.float64),
        "BCENTR": bcentr.astype(np.float64).ravel(),
        "BETAP": betap.astype(np.float64).ravel(),
        "RMAXIS": rmaxis.astype(np.float64).ravel(),
        "ZMAXIS": zmaxis.astype(np.float64).ravel(),
        "PCRL01": pcrl,
        "lmsr": lmsr,
        "lmsz": lmsz,
        "PCPF": pcpf,
        "PCPF_NAMES": list(PCPF_NAMES),
    }


def fetch_precursor_series_h5(
    shot: int,
    efit_dir: str | Path = EFIT_DB_DEFAULT,
    mds_server: str | None = None,
) -> dict[str, Any]:
    """从本地 EFIT h5 读 EFIT 量；PCS 量仍从 MDS 插值到 h5 的 ATIME（与训练集构造一致）。"""
    m_ok, m_msg = mdsplus_available()
    if not m_ok:
        raise PrecursorUnavailableError(
            m_msg + " 若仅有 EFIT h5、无法使用 MDS，需换带 MDSplus 的环境导出（PCS 仍依赖 pcs_east）。"
        )

    fp = Path(efit_dir) / f"{int(shot)}.h5"
    if not fp.is_file():
        raise PrecursorUnavailableError(f"本地无 EFIT 文件：{fp}；请改用 MDS 导出或检查路径。")
    srv = (mds_server or os.environ.get("MDS_HOSTNAME", _DEFAULT_MDS_SERVER)).strip()
    configure_mds_paths(srv)

    with h5py.File(fp, "r") as f:
        atime_full = np.asarray(f["ATIME"][:], dtype=np.float64).ravel()
        R8f = np.asarray(f["R8"][:], dtype=np.float64)
        Z8f = np.asarray(f["Z8"][:], dtype=np.float64)
        if R8f.shape[1] != 8 and R8f.shape[0] == 8:
            R8f = R8f.T
            Z8f = Z8f.T
        T = int(R8f.shape[0])
        if atime_full.size != T:
            raise PrecursorUnavailableError(f"h5 ATIME len {atime_full.size} != R8 T={T}")

        m = atime_full >= 0.0
        atime_u = atime_full[m]
        Tu = int(atime_u.size)
        R8u = R8f[m]
        Z8u = Z8f[m]

        def take(k: str) -> np.ndarray:
            a = np.asarray(f[k][:], dtype=np.float64)
            if a.shape[0] == T:
                return a[m]
            raise PrecursorUnavailableError(f"h5 {k} 第一维应等于 T={T}, got {a.shape}")

        pprime = take("PPRIME")
        ffprim = take("FFPRIM")
        fpol = take("FPOL")
        bcentr = take("BCENTR").ravel()
        betap = take("BETAP").ravel()
        rmaxis = take("RMAXIS").ravel()
        zmaxis = take("ZMAXIS").ravel()

    pcrl = np.asarray(_pcs_series(int(shot), "PCRL01", atime_u.astype(np.float32), srv), dtype=np.float64).ravel()
    lmsr = np.asarray(_pcs_series(int(shot), "lmsr", atime_u.astype(np.float32), srv), dtype=np.float64).ravel()
    lmsz = np.asarray(_pcs_series(int(shot), "lmsz", atime_u.astype(np.float32), srv), dtype=np.float64).ravel()
    pcpf_cols: list[np.ndarray] = []
    for name in PCPF_NAMES:
        yi = _pcs_series(int(shot), name, atime_u.astype(np.float32), srv)
        pcpf_cols.append(np.asarray(yi, dtype=np.float64).ravel())
    pcpf = np.stack(pcpf_cols, axis=1)

    return {
        "shot": int(shot),
        "mds_server": srv,
        "source": "h5_efit_pcs_mds",
        "efit_h5": str(fp.resolve()),
        "ATIME": atime_u,
        "R8": R8u,
        "Z8": Z8u,
        "PPRIME": pprime,
        "FFPRIM": ffprim,
        "FPOL": fpol,
        "BCENTR": bcentr,
        "BETAP": betap,
        "RMAXIS": rmaxis,
        "ZMAXIS": zmaxis,
        "PCRL01": pcrl,
        "lmsr": lmsr,
        "lmsz": lmsz,
        "PCPF": pcpf,
        "PCPF_NAMES": list(PCPF_NAMES),
    }


def fetch_precursor_series(
    shot: int,
    *,
    mds_server: str | None = None,
    efit_dir: str | Path = EFIT_DB_DEFAULT,
    source: str = "auto",
) -> dict[str, Any]:
    """source: auto | mds | h5 — auto 优先 h5，否则全 MDS。"""
    fp = Path(efit_dir) / f"{int(shot)}.h5"
    if source == "h5":
        return fetch_precursor_series_h5(shot, efit_dir=efit_dir, mds_server=mds_server)
    if source == "mds":
        return fetch_precursor_series_mds(shot, mds_server=mds_server)
    if source == "auto":
        if fp.is_file():
            try:
                return fetch_precursor_series_h5(shot, efit_dir=efit_dir, mds_server=mds_server)
            except Exception:
                return fetch_precursor_series_mds(shot, mds_server=mds_server)
        return fetch_precursor_series_mds(shot, mds_server=mds_server)
    raise ValueError(f"unknown source={source!r}")


def save_precursor_npz(bundle: dict[str, Any], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shot_id = int(bundle.get("shot", 0))
    meta = {
        "shot": shot_id,
        "source": bundle["source"],
        "mds_server": bundle.get("mds_server"),
        "efit_h5": bundle.get("efit_h5"),
        "T": int(bundle["ATIME"].shape[0]),
        "PCPF_NAMES": bundle.get("PCPF_NAMES", list(PCPF_NAMES)),
        "arrays": [
            "ATIME",
            "R8",
            "Z8",
            "PPRIME",
            "FFPRIM",
            "FPOL",
            "BCENTR",
            "BETAP",
            "RMAXIS",
            "ZMAXIS",
            "PCRL01",
            "lmsr",
            "lmsz",
            "PCPF",
        ],
    }
    np.savez_compressed(
        out_path,
        ATIME=bundle["ATIME"],
        R8=bundle["R8"],
        Z8=bundle["Z8"],
        PPRIME=bundle["PPRIME"],
        FFPRIM=bundle["FFPRIM"],
        FPOL=bundle["FPOL"],
        BCENTR=bundle["BCENTR"],
        BETAP=bundle["BETAP"],
        RMAXIS=bundle["RMAXIS"],
        ZMAXIS=bundle["ZMAXIS"],
        PCRL01=bundle["PCRL01"],
        lmsr=bundle["lmsr"],
        lmsz=bundle["lmsz"],
        PCPF=bundle["PCPF"],
    )
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def load_precursor_npz(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    raw = np.load(path, allow_pickle=False)
    meta_path = path.with_suffix(".meta.json")
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    out = {
        "shot": int(meta.get("shot", 0)),
        "source": str(meta.get("source", "npz")),
        "mds_server": meta.get("mds_server"),
        "PCPF_NAMES": meta.get("PCPF_NAMES", list(PCPF_NAMES)),
    }
    for k in (
        "ATIME",
        "R8",
        "Z8",
        "PPRIME",
        "FFPRIM",
        "FPOL",
        "BCENTR",
        "BETAP",
        "RMAXIS",
        "ZMAXIS",
        "PCRL01",
        "lmsr",
        "lmsz",
        "PCPF",
    ):
        out[k] = np.asarray(raw[k])
    raw.close()
    if not out["shot"]:
        out["shot"] = int(meta_path.stem.split("_")[-1]) if "_" in meta_path.stem else 0
    return out


def efit_snapshot_row(bundle: dict[str, Any], k: int, *, shot: int | None = None) -> dict[str, Any]:
    """从落盘 bundle 取第 k 帧，字段与 ``load_efit_snapshot_h5`` / freegsnke 管线一致。"""
    k = int(k)
    at = np.asarray(bundle["ATIME"], dtype=np.float64).ravel()
    if k < 0 or k >= at.size:
        raise IndexError(f"precursor row k={k} 越界，ATIME 长度={at.size}")
    sid = int(shot if shot is not None else bundle.get("shot", 0))
    return {
        "shot": sid,
        "t_target": float(at[k]),
        "t_efit": float(at[k]),
        "k_efit": k,
        "bcentr": float(np.asarray(bundle["BCENTR"], dtype=np.float64).ravel()[k]),
        "betap": float(np.asarray(bundle["BETAP"], dtype=np.float64).ravel()[k]),
        "pprime": np.asarray(bundle["PPRIME"][k], dtype=np.float64).ravel(),
        "ffprim": np.asarray(bundle["FFPRIM"][k], dtype=np.float64).ravel(),
        "fpol": np.asarray(bundle["FPOL"][k], dtype=np.float64).ravel(),
        "r8": np.asarray(bundle["R8"][k], dtype=np.float64).ravel(),
        "z8": np.asarray(bundle["Z8"][k], dtype=np.float64).ravel(),
        "rmaxis": float(np.asarray(bundle["RMAXIS"], dtype=np.float64).ravel()[k]),
        "zmaxis": float(np.asarray(bundle["ZMAXIS"], dtype=np.float64).ravel()[k]),
        "source": f"precursor_npz:{bundle.get('source', '')}",
    }


def nearest_row(atime: np.ndarray, t_query: float) -> int:
    t = np.asarray(atime, dtype=np.float64).ravel()
    return int(np.argmin(np.abs(t - float(t_query))))


def export_slice_npz(
    bundle: dict[str, Any],
    k: int,
    out_path: str | Path,
    *,
    pcpf12_override: np.ndarray | None = None,
    ip_a_override: float | None = None,
) -> Path:
    """导出单帧小文件，验证时可直接 load 无需再选行。

    ``pcpf12_override``：例如 BC 预测的 12 路 PCS 电流（安培），仍与第 ``k`` 行的
    EFIT 剖面 / R8Z8 / BCENTR 等同帧；``ip_a_override`` 非空时覆盖该帧的 ``PCRL01``。
    """
    snap = efit_snapshot_row(bundle, k, shot=bundle.get("shot"))
    if pcpf12_override is not None:
        pcpf12 = np.asarray(pcpf12_override, dtype=np.float64).ravel()
        if pcpf12.size != 12:
            raise ValueError(f"pcpf12_override 须为长度 12，得到 {pcpf12.size}")
    else:
        pcpf12 = np.asarray(bundle["PCPF"][k], dtype=np.float64).ravel()
    pcrl = float(ip_a_override) if ip_a_override is not None else float(
        np.asarray(bundle["PCRL01"], dtype=np.float64).ravel()[k]
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        shot=int(bundle.get("shot", 0)),
        k_efit=int(k),
        t_efit=float(snap["t_efit"]),
        pcpf12=pcpf12,
        Ip_A=pcrl,
        lmsr=float(np.asarray(bundle["lmsr"], dtype=np.float64).ravel()[k]),
        lmsz=float(np.asarray(bundle["lmsz"], dtype=np.float64).ravel()[k]),
        bcentr=snap["bcentr"],
        betap=snap["betap"],
        rmaxis=snap["rmaxis"],
        zmaxis=snap["zmaxis"],
        pprime=snap["pprime"],
        ffprim=snap["ffprim"],
        fpol=snap["fpol"],
        r8=snap["r8"],
        z8=snap["z8"],
    )
    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "shot": int(bundle.get("shot", 0)),
                "k_efit": int(k),
                "t_efit": float(snap["t_efit"]),
                "PCPF_NAMES": list(bundle.get("PCPF_NAMES", PCPF_NAMES)),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return out_path


def load_single_slice_npz(path: str | Path) -> dict[str, Any]:
    """读取 ``export_slice_npz`` 生成的单帧文件。"""
    path = Path(path)
    z = np.load(path, allow_pickle=False)
    shot = int(z["shot"])
    k_efit = int(z["k_efit"])
    t_efit = float(z["t_efit"])
    snap = {
        "shot": shot,
        "t_target": t_efit,
        "t_efit": t_efit,
        "k_efit": k_efit,
        "bcentr": float(z["bcentr"]),
        "betap": float(z["betap"]),
        "pprime": np.asarray(z["pprime"], dtype=np.float64).ravel(),
        "ffprim": np.asarray(z["ffprim"], dtype=np.float64).ravel(),
        "fpol": np.asarray(z["fpol"], dtype=np.float64).ravel(),
        "r8": np.asarray(z["r8"], dtype=np.float64).ravel(),
        "z8": np.asarray(z["z8"], dtype=np.float64).ravel(),
        "rmaxis": float(z["rmaxis"]),
        "zmaxis": float(z["zmaxis"]),
        "source": "precursor_slice_npz",
    }
    pcpf12 = np.asarray(z["pcpf12"], dtype=np.float64).ravel()
    ip_a = float(z["Ip_A"])
    z.close()
    return {"snapshot": snap, "pcpf12": pcpf12, "Ip_A": ip_a}


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 freegsnke 验证用 precursor（ATIME 对齐序列）")
    ap.add_argument("--shot", type=int, required=True)
    ap.add_argument("--out", type=str, default=None, help="输出 .npz，默认 results/freegsnke_precursors/{shot}/precursor.npz")
    ap.add_argument("--mds-server", type=str, default=None)
    ap.add_argument("--efit-dir", type=str, default=str(EFIT_DB_DEFAULT))
    ap.add_argument("--source", type=str, default="auto", choices=("auto", "mds", "h5"))
    ap.add_argument(
        "--export-slice-k",
        type=int,
        default=None,
        help="若指定，则额外写出单帧 slice_kXXX.npz（与全序列同一目录）",
    )
    args = ap.parse_args()

    out = args.out
    if out is None:
        out = str(_REPO_ROOT / "results" / "freegsnke_precursors" / str(args.shot) / "precursor.npz")

    try:
        bundle = fetch_precursor_series(
            args.shot,
            mds_server=args.mds_server,
            efit_dir=args.efit_dir,
            source=args.source,
        )
    except PrecursorUnavailableError as e:
        print(str(e), file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"导出失败：{e}", file=sys.stderr)
        return 1

    bundle["shot"] = int(args.shot)
    p = save_precursor_npz(bundle, out)
    print(json.dumps({"ok": True, "out": str(p.resolve()), "T": int(bundle["ATIME"].shape[0])}, indent=2))

    if args.export_slice_k is not None:
        sk = Path(p).parent / f"slice_k{int(args.export_slice_k):04d}.npz"
        export_slice_npz(bundle, int(args.export_slice_k), sk)
        print(json.dumps({"slice": str(sk.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
