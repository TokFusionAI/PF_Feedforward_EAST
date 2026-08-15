import os
import pathlib
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import h5py
import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required by scan_data.compat") from exc

# EAST MDS server 默认值取自 mds_bootstrap._DEFAULT_MDS_SERVER（单一真相源）：
# 集群直连 EAST 用 IP (mds.ipp.ac.cn == mds.ipp.ac.cn)，因全集群 DNS 坏、名字解析失败。
from .mds_bootstrap import _DEFAULT_MDS_SERVER


def screen_print(msg: str) -> None:
    print(msg, flush=True)


def strpath2path(path_like) -> pathlib.Path:
    if isinstance(path_like, pathlib.Path):
        return path_like
    path_s = os.path.expanduser(os.path.expandvars(str(path_like)))
    return pathlib.Path(path_s)


def load_yaml_config(path_like):
    path = strpath2path(path_like)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def _write_h5_value(parent, key, value, is_overwrite=False):
    if key in parent:
        if is_overwrite:
            del parent[key]
        else:
            del parent[key]

    if isinstance(value, dict):
        grp = parent.create_group(key)
        for sub_key, sub_val in value.items():
            _write_h5_value(grp, sub_key, sub_val, is_overwrite=is_overwrite)
        return

    if value is None:
        parent.create_dataset(key, data=h5py.Empty("f4"))
        return

    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "O"}:
        parent.create_dataset(key, data=str(value))
    else:
        parent.create_dataset(key, data=arr)


def save_to_file(file_name, data_dict, is_overwrite=False):
    file_path = strpath2path(file_name)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if file_path.exists() else "w"
    with h5py.File(file_path, mode) as hf:
        for key, val in data_dict.items():
            _write_h5_value(hf, key, val, is_overwrite=is_overwrite)


def _h5_node_to_obj(node):
    if isinstance(node, h5py.Dataset):
        if node.shape is None:
            return None
        return node[()]
    out = {}
    for k, v in node.items():
        out[k] = _h5_node_to_obj(v)
    return out


def convert_hdf5_2dict(h5_or_path):
    if isinstance(h5_or_path, (str, pathlib.Path, os.PathLike)):
        with h5py.File(strpath2path(h5_or_path), "r") as hf:
            return _h5_node_to_obj(hf)
    return _h5_node_to_obj(h5_or_path)


def hf_keys_fetch(h5_file, keys):
    result = {}
    with h5py.File(strpath2path(h5_file), "r") as hf:
        for key in keys:
            if key not in hf:
                result[key] = None
                continue
            node = hf[key]
            result[key] = _h5_node_to_obj(node)
    return result


def calc_sample_frequency(times):
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1 or t.size < 2:
        return None
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return None
    return float(1.0 / np.median(dt))


def interp_nD_time_arr(dst_time, src_time, src_data, dtype=np.float32):
    if src_data is None:
        return None
    x_new = np.asarray(dst_time, dtype=np.float64)
    x_old = np.asarray(src_time, dtype=np.float64)
    y_old = np.asarray(src_data)
    if y_old.ndim == 1:
        y_new = np.interp(x_new, x_old, y_old)
        return y_new.astype(dtype)

    lead = y_old.shape[0]
    rest = int(np.prod(y_old.shape[1:]))
    y2 = y_old.reshape(lead, rest)
    out = np.empty((x_new.size, rest), dtype=np.float64)
    for i in range(rest):
        out[:, i] = np.interp(x_new, x_old, y2[:, i])
    return out.reshape((x_new.size, *y_old.shape[1:])).astype(dtype)


@dataclass
class ShapeParams:
    R8: np.ndarray
    Z8: np.ndarray
    rsurf: np.ndarray
    zsurf: np.ndarray
    aminor: float
    bminor: float
    elong: float
    triu: float
    tril: float
    squo: float
    squi: float
    sqli: float
    sqlo: float


def _build_arc(start, end, n, step=1):
    """Boundary arc indices from *start* to *end* (exclusive), wrapping at n."""
    if step > 0:
        if end > start:
            return np.arange(start, end)
        return np.concatenate([np.arange(start, n), np.arange(0, end)])
    if end < start:
        return np.arange(start, end, step=-1)
    return np.concatenate([np.arange(start, -1, step=-1),
                          np.arange(n - 1, end, step=-1)])


def _find_squareness(r_arc, z_arc, r_lo, r_hi, z_lo, z_hi,
                      flip_condition=False):
    """Squareness control point via diagonal crossing in normalised coords.

    Normalises the arc into a unit square ``(0,0)–(1,1)`` and finds where
    the boundary crosses the diagonal *y = x*.  Returns ``(R, Z, squareness)``.
    ``flip_condition=True`` uses the *x > y* crossing instead.
    """
    dr = r_hi - r_lo
    dz = z_hi - z_lo
    if abs(dr) < 1e-12 or abs(dz) < 1e-12 or len(r_arc) < 2:
        return np.nan, np.nan, 0.0
    x = (r_arc - r_lo) / dr
    y = (z_arc - z_lo) / dz
    found = False
    for i in range(1, len(x)):
        if (not flip_condition and y[i] > x[i]) or \
           (flip_condition and x[i] > y[i]):
            found = True
            break
    if not found:
        return np.nan, np.nan, 0.0
    denom = x[i] - x[i - 1] - y[i] + y[i - 1]
    if abs(denom) < 1e-14:
        return np.nan, np.nan, 0.0
    f = (y[i - 1] - x[i - 1]) / denom
    b = f * x[i] + (1 - f) * x[i - 1]
    return (float(b * dr + r_lo),
            float(b * dz + z_lo),
            float((b * np.sqrt(2) - 1) / (np.sqrt(2) - 1)))


def shape_params(r, z):
    """8-point shape parameters via squareness interpolation (EFIT method).

    Implements the same algorithm as ``EFIT_tools.shape_params``: divides the
    boundary into four arcs between adjacent extreme points, normalises each
    arc into a unit square, and locates the diagonal crossing to obtain the
    squareness control points (2/4/6/8).
    """
    r = np.asarray(r, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    n = len(r)
    if n < 8:
        nan8 = np.full(8, np.nan, dtype=np.float32)
        return ShapeParams(R8=nan8, Z8=nan8,
                          rsurf=np.float32(np.nan), zsurf=np.float32(np.nan),
                          aminor=np.nan, bminor=np.nan, elong=np.nan,
                          triu=np.nan, tril=np.nan,
                          squo=np.nan, squi=np.nan, sqli=np.nan, sqlo=np.nan)

    # Extreme points: outer(0), upper(2), inner(4), lower(6)
    io = int(np.argmax(r))
    iu = int(np.argmax(z))
    ii = int(np.argmin(r))
    il = int(np.argmin(z))

    # Signed area → boundary winding direction
    A = float(np.sum((r[:n - 1] + r[1:n]) * (z[1:n] - z[:n - 1])))

    R8 = np.array([r[io], np.nan, r[iu], np.nan,
                   r[ii], np.nan, r[il], np.nan], dtype=np.float64)
    Z8 = np.array([z[io], np.nan, z[iu], np.nan,
                   z[ii], np.nan, z[il], np.nan], dtype=np.float64)

    step = 1 if A > 0 else -1
    k1 = _build_arc(io, iu, n, step)  # outer → upper
    k2 = _build_arc(iu, ii, n, step)  # upper → inner
    k3 = _build_arc(ii, il, n, step)  # inner → lower
    k4 = _build_arc(il, io, n, step)  # lower → outer

    # Squareness interpolation per quadrant
    R8[1], Z8[1], squo = _find_squareness(
        r[k1], z[k1], R8[2], R8[0], Z8[0], Z8[2])
    R8[3], Z8[3], squi = _find_squareness(
        r[k2], z[k2], R8[2], R8[4], Z8[4], Z8[2], flip_condition=True)
    R8[5], Z8[5], sqli = _find_squareness(
        r[k3], z[k3], R8[6], R8[4], Z8[4], Z8[6])
    R8[7], Z8[7], sqlo = _find_squareness(
        r[k4], z[k4], R8[6], R8[0], Z8[0], Z8[6], flip_condition=True)

    rsurf = float(0.5 * (R8[0] + R8[4]))
    zsurf = float(0.5 * (Z8[2] + Z8[6]))
    aminor = float(0.5 * (R8[0] - R8[4]))
    bminor = float(0.5 * (Z8[2] - Z8[6]))
    elong = bminor / (aminor + 1e-12)
    triu = (rsurf - float(R8[2])) / (aminor + 1e-12)
    tril = (rsurf - float(R8[6])) / (aminor + 1e-12)

    return ShapeParams(
        R8=R8.astype(np.float32),
        Z8=Z8.astype(np.float32),
        rsurf=np.float32(rsurf),
        zsurf=np.float32(zsurf),
        aminor=float(aminor),
        bminor=float(bminor),
        elong=float(elong),
        triu=float(triu),
        tril=float(tril),
        squo=float(squo),
        squi=float(squi),
        sqli=float(sqli),
        sqlo=float(sqlo),
    )


def _set_tree_env(tree_name: str, mds_server: str | None):
    """与 ``scan_data/notebook/scan_data_rtefit_test.ipynb`` 一致：打开树前设置 path，且同时写树名小写与大写前缀。

    部分 MDSplus / mdsthin 变体只认 ``EFIT_EAST_path`` 或只认 ``efit_east_path``，双写避免 ``FOPENR``。
    """
    srv = (
        (mds_server or os.environ.get("MDS_HOSTNAME") or os.environ.get("MDS_HOST") or _DEFAULT_MDS_SERVER)
    ).strip()
    if not srv:
        return
    val = f"{srv}::"
    os.environ[f"{tree_name}_path"] = val
    os.environ[f"{tree_name.upper()}_path"] = val


from .mds_bootstrap import bootstrap_mdsplus, ensure_default_mds_connection


def _fetch_node_data(shot: int, tree_name: str, node_name: str, mds_server=None):
    if not bootstrap_mdsplus():
        raise ModuleNotFoundError(
            "Neither mdsthin nor MDSplus is available; see scan_data/mds_bootstrap.py "
            "and scan_data/notebook/scan_data_rtefit_test.ipynb §1"
        )

    _set_tree_env(tree_name, mds_server)
    ensure_default_mds_connection(mds_server)

    name = node_name if str(node_name).startswith("\\") else f"\\{node_name}"
    mds_mod = sys.modules.get("MDSplus")
    mod_file = (getattr(mds_mod, "__file__", "") or "").replace("\\", "/")

    # mdsthin.Tree 走本地模型路径，EAST 上常报 TREE-E-FOPENR；rtefit 用 Connection.openTree + get（TDI TreeOpen($,$)）
    if mds_mod is not None and "mdsthin" in mod_file:
        from MDSplus import Connection, getDefaultConnection

        srv = (mds_server or os.environ.get("MDS_HOSTNAME") or os.environ.get("MDS_HOST") or _DEFAULT_MDS_SERVER).strip()
        conn = getDefaultConnection()
        if conn is None:
            conn = Connection(srv)
        conn.openTree(tree_name, int(shot))
        try:
            data = np.asarray(conn.get(name).data())
            try:
                t = np.asarray(conn.get(f"dim_of({name})").data())
            except Exception:
                t = None
        finally:
            try:
                conn.closeTree(tree_name, int(shot))
            except Exception:
                pass
        return data, t

    from MDSplus import Tree

    with Tree(tree=tree_name, shot=shot, mode="NORMAL") as tree:
        n = tree.getNode(name)
        data = np.asarray(n.getData().data())
        try:
            t = np.asarray(n.dim_of().data())
        except Exception:
            t = None
    return data, t


class _GetDataCompat:
    @staticmethod
    def getNodeDataByTree(shot, tree_name, node_name, mds_server=None):
        return _fetch_node_data(shot, tree_name, node_name, mds_server=mds_server)

    @staticmethod
    def getShotData(shot, nodeList, start=0, end=None, resampled_rate=1e4,
                    is_debug=False, mds_server=None):
        del start, end, resampled_rate, is_debug
        trees = ["pcs_east", "east", "efit_east", "energy_east"]
        out = {"shot": int(shot)}
        for node in nodeList:
            data = None
            t = None
            for tree in trees:
                try:
                    data, t = _fetch_node_data(shot, tree, node, mds_server=mds_server)
                    break
                except Exception:
                    continue
            out[node] = data
            out[f"{node}_time"] = t
        if "time" not in out:
            for node in nodeList:
                tn = out.get(f"{node}_time")
                if tn is not None:
                    out["time"] = tn
                    break
        return out

    @staticmethod
    def getRawShotData(shot, nodeList, dtype=np.float32, is_debug=False, mds_server=None):
        del is_debug
        out = _GetDataCompat.getShotData(
            shot,
            nodeList,
            mds_server=mds_server,
        )
        casted = {}
        for k, v in out.items():
            if isinstance(v, np.ndarray):
                casted[k] = v.astype(dtype)
            else:
                casted[k] = v
        return casted

    @staticmethod
    def getData(mds_server=None):
        class _Client:
            def __init__(self, server):
                self.server = server

            def fetch_nodes_data_by_tree(self, shot, node_names, tree_name,
                                         is_debug=False, is_pure_node=True):
                del is_debug, is_pure_node
                out = {}
                for node in node_names:
                    try:
                        d, _ = _fetch_node_data(shot, tree_name, node, mds_server=self.server)
                        out[node] = d
                    except Exception:
                        out[node] = None
                return out

        return _Client(mds_server)


getData = _GetDataCompat


class _MDSConfigCompat:
    @staticmethod
    def get_mds_config():
        host = os.environ.get("MDSSCAN_HOSTNAME", _DEFAULT_MDS_SERVER)
        return SimpleNamespace(hostname=host)


mdsConfig = _MDSConfigCompat


class _EASTAnalyzerCompat:
    class EASTAnalyzer:
        def __init__(self, shot_file, mds_server=None):
            del mds_server
            keys = ["PCRL01", "time", "valid_control_start", "valid_control_end"]
            self.shot_dict = hf_keys_fetch(shot_file, keys)
            if self.shot_dict.get("valid_control_start") is None or self.shot_dict.get("valid_control_end") is None:
                t = self.shot_dict.get("time")
                if t is not None and len(t) > 1:
                    self.shot_dict["valid_control_start"] = float(t[0])
                    self.shot_dict["valid_control_end"] = float(t[-1])

        def find_continuous_flat_segments(self, flat_threshold=1e-3, min_duration=0.5):
            ip = self.shot_dict.get("PCRL01")
            t = self.shot_dict.get("time")
            if ip is None or t is None or len(t) < 3:
                return []
            ip = np.asarray(ip)
            t = np.asarray(t)
            didt = np.abs(np.gradient(ip, t))
            mask = didt <= flat_threshold
            return _mask_to_segments(mask, t, min_duration=min_duration)

        def find_relative_flat_segments(self, n_decimal_palaces=2, min_duration=0.5, max_slow_rise_rate=0.1):
            del max_slow_rise_rate
            return self.find_continuous_flat_segments(
                flat_threshold=10 ** (-int(n_decimal_palaces)),
                min_duration=min_duration,
            )


def _mask_to_segments(mask, t, min_duration=0.5):
    mask = np.asarray(mask, dtype=bool)
    t = np.asarray(t, dtype=np.float64)
    if mask.size != t.size:
        return []
    segments = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        if t[j - 1] - t[i] >= min_duration:
            segments.append((float(t[i]), float(t[j - 1])))
        i = j
    return segments


eastAnalyzer = _EASTAnalyzerCompat