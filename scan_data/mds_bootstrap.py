"""MDS 客户端引导：与 ``notebook/scan_data_rtefit_test.ipynb`` §1 思路一致（已按 mdsthin 实际布局修正）。

MIT ``mdsthin`` 的 ``Tree`` 在子包 ``mdsthin.MDSplus`` 中，**不能**把顶层 ``mdsthin`` 注册为 ``sys.modules['MDSplus']``，
否则 ``from MDSplus import Tree`` 会失败。
"""

from __future__ import annotations

import os
import sys

# EAST MDS server. 集群(login+计算节点)到该主机有直连路由(RTT~0.16ms，同机房/校园网)，
# 但全集群 DNS 坏了，名字 mds.ipp.ac.cn 解析失败 → 默认直接用 IP，避免卡在 DNS。
# 仍可用 MDS_HOSTNAME / mds_server 参数覆盖。compat.py 复用此常量。
_DEFAULT_MDS_SERVER = "mds.ipp.ac.cn"


def bootstrap_mdsplus() -> bool:
    """供 ``from MDSplus import Tree`` 使用：优先绑定 ``mdsthin.MDSplus``，否则顶层 ``mdsthin``（若含 Tree），再否则原生 ``MDSplus``。"""
    mod = sys.modules.get("MDSplus")
    if mod is not None and getattr(mod, "Tree", None) is not None:
        return True
    if mod is not None:
        # 曾错误绑定为无 Tree 的顶层 mdsthin 时，撤掉后重绑
        sys.modules.pop("MDSplus", None)

    try:
        import mdsthin.MDSplus as mds_compat

        sys.modules["MDSplus"] = mds_compat
        return True
    except ImportError:
        pass
    try:
        import mdsthin

        if getattr(mdsthin, "Tree", None) is not None:
            sys.modules["MDSplus"] = mdsthin
            return True
    except ImportError:
        pass
    try:
        import MDSplus  # noqa: F401

        return True
    except ImportError:
        return False


def ensure_default_mds_connection(mds_server: str | None) -> None:
    """为 **mdsthin** 建立 ``Tree`` 所需的默认 ``Connection``。

    rtefit 测试 notebook 里用 ``Conn = MDSplus.Connection`` 再 ``Conn(mds_server)`` 发 TDI；
    ``mdsthin.ext.tree.Tree`` 则要求 ``getDefaultConnection()`` 非空，否则会报
    ``Unable to create an mdsthin.Tree without a connection``。

    C 版官方 ``MDSplus`` 一般不走此分支（模块路径不含 ``mdsthin`` 时直接返回）。
    """
    mod = sys.modules.get("MDSplus")
    if mod is None:
        return
    mod_file = (getattr(mod, "__file__", "") or "").replace("\\", "/")
    if "mdsthin" not in mod_file:
        return

    srv = (
        (mds_server or os.environ.get("MDS_HOSTNAME") or os.environ.get("MDS_HOST") or _DEFAULT_MDS_SERVER)
    ).strip()
    if srv:
        os.environ.setdefault("MDS_HOST", srv)

    gdc = getattr(mod, "getDefaultConnection", None)
    sdc = getattr(mod, "setDefaultConnection", None)
    Conn = getattr(mod, "Connection", None)
    if not (callable(gdc) and callable(sdc) and Conn is not None):
        return
    if gdc() is None:
        sdc(Conn(srv))


def mdsplus_available() -> tuple[bool, str]:
    """若无法加载 MDS 客户端，返回 (False, 说明)。与「炮号不在库」是两类问题。

    供 notebook / ``bc.precursor_export`` 使用；实现仅依赖本模块，避免导入重型 ``precursor_export``
    时因 **Jupyter 缓存旧版** ``sys.modules['precursor_export']`` 导致 ``mdsplus_available`` 缺失。
    """
    if bootstrap_mdsplus():
        return True, ""
    return (
        False,
        "当前环境无法加载 mdsthin 或 MDSplus（与「炮号不在数据库」不同）。"
        "若使用 mdsthin：应绑定子包 ``import mdsthin.MDSplus`` 再 ``sys.modules['MDSplus']=mdsthin.MDSplus``，"
        "或由站点安装完整 ``mdsplus``。本仓库 ``bootstrap_mdsplus()`` 已按此处理；"
        "若仍失败请 ``import sys; sys.modules.pop('MDSplus', None)`` 后重试。"
        " 使用 mdsthin 时还需 MDSip 可达；``ensure_default_mds_connection`` 会在读树前建立 ``Connection``。",
    )
