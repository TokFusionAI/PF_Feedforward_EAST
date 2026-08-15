#!/usr/bin/env python3
"""从 ``meta/test_shots.txt`` 随机抽取若干炮号，并生成可执行的探针脚本。

典型用法（仓库根目录，需后续在能连 MDS + 有 torch 的环境执行生成的 shell）::

    python -m bc.batch_freegsnke.random_test_shots --n 3 --seed 42

（``python -m bc.random_test_shots`` 薄包装仍可用。）

写出 ``results/random_freegsnke_shots/seed<seed>/``::

    - shots.txt          一行一炮号
    - shots.json         元数据（源文件、seed、列表）
    - run_pipeline.sh    precursor → pred 三阶段+GS → EFIT 自洽+GS

若某炮 ``precursor_export`` 失败（无归档/网络），脚本内对该炮 ``continue``，不中断其余炮。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent


def _read_shots(path: Path) -> list[int]:
    txt = path.read_text(encoding="utf-8")
    return sorted({int(x) for x in txt.split() if x.strip().isdigit()})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots-file", type=str, default=str(_REPO / "meta" / "test_shots.txt"))
    ap.add_argument("--n", type=int, default=3, help="抽取数量")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（复现同一组炮号）")
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="输出目录，默认 results/random_freegsnke_shots/seed<seed>",
    )
    args = ap.parse_args()

    sf = Path(args.shots_file)
    if not sf.is_file():
        print(f"找不到 {sf}", file=sys.stderr)
        return 2

    shots = _read_shots(sf)
    if len(shots) < int(args.n):
        print(f"文件中仅 {len(shots)} 个炮号，少于 --n={args.n}", file=sys.stderr)
        return 3

    rng = random.Random(int(args.seed))
    chosen = sorted(rng.sample(shots, int(args.n)))

    out = Path(args.out_dir) if args.out_dir else _REPO / "results" / "random_freegsnke_shots" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    (out / "shots.txt").write_text("\n".join(str(s) for s in chosen) + "\n", encoding="utf-8")
    meta = {
        "shots": chosen,
        "n": len(chosen),
        "seed": int(args.seed),
        "source_file": str(sf.resolve()),
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "shots.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    repo = str(_REPO.resolve())
    sh_list = " ".join(str(s) for s in chosen)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'REPO="{repo}"',
        'cd "$REPO"',
        f"# random_test_shots seed={args.seed} shots={chosen}",
        "export PYTHONPATH=\"${REPO}:${PYTHONPATH:-}\"",
        f"for S in {sh_list}; do",
        '  echo "======== shot $S ========"',
        '  python -m bc.precursor_export --shot "$S" --source auto || { echo "[SKIP] precursor_export failed shot $S"; continue; }',
        f'  python -m bc.batch_freegsnke.run_freegsnke_pred_three_phases --shot "$S" --run-gs \\',
        f'    --gs-out-dir "results/freegsnke_eval_random/seed{args.seed}/$S" || '
        f'{{ echo "[WARN] run_freegsnke_pred_three_phases failed $S"; continue; }}',
        '  python -m bc.batch_freegsnke.export_efit_self_slices --shot "$S" --run-gs \\',
        f'    --from-manifest "results/freegsnke_pred_three_phases/$S/three_phases_manifest.json" \\',
        f'    --gs-out-dir "results/freegsnke_eval_efit_self_random/seed{args.seed}/$S" || '
        f'echo "[WARN] export_efit_self_slices failed $S"',
        "done",
        "",
    ]
    sh_path = out / "run_pipeline.sh"
    sh_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sh_path.chmod(sh_path.stat().st_mode | 0o111)

    print(json.dumps(meta, indent=2))
    print(f"已写入 {out}", flush=True)
    print(f"执行: bash {sh_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
