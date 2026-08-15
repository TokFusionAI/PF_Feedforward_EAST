"""兼容入口：实现位于 ``bc.training._smoke_test``。"""
from __future__ import annotations

from bc.training._smoke_test import main

if __name__ == "__main__":
    raise SystemExit(main())
