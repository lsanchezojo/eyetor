"""Receipt script validation behavior."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


RECEIPT_SCRIPT = Path("skills/shopping/scripts/receipt.py")


def _run_receipt(*args: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(RECEIPT_SCRIPT), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout.strip(), proc.stderr
    return json.loads(proc.stdout)


def test_add_without_date_requests_reconfirm_instead_of_argparse_error() -> None:
    result = _run_receipt(
        "add",
        "--store",
        "Alcampo Sevilla",
        "--items",
        '[{"name":"Pan","price":1.0}]',
        "--total",
        "1.0",
    )

    assert result == {
        "ok": True,
        "needs_reconfirm": True,
        "reason": "missing date",
    }
