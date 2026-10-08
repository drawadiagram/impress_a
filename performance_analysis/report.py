"""Run every analysis and write one report.

    python -m performance_analysis.report [--runs ROOT] [--out DIR] [--jobs ...]

Writes `<out>/report.md` with every section, `<out>/tables/*.csv` and `<out>/figures/*.png`.
An analysis that raises is reported as a failed section rather than stopping the others.
"""
from __future__ import annotations

import datetime as _dt
import importlib
import subprocess
import traceback
from pathlib import Path

from .common import make_ctx, parser

ANALYSES = ["c1_stage_times", "c2_cost_calibration", "c3_allocation", "c4_scrutiny",
            "c5_drift", "s1_gate_pressure", "s2_upstream_downstream", "s3_front",
            "s4_boltz_confidence", "s5_diversity", "s6_seed_audit", "s7_relax"]


def main() -> None:
    a = parser(__doc__).parse_args()
    ctx = make_ctx(a)
    sections, failed = [], []
    for name in ANALYSES:
        mod = importlib.import_module(f"{__package__}.{name}")
        try:
            sections.append(mod.analyse(ctx))
        except Exception:  # noqa: BLE001 - one broken analysis must not hide the rest
            failed.append(name)
            sections.append(f"## {name} - FAILED\n\n```\n{traceback.format_exc()}```")
    head = [
        "# IMPRESS-A run performance report",
        (f"Generated {_dt.datetime.now(_dt.UTC).isoformat(timespec='seconds')} from `{ctx.runs}` "
        f"(jobs {', '.join(ctx.jobs)}), code at `{_commit()}`. "
        f"sacct: {getattr(ctx, 'sacct_source', 'snapshot')}."),
        ("Small-n throughout: intervals, not p-values. S5/S6 show the designs fall into far "
        "fewer independent groups than their count, so every interval here is optimistic."),
    ]
    if failed:
        head.append(f"**Failed sections:** {', '.join(failed)}")
    (ctx.out / "report.md").write_text("\n\n".join(head + sections) + "\n")
    print(f"wrote {ctx.out / 'report.md'}" + (f" ({len(failed)} failed)" if failed else ""))
    if failed:
        raise SystemExit(1)


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=Path(__file__).parent, timeout=10,
                              check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


if __name__ == "__main__":
    main()
