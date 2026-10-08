"""Snapshot SLURM accounting for the analysed jobs, so the analysis works off-cluster.

    python -m performance_analysis.snapshot_sacct [--runs ROOT] [--jobs ...] [--dry-run]

Writes `<run root>/sacct.json`: for each job, the allocation line and the `.batch` step
(where TotalCPU and MaxRSS live). sacct is only reachable on the cluster and its records
age out, so take the snapshot once on Delta; the export then carries it. Without a snapshot
the analyses query sacct live and say so.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from . import data

FIELDS = ("JobID", "JobName", "Partition", "NNodes", "AllocTRES", "Submit", "Start", "End",
          "Elapsed", "ElapsedRaw", "TotalCPU", "MaxRSS", "State", "ExitCode")


def query(jobs) -> dict:
    """{job: {"alloc": {...}, "batch": {...}}} straight from sacct; {} if sacct is unavailable."""
    try:
        out = subprocess.run(["sacct", "-P", "-n", "-j", ",".join(jobs),
                              "--format=" + ",".join(FIELDS)],
                             capture_output=True, text=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    res: dict = {}
    for line in out.splitlines():
        row = dict(zip(FIELDS, line.split("|"), strict=False))
        jid, _, step = row["JobID"].partition(".")
        if jid in jobs and step in ("", "batch"):
            res.setdefault(jid, {})["alloc" if not step else "batch"] = row
    return res


def tres(s: str) -> dict:
    """'billing=8000,cpu=64,gres/gpu=4,mem=240G,node=1' -> {'cpu': 64, 'gpu': 4, ...}"""
    out = {}
    for kv in (s or "").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            k = k.replace("gres/", "")
            try:
                out[k] = int(v)
            except ValueError:
                out[k] = v
    return out


def seconds(hms: str) -> float:
    """sacct durations: [D-]HH:MM:SS[.fff] or MM:SS.fff."""
    if not hms:
        return float("nan")
    days = 0
    if "-" in hms:
        d, hms = hms.split("-", 1)
        days = int(d)
    parts = [float(p) for p in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return days * 86400 + h * 3600 + m * 60 + s


def mem_gb(s: str) -> float:
    if not s:
        return float("nan")
    unit = s[-1].upper()
    mult = {"K": 1 / 1024 ** 2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}.get(unit)
    return float(s[:-1]) * mult if mult else float(s) / 1024 ** 3


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=None)
    p.add_argument("--jobs", nargs="+", default=list(data.TIER_AB))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    runs = a.runs or data.default_runs()
    snap = query(a.jobs)
    missing = sorted(set(a.jobs) - set(snap))
    if missing:
        raise SystemExit(f"sacct returned nothing for {missing}; not writing a partial snapshot")
    text = json.dumps(snap, indent=1, sort_keys=True)
    if a.dry_run:
        print(text)
        return
    (runs / "sacct.json").write_text(text + "\n")
    print(f"wrote {runs / 'sacct.json'} ({len(snap)} jobs)")


if __name__ == "__main__":
    main()
