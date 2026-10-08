"""Measured timings from a job's `campaign.log`.

The only per-task wall time anywhere in a run comes from asyncflow's state lines:

    ... radical.asyncflow.workflow_manager task.000007 is in RUNNING state
    ... radical.asyncflow.workflow_manager task.000007 is in DONE state

The task ids are global and say nothing about which tool ran. Every Tier A/B run is one
serial chain (`replicas: 1`, `concurrency: 1`), so between `submitted rNNNN: {...}` and
`reaped rNNNN` the k-th task is the k-th entry of that submitted dict. That mapping holds only
for serial chains: a run whose task count disagrees with its stage count raises
TimelineMismatch instead of guessing, and so would a two-lineage run.

Times are the log's local wall clock (naive datetimes), which is also what sacct reports.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

_TS = r"(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})"
_PATTERNS = {
    "launch": re.compile(_TS + r" .*impress_a\.launch\s+host="),
    "backend_ready": re.compile(_TS + r" .*engine: \w+ backend ready"),
    "submitted": re.compile(_TS + r" .*executor submitted (?P<run>r\d+): (?P<nodes>\{.*?\})"),
    "reaped": re.compile(_TS + r" .*executor reaped (?P<run>r\d+):"),
    "task": re.compile(_TS + r" .*\btask\.(?P<n>\d+) is in (?P<state>RUNNING|DONE|FAILED|CANCELED)"),
    "stop": re.compile(_TS + r" .*stop requested:"),
    "shutdown_done": re.compile(_TS + r" .*Shutdown completed for all components"),
    "returned": re.compile(_TS + r" .*campaign returned after"),
}


class TimelineMismatch(ValueError):
    """The log cannot be mapped onto the run's stages without guessing."""


def _ts(s: str) -> datetime:
    # the log's local wall clock, compared only with itself and with sacct (also local)
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f")  # noqa: DTZ007


@dataclass
class Timeline:
    tasks: pd.DataFrame                          # run, stage, task_id, tool, start, end, seconds
    runs: pd.DataFrame                           # run, submitted, reaped, seconds
    marks: dict[str, datetime] = field(default_factory=dict)  # launch, backend_ready, ...


def parse(log: Path | str) -> Timeline:
    text = Path(log).read_text(errors="replace")
    marks: dict[str, datetime] = {}
    runs: dict[str, dict] = {}
    current: str | None = None
    started: dict[str, datetime] = {}
    task_rows: list[dict] = []

    for line in text.splitlines():
        for kind in ("launch", "backend_ready", "stop", "shutdown_done", "returned"):
            m = _PATTERNS[kind].match(line)
            if m:
                marks.setdefault(kind, _ts(m["ts"]))
        if m := _PATTERNS["submitted"].match(line):
            current = m["run"]
            nodes = ast.literal_eval(m["nodes"])      # the executor logs a Python dict repr
            runs[current] = {"run": current, "submitted": _ts(m["ts"]), "reaped": None,
                             "stages": list(nodes.items()), "seen": []}
        elif m := _PATTERNS["reaped"].match(line):
            runs[m["run"]]["reaped"] = _ts(m["ts"])
            current = None
        elif m := _PATTERNS["task"].match(line):
            n, state, ts = int(m["n"]), m["state"], _ts(m["ts"])
            if state == "RUNNING":
                started[n] = ts
                if current is None:
                    raise TimelineMismatch(f"task.{n} started outside any submitted run")
                runs[current]["seen"].append(n)
            else:
                task_rows.append({"n": n, "start": started.get(n), "end": ts, "state": state})

    ended = {r["n"]: r for r in task_rows}
    rows = []
    for r in runs.values():
        if len(r["seen"]) != len(r["stages"]):
            raise TimelineMismatch(
                f"{r['run']}: {len(r['seen'])} tasks in the log for {len(r['stages'])} stages; "
                "the serial-chain mapping does not apply")
        for k, (n, (task_id, tool)) in enumerate(zip(r["seen"], r["stages"], strict=True)):
            e = ended.get(n)
            if e is None or e["start"] is None:
                raise TimelineMismatch(f"{r['run']}: task.{n} ({tool}) has no RUNNING/DONE pair")
            rows.append({"run": r["run"], "stage": k, "task_id": task_id, "tool": tool,
                         "task_no": n, "state": e["state"], "start": e["start"], "end": e["end"],
                         "seconds": (e["end"] - e["start"]).total_seconds()})
    tasks = pd.DataFrame(rows)
    run_df = pd.DataFrame([{"run": r["run"], "submitted": r["submitted"], "reaped": r["reaped"],
                            "seconds": (r["reaped"] - r["submitted"]).total_seconds()
                            if r["reaped"] else None} for r in runs.values()])
    return Timeline(tasks=tasks, runs=run_df, marks=marks)


def phases(tl: Timeline, acct: dict | None = None) -> dict[str, float]:
    """Seconds spent in each phase of one job, end to end.

    queue        sacct Submit -> Start (needs the sacct snapshot)
    prelaunch    Start -> the launcher's first line (module loads, env, venv)
    startup      launcher -> backend ready
    tasks        sum of task RUNNING -> DONE
    in_run_gaps  inside a run, everything that is not a task (dispatch, absorb, QC)
    between_runs reaped -> next submitted (policy decision, composition, admission)
    lead_in      backend ready -> first submit
    shutdown     last reap -> campaign returned (stop, teardown)
    epilogue     campaign returned -> sacct End (the EXIT trap)
    """
    m, t, r = tl.marks, tl.tasks, tl.runs.sort_values("submitted")
    out: dict[str, float] = {}
    if acct:
        sub, start, end = (_iso(acct.get(k)) for k in ("Submit", "Start", "End"))
        if sub and start:
            out["queue"] = (start - sub).total_seconds()
        if start and "launch" in m:
            out["prelaunch"] = (m["launch"] - start).total_seconds()
        if end and "returned" in m:
            out["epilogue"] = (end - m["returned"]).total_seconds()
    if "launch" in m and "backend_ready" in m:
        out["startup"] = (m["backend_ready"] - m["launch"]).total_seconds()
    if "backend_ready" in m and len(r):
        out["lead_in"] = (r.submitted.iloc[0] - m["backend_ready"]).total_seconds()
    out["tasks"] = float(t.seconds.sum())
    out["in_run_gaps"] = float(r.seconds.sum() - t.seconds.sum())
    out["between_runs"] = float(sum((b - a).total_seconds() for a, b in
                                    zip(r.reaped.iloc[:-1], r.submitted.iloc[1:])))
    if "returned" in m and len(r):
        out["shutdown"] = (m["returned"] - r.reaped.iloc[-1]).total_seconds()
    return out


def _iso(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s and s not in ("Unknown", "None") else None
