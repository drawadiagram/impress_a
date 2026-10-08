"""Shared context, output helpers and small-n statistics for the analyses.

Every analysis is a function `analyse(ctx) -> str` that writes its CSVs and figures under
`ctx.out` and returns one markdown section. `cli(analyse)` runs one on its own; `report.py`
runs them all. Statistics are descriptive on purpose: n is 15-17 designs, so intervals, never
p-values, and every section states its n.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, timeline

RNG = np.random.default_rng(20261008)   # fixed, so a re-run reproduces every interval


class Ctx:
    def __init__(self, runs: Path, out: Path, jobs=data.TIER_AB):
        self.runs, self.out, self.jobs = Path(runs), Path(out), tuple(jobs)
        (self.out / "figures").mkdir(parents=True, exist_ok=True)
        (self.out / "tables").mkdir(parents=True, exist_ok=True)

    @cached_property
    def tasks(self) -> pd.DataFrame:
        return data.tasks_table(self.runs, self.jobs)

    @cached_property
    def run_rows(self) -> pd.DataFrame:
        return data.runs_table(self.runs, self.jobs)

    @cached_property
    def gates(self) -> pd.DataFrame:
        return data.gates_table(self.tasks)

    @cached_property
    def timelines(self) -> dict[str, timeline.Timeline]:
        return {j: timeline.parse(data.job_dir(self.runs, j) / "campaign.log") for j in self.jobs}

    @cached_property
    def timed_tasks(self) -> pd.DataFrame:
        """Timeline tasks with the job attached, joined to the ledger's tasks."""
        frames = [tl.tasks.assign(job=j) for j, tl in self.timelines.items()]
        t = pd.concat(frames, ignore_index=True)
        return t.merge(self.tasks[["job", "run", "task_id", "qc"]], on=["job", "run", "task_id"],
                       how="left", validate="one_to_one")

    @cached_property
    def sacct(self) -> dict:
        """The snapshot if one was taken, else a live query (empty off-cluster)."""
        from . import snapshot_sacct
        snap = data.sacct(self.runs)
        self.sacct_source = "snapshot" if snap else "live sacct"
        return snap or snapshot_sacct.query(list(self.jobs))

    @cached_property
    def specs(self) -> dict:
        return data.specs()

    @cached_property
    def objectives(self) -> list[dict]:
        return data.objectives(data.job_dir(self.runs, self.jobs[0]))

    def job_order(self) -> list[str]:
        """Jobs in the order they ran."""
        return sorted(self.jobs, key=lambda j: (self.timelines[j].marks.get("launch") is None,
                                     self.timelines[j].marks.get("launch") or 0, j))

    # -- output -------------------------------------------------------------------------------
    def table(self, df: pd.DataFrame, name: str) -> str:
        p = self.out / "tables" / f"{name}.csv"
        df.to_csv(p, index=False)
        return f"tables/{name}.csv"

    def figure(self, fig, name: str) -> str:
        p = self.out / "figures" / f"{name}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return f"![{name}](figures/{name}.png)"


def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    _plt.rcParams.update({"figure.figsize": (7, 4), "axes.spines.top": False,
                          "axes.spines.right": False, "font.size": 9})
    return _plt


def md_table(df: pd.DataFrame, floatfmt: str = ".3g") -> str:
    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else format(v, floatfmt)
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


# -- small-n statistics ---------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bootstrap(fn, *arrays, n: int = 5000, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile interval of fn(*resampled arrays); arrays are resampled jointly (paired)."""
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    m = len(arrays[0])
    stats = []
    for _ in range(n):
        idx = RNG.integers(0, m, m)
        v = fn(*(a[idx] for a in arrays))
        if np.isfinite(v):
            stats.append(v)
    if not stats:
        return (float("nan"), float("nan"))
    return tuple(np.quantile(stats, [alpha / 2, 1 - alpha / 2]))


def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


# -- command line ---------------------------------------------------------------------------

def parser(doc: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=doc)
    p.add_argument("--runs", type=Path, default=None,
                   help="run root (default: $IMPRESS_A_RUNS or $WORK_DIR/impress_a_runs)")
    p.add_argument("--out", type=Path, default=None,
                   help="output dir (default: $WORK_DIR/analysis/<today>); never the checkout")
    p.add_argument("--jobs", nargs="+", default=list(data.TIER_AB))
    return p


def make_ctx(a: argparse.Namespace) -> Ctx:
    runs = a.runs or data.default_runs()
    out = a.out
    if out is None:
        work = os.environ.get("WORK_DIR")
        if not work:
            raise SystemExit("set --out or WORK_DIR")
        out = Path(work) / "analysis" / _dt.datetime.now(_dt.UTC).date().isoformat()
    repo = Path(__file__).resolve().parent.parent
    if out.resolve().is_relative_to(repo) and not out.resolve().is_relative_to(
            repo / "performance_analysis" / "out"):
        raise SystemExit(f"refusing to write analysis output into the checkout: {out}")
    return Ctx(runs, out, a.jobs)


def cli(analyse) -> None:
    a = parser(analyse.__doc__).parse_args()
    ctx = make_ctx(a)
    md = analyse(ctx)
    name = analyse.__module__.rsplit(".", 1)[-1]
    (ctx.out / f"{name}.md").write_text(md + "\n")
    print(md)
