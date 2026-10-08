"""C1 - measured wall time per stage: spread, job-to-job variation, and first-run warm-up."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from .common import RNG, Ctx, cli, md_table, plt


def analyse(ctx: Ctx) -> str:
    t = ctx.timed_tasks.copy()
    t["run_pos"] = t.groupby("job").run.rank(method="dense").astype(int)
    order = [s for s in data.STAGES if s in set(t.tool)]

    summ = (t.groupby("tool").seconds
            .agg(n="count", median="median", q1=lambda s: s.quantile(.25),
                 q3=lambda s: s.quantile(.75), min="min", max="max")
            .reindex(order).reset_index())
    summ["max/min"] = summ["max"] / summ["min"]
    ctx.table(t[["job", "run", "run_pos", "stage", "tool", "start", "end", "seconds"]],
              "c1_task_times")
    ctx.table(summ, "c1_stage_summary")

    # warm-up: first run of a job vs the rest, trust jobs only (the smoke jobs ran once)
    multi = t[t.job.isin([j for j in ctx.jobs if t[t.job == j].run.nunique() > 1])]
    warm = []
    for tool in order:
        s = multi[multi.tool == tool]
        first, rest = s[s.run_pos == 1].seconds.values, s[s.run_pos > 1].seconds.values
        if len(first) and len(rest):
            diff = np.median(first) - np.median(rest)
            lo, hi = _diff_ci(first, rest)
            warm.append({"tool": tool, "n_first": len(first), "n_rest": len(rest),
                         "median_first_s": np.median(first), "median_rest_s": np.median(rest),
                         "diff_s": diff, "ci95": f"[{lo:.1f}, {hi:.1f}]"})
    warm = pd.DataFrame(warm)
    ctx.table(warm, "c1_first_run_warmup")

    by_job = t.pivot_table(index="tool", columns="job", values="seconds", aggfunc="median") \
        .reindex(order)[ctx.job_order()].round(1).reset_index()

    P = plt()
    fig, ax = P.subplots(figsize=(8, 4))
    jobs = ctx.job_order()
    for i, tool in enumerate(order):
        for k, j in enumerate(jobs):
            y = t[(t.tool == tool) & (t.job == j)].seconds
            x = np.full(len(y), i + (k - (len(jobs) - 1) / 2) * 0.12)
            ax.scatter(x, y, s=14, label=j if i == 0 else None, alpha=.8)
    ax.set_xticks(range(len(order)), order, rotation=20)
    ax.set_ylabel("wall time (s)")
    ax.set_title("C1  per-task wall time, by stage and job (each point is one task)")
    ax.legend(fontsize=7, frameon=False)
    img = ctx.figure(fig, "c1_stage_times")

    rf = summ.set_index("tool").loc["rfd3_design"] if "rfd3_design" in order else None
    lead = (f"rfd3 spans {rf['min']:.0f}-{rf['max']:.0f} s ({rf['max/min']:.1f}x) over "
            f"{int(rf['n'])} tasks." if rf is not None else "")
    return "\n\n".join([
        "## C1 - Stage wall times",
        (f"n = {len(t)} tasks from {t.job.nunique()} jobs; time is asyncflow RUNNING -> DONE from "
        f"`campaign.log`, so it includes process start and model load inside the task. {lead}"),
        md_table(summ, ".1f"), img,
        "**Median by job** (seconds, jobs in run order):", md_table(by_job, ".1f"),
        ("**First run of a job vs later runs** (trust jobs; difference of medians, bootstrap 95% "
        "interval). A positive difference is warm-up: caches, page cache, compiled kernels."),
        md_table(warm, ".1f") if len(warm) else "_no multi-run jobs_",
    ])


def _diff_ci(a, b, n=5000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [np.median(RNG.choice(a, len(a))) - np.median(RNG.choice(b, len(b))) for _ in range(n)]
    return tuple(np.quantile(d, [.025, .975]))


if __name__ == "__main__":
    cli(analyse)
