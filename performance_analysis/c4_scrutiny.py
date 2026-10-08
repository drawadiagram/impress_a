"""C4 - what scrutiny costs: untrusted runs (forced dry run, suspect marking) vs trusted ones.

Same pattern, same allocation shape, so the difference is the interlock's overhead plus noise.
Only runs that ran the full six-stage chain are compared. n is small (the three trust jobs
have 7 trusted runs between them), so the result is a difference with an interval, not a
verdict - and the first run of every job is untrusted, so warm-up (C1) is confounded with
scrutiny; the comparison is repeated without first runs to show how much that matters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import RNG, Ctx, cli, md_table, plt


def analyse(ctx: Ctx) -> str:
    rows = []
    for job, tl in ctx.timelines.items():
        per_task = tl.tasks.groupby("run").seconds.sum()
        for r in tl.runs.itertuples():
            rows.append({"job": job, "run": r.run, "run_s": r.seconds,
                         "task_s": per_task.get(r.run, np.nan),
                         "overhead_s": r.seconds - per_task.get(r.run, np.nan)})
    runs = pd.DataFrame(rows).merge(ctx.run_rows[["job", "run", "trusted", "n_tasks"]],
                                    on=["job", "run"])
    runs = runs[runs.n_tasks == 6]
    runs["first"] = runs.groupby("job").run.transform("min") == runs.run
    ctx.table(runs, "c4_runs")

    out = []
    for label, sub in (("all runs", runs), ("excluding each job's first run", runs[~runs["first"]])):
        tr, un = sub[sub.trusted], sub[~sub.trusted]
        for col in ("run_s", "overhead_s"):
            if len(tr) and len(un):
                d = np.median(un[col]) - np.median(tr[col])
                boots = [np.median(RNG.choice(un[col].values, len(un))) -
                         np.median(RNG.choice(tr[col].values, len(tr))) for _ in range(5000)]
                lo, hi = np.quantile(boots, [.025, .975])
                out.append({"subset": label, "measure": col, "n_untrusted": len(un),
                            "n_trusted": len(tr), "median_untrusted": np.median(un[col]),
                            "median_trusted": np.median(tr[col]), "untrusted-trusted": d,
                            "ci95": f"[{lo:.1f}, {hi:.1f}]"})
    cmp_ = pd.DataFrame(out)
    ctx.table(cmp_, "c4_scrutiny_cost")

    P = plt()
    fig, ax = P.subplots(figsize=(6, 3.2))
    for k, (lab, sub) in enumerate((("untrusted", runs[~runs.trusted]), ("trusted", runs[runs.trusted]))):
        ax.scatter(sub.run_s, np.full(len(sub), k) + RNG.uniform(-.08, .08, len(sub)), s=18,
                   label=lab)
    ax.set_yticks([0, 1], ["untrusted", "trusted"])
    ax.set_xlabel("run wall time, submitted -> reaped (s)")
    ax.set_title("C4  run time by trust state (six-stage runs)")
    img = ctx.figure(fig, "c4_scrutiny")
    return "\n\n".join([
        "## C4 - Cost of scrutiny",
        (f"n = {len(runs)} six-stage runs ({int(runs.trusted.sum())} trusted). `overhead_s` is run "
        "time not spent inside a task: dispatch, absorb, QC, provenance."),
        md_table(cmp_, ".1f"), img,
    ])


if __name__ == "__main__":
    cli(analyse)
