"""C5 - did performance move between jobs (and commits)?

Jobs in run order with the commit each ran at (the manifest's `git_sha`; jobs backfilled
before 2026-10-07 have none). Per-stage medians and orchestration overhead per job. With 1-5
runs per job this can show a step change, not a trend.
"""
from __future__ import annotations

import pandas as pd

from . import data, timeline
from .common import Ctx, cli, md_table, plt


def analyse(ctx: Ctx) -> str:
    t = ctx.timed_tasks
    jobs = ctx.job_order()
    rows = []
    for j in jobs:
        tl, man = ctx.timelines[j], data.manifest(ctx.runs, j)
        ph = timeline.phases(tl)
        nrun = len(tl.runs)
        rows.append({"job": j, "launched": tl.marks.get("launch"),
                     "commit": (man.get("git_sha") or "")[:7] or "backfilled",
                     "host": _host(ctx, j), "runs": nrun,
                     "startup_s": ph.get("startup"),
                     "overhead_per_run_s": (ph["in_run_gaps"] + ph["between_runs"]) / nrun,
                     **{f"{s}_s": t[(t.job == j) & (t.tool == s)].seconds.median()
                        for s in data.STAGES}})
    df = pd.DataFrame(rows)
    ctx.table(df, "c5_drift")

    P = plt()
    fig, ax = P.subplots(figsize=(7, 3.6))
    for s in data.STAGES:
        ax.plot(range(len(df)), df[f"{s}_s"], marker="o", label=s)
    ax.set_xticks(range(len(df)), [f"{j}\n{c}" for j, c in zip(df.job, df.commit, strict=True)],
                  fontsize=7)
    ax.set_ylabel("median task seconds")
    ax.set_yscale("log")
    ax.set_title("C5  per-stage median by job, in run order")
    ax.legend(fontsize=7, frameon=False, ncol=3)
    img = ctx.figure(fig, "c5_drift")
    show = df.drop(columns=["launched"])
    return "\n\n".join([
        "## C5 - Drift across jobs",
        (f"n = {len(df)} jobs. Each job ran on a different node (`host`), so node-to-node "
        "variation is confounded with code changes."),
        md_table(show, ".3g"), img,
    ])


def _host(ctx: Ctx, job: str) -> str:
    log = data.job_dir(ctx.runs, job) / "campaign.log"
    for line in log.read_text(errors="replace").splitlines()[:20]:
        if "host=" in line:
            return line.split("host=", 1)[1].split()[0].split(".")[0]
    return ""


if __name__ == "__main__":
    cli(analyse)
