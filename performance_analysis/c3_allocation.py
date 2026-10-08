"""C3 - where each allocation's time and hardware go.

Phases come from `timeline.phases` (launcher and executor log lines) plus the sacct snapshot
(queue, prelaunch, epilogue). Hardware use is bounded from the timeline, because nothing in a
run records GPU utilisation: a GPU stage is counted as one GPU fully busy for its wall time,
which OVERSTATES use (model load and CPU-side work happen inside that time). The GPU share is
therefore an upper bound.
"""
from __future__ import annotations

import pandas as pd

from . import timeline
from .common import Ctx, cli, md_table, plt
from .snapshot_sacct import mem_gb, seconds, tres

PHASES = ["prelaunch", "startup", "lead_in", "tasks", "in_run_gaps", "between_runs",
          "shutdown", "epilogue"]


def analyse(ctx: Ctx) -> str:
    acct = ctx.sacct
    gpu_tools = {k for k, s in ctx.specs.items() if s.resources.gpus}
    rows, ph_rows = [], []
    for job in ctx.job_order():
        tl = ctx.timelines[job]
        a = acct.get(job, {})
        alloc, batch = a.get("alloc", {}), a.get("batch", {})
        ph = timeline.phases(tl, alloc)
        ph_rows.append({"job": job, **{p: ph.get(p, float("nan")) for p in ["queue", *PHASES]}})
        elapsed = float(alloc.get("ElapsedRaw") or "nan")
        r = tres(alloc.get("AllocTRES", ""))
        n_gpu, n_cpu = r.get("gpu", 0), r.get("cpu", 0)
        gpu_busy = tl.tasks[tl.tasks.tool.isin(gpu_tools)].seconds.sum()
        rows.append({
            "job": job, "partition": alloc.get("Partition"), "gpus": n_gpu, "cpus": n_cpu,
            "elapsed_s": elapsed, "queue_s": ph.get("queue"),
            "task_s": ph["tasks"], "task_share": ph["tasks"] / elapsed,
            "gpu_s_billed": n_gpu * elapsed, "gpu_s_busy_max": gpu_busy,
            "gpu_share_max": gpu_busy / (n_gpu * elapsed) if n_gpu else float("nan"),
            "cpu_eff": seconds(batch.get("TotalCPU", "")) / (n_cpu * elapsed) if n_cpu else
            float("nan"),
            "maxrss_gb": mem_gb(batch.get("MaxRSS", "")),
            "mem_gb": mem_gb(str(r.get("mem", ""))) if r.get("mem") else float("nan"),
        })
    use = pd.DataFrame(rows)
    phases = pd.DataFrame(ph_rows)
    ctx.table(use, "c3_utilisation")
    ctx.table(phases, "c3_phases")

    P = plt()
    fig, ax = P.subplots(figsize=(7, 3.4))
    left = pd.Series(0.0, index=phases.index)
    for i, p in enumerate(PHASES):
        v = phases[p].fillna(0).clip(lower=0)
        ax.barh(phases.job, v, left=left, label=p, color=P.cm.tab10(i))
        left += v
    ax.set_xlabel("seconds of the allocation (queue wait excluded)")
    ax.set_title("C3  allocation time by phase")
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.2))
    img = ctx.figure(fig, "c3_phases")

    trust = use[use.job.isin([j for j in ctx.jobs if ctx.timelines[j].runs.shape[0] > 1])]
    head = (f"Across the trust jobs, at most **{trust.gpu_share_max.max():.0%}** of the billed "
            f"GPU time had a GPU task running (one GPU busy, three idle, and less than that once "
            f"model load is subtracted); tasks fill {trust.task_share.median():.0%} of elapsed; "
            f"CPU efficiency is {trust.cpu_eff.median():.0%}; peak memory "
            f"{trust.maxrss_gb.max():.1f} of {trust.mem_gb.max():.0f} GB."
            if len(trust) else "")
    return "\n\n".join([
        "## C3 - Where the allocation goes",
        f"n = {len(use)} jobs; sacct from the {getattr(ctx, 'sacct_source', 'snapshot')}. "
        "GPU use is an upper bound: no run records utilisation (a gap, see the report's "
        "recommendations). " + head,
        md_table(use, ".3g"), img,
        ("**Phase seconds** (queue is wall time before the allocation started, so it is not in "
        "the bar):"), md_table(phases, ".1f"),
    ])


if __name__ == "__main__":
    cli(analyse)
