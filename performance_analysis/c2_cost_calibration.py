"""C2 - cost-model calibration: each spec's declared cost against measured task time.

`cost_model` figures are per-invocation hours on the tool's own resource dimension (gpu_hours
for a GPU tool, cpu_hours for a CPU tool), and the specs' comments measure them the same way:
wall seconds / 3600 (e.g. fastrelax "25.9s = 0.0072 h"). They feed gate 5 and the 10%
untrusted-pattern cap, which REFUSE graphs, so the question is headroom, not accuracy.

The ledger's own `cost` is the admission estimate echoed back, not a measurement, so it is
deliberately not used here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from .common import Ctx, cli, md_table, plt

SAFETY = 3.0        # recommended = p90(measured) x SAFETY; stated, so it can be argued with
UNTRUSTED_CAP = 0.10


def analyse(ctx: Ctx) -> str:
    t = ctx.timed_tasks
    rows = []
    for tool in [s for s in data.STAGES if s in set(t.tool)]:
        spec = ctx.specs[tool]
        dim = "gpu_hours" if spec.resources.gpus else "cpu_hours"
        declared = spec.cost_model.cost.get(dim, 0.0)
        h = t[t.tool == tool].seconds.values / 3600
        p90 = float(np.quantile(h, .9))
        rows.append({"tool": tool, "dim": dim.split("_")[0], "n": len(h),
                     "declared_h": declared, "median_h": float(np.median(h)), "p90_h": p90,
                     "max_h": float(h.max()), "declared/median": declared / np.median(h),
                     "declared/max": declared / h.max(),
                     "recommended_h": round(p90 * SAFETY, 4)})
    cal = pd.DataFrame(rows)
    ctx.table(cal, "c2_cost_calibration")

    # whole chain, as admission sees it: sum of the gpu-dimension declared costs
    gpu = cal[cal.dim == "gpu"]
    chain_declared = gpu.declared_h.sum()
    per_run = (t[t.tool.isin(gpu.tool)].groupby(["job", "run"]).seconds.sum() / 3600)
    full = per_run[t.groupby(["job", "run"]).tool.nunique() == 6]
    budget = _budget(ctx)
    cap = UNTRUSTED_CAP * budget if budget else float("nan")
    chain = pd.DataFrame([
        {"quantity": "declared GPU-h per six-stage chain (current specs)",
         "value": chain_declared},
        {"quantity": "measured GPU-h per chain, median", "value": float(full.median())},
        {"quantity": "measured GPU-h per chain, max", "value": float(full.max())},
        {"quantity": "untrusted cap (10% of the trust budget)", "value": cap},
        {"quantity": "lineages admissible untrusted, declared costs", "value":
            float(np.floor(cap / chain_declared)) if chain_declared else float("nan")},
        {"quantity": "lineages admissible untrusted, at the recommended costs", "value":
            float(np.floor(cap / gpu.recommended_h.sum()))},
        {"quantity": "replicas-2 estimate (plans/next-run-replicas.md: 0.34)",
         "value": 2 * chain_declared},
        {"quantity": "recommended GPU-h per chain", "value": gpu.recommended_h.sum()},
    ])
    ctx.table(chain, "c2_chain_admission")

    P = plt()
    fig, ax = P.subplots(figsize=(7, 3.6))
    y = np.arange(len(cal))
    for i, tool in enumerate(cal.tool):
        h = t[t.tool == tool].seconds.values / 3600
        ax.scatter(h, np.full(len(h), i), s=12, color="0.35", zorder=3)
    ax.scatter(cal.declared_h, y, marker="|", s=300, color="C3", label="declared cost_model")
    ax.scatter(cal.recommended_h, y, marker="|", s=300, color="C0",
               label=f"p90 x {SAFETY:g}")
    ax.set_xscale("log")
    ax.set_yticks(y, [f"{a} ({d})" for a, d in zip(cal.tool, cal.dim, strict=True)])
    ax.set_xlabel("hours per invocation (log)")
    ax.set_title("C2  declared cost vs measured task time")
    ax.legend(frameon=False, fontsize=7)
    img = ctx.figure(fig, "c2_cost_calibration")

    tight = cal[cal["declared/max"] < 1.5]
    warn = ("**Below 1.5x the worst observed task:** " + ", ".join(tight.tool) +
            ". A single slow draw like 22684607's rfd3 would exceed the declared figure."
            if len(tight) else "Every declared cost is at least 1.5x the worst observed task.")
    return "\n\n".join([
        "## C2 - Cost-model calibration",
        (f"n = {len(t)} timed tasks. Measured = wall seconds / 3600 (one GPU, or the tool's "
        "own process for CPU tools), the same unit the spec comments use. The ledger's `cost` "
        "field is the admission estimate echoed back, so it is not evidence and is not used."),
        md_table(cal, ".3g"), img, warn,
        f"**Admission arithmetic** (trust budget {budget} GPU-h):", md_table(chain, ".3g"),
    ])


def _budget(ctx: Ctx) -> float | None:
    jd = data.job_dir(ctx.runs, ctx.jobs[0])
    for e in data._jsonl(data.campaign_dir(jd) / "provenance/campaign.jsonl"):
        if "budget" in e:
            return e["budget"].get("gpu_hours")
    return None


if __name__ == "__main__":
    cli(analyse)
