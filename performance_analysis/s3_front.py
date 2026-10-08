"""S3 - objective space and the Pareto front, pooled across jobs.

The front is computed with the campaign's own `impress_a.core.pareto`, from nodes rebuilt
out of each run's recorded node metrics and QC verdict, so the analysis front and the
campaign front cannot disagree on method. Each job's recorded front (its `terminated`
transition) is reproduced first, as a check that the rebuild is faithful.
"""
from __future__ import annotations

import json

import pandas as pd

from . import data
from .common import Ctx, cli, md_table, plt


def _nodes(rows: pd.DataFrame):
    from impress_a.core.artifacts import Property, PropertySource
    from impress_a.core.qc import QCReport, QCVerdict
    from impress_a.core.tree import DesignNode
    src = PropertySource(name="analysis-rebuild")
    out = []
    for r in rows.itertuples():
        props = {c[2:]: Property(name=c[2:], value=float(getattr(r, c)), source=src)
                 for c in rows.columns if c.startswith("m_") and pd.notna(getattr(r, c))}
        out.append(DesignNode(id=f"{r.job}:{r.run}", properties=props,
                              qc=QCReport(verdict=QCVerdict(r.verdict))))
    return out


def recorded_front(ctx: Ctx, job: str) -> list[str]:
    jd = data.job_dir(ctx.runs, job)
    for e in reversed(data._jsonl(data.campaign_dir(jd) / "provenance/transitions.jsonl")):
        if e.get("event") == "terminated":
            return e.get("front", [])
    return []


def analyse(ctx: Ctx) -> str:
    from impress_a.core.pareto import Objective, pareto_front
    objs = [Objective(**o) for o in ctx.objectives]
    r = ctx.run_rows.copy()

    checks = []
    for job in ctx.jobs:
        sub = r[r.job == job]
        got = sorted(sub.set_index("run").loc[[n.id.split(":")[1] for n in
                                               pareto_front(_nodes(sub), objs)], "node"])
        want = sorted(recorded_front(ctx, job))
        checks.append({"job": job, "recorded_front": ",".join(want), "rebuilt_front":
                       ",".join(got), "match": got == want})
    checks = pd.DataFrame(checks)

    pooled = {n.id for n in pareto_front(_nodes(r), objs)}
    r["on_pooled_front"] = (r.job + ":" + r.run).isin(pooled)
    job_fronts = {n for j in ctx.jobs for n in recorded_front(ctx, j)}
    r["on_job_front"] = r.node.isin(job_fronts)
    names = [o.name for o in objs]
    cols = ["job", "run", "trusted", "verdict", "on_job_front", "on_pooled_front",
            *[f"m_{n}" for n in names]]
    ctx.table(r[cols], "s3_designs")
    ctx.table(checks, "s3_front_check")

    P = plt()
    fig, ax = P.subplots(figsize=(5.5, 4))
    for verdict, mk in (("pass", "o"), ("suspect", "s"), ("fail", "x")):
        s = r[r.verdict == verdict]
        ax.scatter(s.m_complex_plddt, s.m_ligand_iptm, marker=mk, s=30,
                   c=["C3" if f else "0.5" for f in s.on_pooled_front], label=verdict)
    for o in objs:
        if o.name == "complex_plddt" and o.min is not None:
            ax.axvline(o.min, ls="--", color="0.6", lw=1)
        if o.name == "ligand_iptm" and o.min is not None:
            ax.axhline(o.min, ls="--", color="0.6", lw=1)
    ax.set_xlabel("complex_plddt")
    ax.set_ylabel("ligand_iptm")
    ax.set_title("S3  Boltz objectives; red = pooled front, marker = QC verdict")
    ax.legend(frameon=False, fontsize=7)
    img = ctx.figure(fig, "s3_front")
    front = r[r.on_pooled_front][cols]
    return "\n\n".join([
        "## S3 - Objective space and the front",
        (f"n = {len(r)} designs, objectives {json.dumps(names)} with the campaign's constraints. "
        f"Rebuild check, each job's recorded front reproduced: "
        f"**{int(checks.match.sum())}/{len(checks)}**."),
        md_table(checks), "**Pooled front** (all jobs together):", md_table(front, ".3g"), img,
    ])


if __name__ == "__main__":
    cli(analyse)
