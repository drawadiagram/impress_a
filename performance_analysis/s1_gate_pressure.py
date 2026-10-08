"""S1 - metric distributions and gate pressure: which thresholds bind, and by how much.

Every `metric_in_range` gate in the CURRENT specs is re-evaluated against each task's
recorded metric, so all five jobs are judged by one rule set (the gates changed between jobs:
roles arrived with decision 0013, packmin's bound was removed after 22726105). The as-recorded
outcomes are tabulated separately, so the two can be compared.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import Ctx, cli, md_table, plt, wilson


def range_gates(specs: dict) -> list[dict]:
    out = []
    for tool, s in specs.items():
        for g in s.qc_gates:
            if g.id == "metric_in_range":
                out.append({"tool": tool, "metric": g.params["metric"], "role": g.role,
                            "min": g.params.get("min"), "max": g.params.get("max")})
    return out


def margin(v: float, lo, hi) -> float:
    """Signed distance to the nearest bound: positive inside, negative outside."""
    d = []
    if lo is not None:
        d.append(v - lo)
    if hi is not None:
        d.append(hi - v)
    return min(d) if d else float("nan")


def analyse(ctx: Ctx) -> str:
    t = ctx.tasks
    gates = range_gates(ctx.specs)
    evals = []
    for g in gates:
        col = f"m_{g['metric']}"
        sub = t[(t.tool == g["tool"])]
        for r in sub.itertuples():
            v = getattr(r, col, np.nan)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            m = margin(float(v), g["min"], g["max"])
            evals.append({"job": r.job, "run": r.run, **g, "value": float(v), "margin": m,
                          "pass": m >= 0})
    ev = pd.DataFrame(evals)
    ctx.table(ev, "s1_gate_evaluations")

    summ = []
    for (tool, metric, role), s in ev.groupby(["tool", "metric", "role"], sort=False):
        k, n = int(s["pass"].sum()), len(s)
        lo, hi = wilson(k, n)
        misses = s[~s["pass"]]
        summ.append({"tool": tool, "metric": metric, "role": role,
                     "bound": _bound(s.iloc[0]), "n": n, "pass": k, "rate": k / n,
                     "wilson95": f"[{lo:.2f}, {hi:.2f}]",
                     "median": s.value.median(),
                     "median_miss_margin": misses.margin.median() if len(misses) else np.nan})
    summ = pd.DataFrame(summ).sort_values(["role", "rate"])
    ctx.table(summ, "s1_gate_pass_rates")

    # per design: all acceptance gates at once (six-stage designs only; the smoke job without
    # boltz cannot be judged on the two Boltz gates)
    acc = ev[ev.role == "acceptance"]
    n_acc = len({(g["tool"], g["metric"]) for g in gates if g["role"] == "acceptance"})
    per = acc.groupby(["job", "run"]).agg(n=("pass", "size"), passed=("pass", "sum"),
                                          failing=("metric", lambda m: ",".join(
                                              sorted(m[~acc.loc[m.index, "pass"]]))))
    full = per[per.n == n_acc]
    k = int((full.passed == full.n).sum())
    lo, hi = wilson(k, len(full))
    single = full[full.passed == full.n - 1].failing.value_counts()
    trusted = ctx.run_rows[ctx.run_rows.trusted]
    tk = per.reset_index().merge(trusted[["job", "run"]], on=["job", "run"])
    p_all = k / len(full)
    p_none = (1 - p_all) ** len(tk)
    ctx.table(per.reset_index(), "s1_designs")

    recorded = (ctx.gates.assign(role=ctx.gates.role.fillna("(none recorded)"))
                .groupby(["tool", "gate", "role", "outcome"]).size().unstack(fill_value=0)
                .reset_index())
    ctx.table(recorded, "s1_recorded_outcomes")

    acc_gates = [g for g in gates if g["role"] == "acceptance"]
    P = plt()
    fig, axes = P.subplots(1, len(acc_gates), figsize=(2.1 * len(acc_gates), 3), squeeze=False)
    for ax, g in zip(axes[0], acc_gates, strict=True):
        s = ev[(ev.tool == g["tool"]) & (ev.metric == g["metric"])]
        ax.scatter(np.zeros(len(s)) + np.linspace(-.15, .15, len(s)), s.value,
                   c=np.where(s["pass"], "C0", "C3"), s=14)
        for b in (g["min"], g["max"]):
            if b is not None and s.value.min() - .1 < b < s.value.max() + .1:
                ax.axhline(b, color="0.4", ls="--", lw=1)
        ax.set_xticks([])
        ax.set_title(g["metric"], fontsize=8)
    fig.suptitle("S1  acceptance metrics vs thresholds (red = miss)", fontsize=9)
    fig.tight_layout(w_pad=2.0)
    img = ctx.figure(fig, "s1_acceptance")

    return "\n\n".join([
        "## S1 - Gate pressure",
        (f"Every `metric_in_range` gate in the current specs, re-evaluated against the recorded "
        f"metric of each task (n per gate below). **{k} of {len(full)}** six-stage designs clear "
        f"all {n_acc} acceptance gates (Wilson 95% [{lo:.2f}, {hi:.2f}]). "
        f"{int((tk.passed == tk.n).sum())} of the {len(tk)} trusted designs did; at the pooled "
        f"rate, none of {len(tk)} passing has probability {p_none:.0%} - unlucky, not anomalous. "
        f"The largest acceptance miss is {-acc[~acc['pass']].margin.min():.3f} below its bound "
        "(margins in the metric's own units)."),
        md_table(summ, ".3g"), img,
        ("**Designs that missed exactly one acceptance gate**, by the gate they missed (the "
        "gates that alone cost a `pass`):"),
        md_table(single.rename_axis("gate").reset_index(name="designs")) if len(single)
        else "_none_",
        ("**As recorded at run time** (gate outcomes in the ledger; 22702568 and earlier carry no "
        "role, and 22726105's packmin bound has since been removed):"),
        md_table(recorded),
    ])


def _bound(r) -> str:
    lo, hi = (None if v is None or pd.isna(v) else v for v in (r["min"], r["max"]))
    if lo is not None and hi is not None:
        return f"[{lo:g}, {hi:g}]"
    return f">= {lo:g}" if lo is not None else f"<= {hi:g}"


if __name__ == "__main__":
    cli(analyse)
