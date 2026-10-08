"""S7 - relax convergence: does fastrelax rescue every packmin pose?

Since job 22726105, packmin no longer bounds its own score: whether a pose exploded is judged
by fastrelax converging (its integrity gates `total_score <= 0` and `fa_rep <= 500`). This
checks that ruling against every pose, not the one that prompted it (22726105 r0004,
packmin +1064.5 -> fastrelax -336.0).
"""
from __future__ import annotations

import pandas as pd

from .common import Ctx, cli, md_table, plt, spearman
from .s1_gate_pressure import range_gates


def analyse(ctx: Ctx) -> str:
    t = ctx.tasks
    pk = t[t.tool == "packmin"][["job", "run", "m_total_score"]].rename(
        columns={"m_total_score": "packmin_score"})
    fr = t[t.tool == "fastrelax"][["job", "run", "m_total_score", "m_fa_rep"]].rename(
        columns={"m_total_score": "relaxed_score", "m_fa_rep": "relaxed_fa_rep"})
    df = pk.merge(fr, on=["job", "run"])
    df["relax_drop"] = df.relaxed_score - df.packmin_score
    bounds = {g["metric"]: g["max"] for g in range_gates(ctx.specs) if g["tool"] == "fastrelax"}
    df["score_margin"] = bounds.get("total_score", 0.0) - df.relaxed_score
    df["fa_rep_margin"] = bounds.get("fa_rep", 500.0) - df.relaxed_fa_rep
    ctx.table(df, "s7_relax")

    pos = df[df.packmin_score > 0]
    rho = spearman(df.packmin_score, df.relaxed_score)
    summ = pd.DataFrame([
        {"quantity": "poses", "value": len(df)},
        {"quantity": "packmin score > 0 (pre-relax)", "value": len(pos)},
        {"quantity": "of those, relaxed below 0", "value": int((pos.relaxed_score < 0).sum())},
        {"quantity": "relaxed total_score, max (gate <= 0)", "value": df.relaxed_score.max()},
        {"quantity": "relaxed fa_rep, max (gate <= 500)", "value": df.relaxed_fa_rep.max()},
        {"quantity": "smallest total_score margin", "value": df.score_margin.min()},
        {"quantity": "smallest fa_rep margin", "value": df.fa_rep_margin.min()},
        {"quantity": "Spearman packmin vs relaxed score", "value": rho},
    ])

    P = plt()
    fig, ax = P.subplots(figsize=(5.5, 3.6))
    ax.scatter(df.packmin_score, df.relaxed_score, s=16)
    worst = df.loc[df.packmin_score.idxmax()]
    ax.annotate(f"{worst.job} {worst.run}", (worst.packmin_score, worst.relaxed_score),
                fontsize=7, xytext=(-90, 10), textcoords="offset points")
    ax.axhline(bounds.get("total_score", 0.0), ls="--", color="0.5", lw=1)
    ax.set_xlabel("packmin total_score (pre-relax)")
    ax.set_ylabel("fastrelax total_score")
    ax.set_title("S7  every pose against the relax convergence gate")
    img = ctx.figure(fig, "s7_relax")
    return "\n\n".join([
        "## S7 - Relax convergence",
        f"n = {len(df)} poses. Every pose relaxed inside both integrity gates; the pre-relax "
        "score says little about the relaxed one (Spearman below)." if
        (df.score_margin >= 0).all() and (df.fa_rep_margin >= 0).all() else
        f"n = {len(df)} poses. **Some poses failed a convergence gate** - see the table.",
        md_table(summ, ".4g"), img,
    ])


if __name__ == "__main__":
    cli(analyse)
