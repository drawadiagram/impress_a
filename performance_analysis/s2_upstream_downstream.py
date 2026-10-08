"""S2 - does an upstream signal predict a downstream one? (exploratory)

If an early, cheap stage predicted Boltz's verdict, a policy could stop a lineage before the
GPU-heavy prediction. Spearman correlation per pair over the six-stage designs, with a paired
bootstrap interval. With n = 16 an interval that spans zero is the expected result for all but
strong effects; read these as which pairs deserve a bigger sample, not as findings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import Ctx, bootstrap, cli, md_table, plt, spearman

PAIRS = [
    ("ss_fraction", "complex_plddt", "rfd3 secondary structure -> Boltz pLDDT"),
    ("overall_confidence", "ligand_iptm", "LigandMPNN confidence -> Boltz ligand ipTM"),
    ("ligand_confidence", "ligand_iptm", "LigandMPNN ligand confidence -> Boltz ligand ipTM"),
    ("total_score", "shape_complementarity", "relaxed score -> shape complementarity"),
    ("shape_complementarity", "ligand_iptm", "shape complementarity -> Boltz ligand ipTM"),
    ("fa_rep", "complex_plddt", "relaxed fa_rep -> Boltz pLDDT"),
    ("complex_plddt", "ligand_iptm", "Boltz pLDDT <-> ligand ipTM"),
]


def analyse(ctx: Ctx) -> str:
    r = ctx.run_rows
    rows = []
    for x, y, label in PAIRS:
        cx, cy = f"m_{x}", f"m_{y}"
        if cx not in r or cy not in r:
            continue
        s = r[[cx, cy]].dropna()
        rho = spearman(s[cx], s[cy])
        lo, hi = bootstrap(spearman, s[cx], s[cy])
        rows.append({"pair": label, "x": x, "y": y, "n": len(s), "spearman": rho,
                     "ci95_lo": lo, "ci95_hi": hi,
                     "reading": "interval excludes 0" if lo > 0 or hi < 0 else "inconclusive"})
    df = pd.DataFrame(rows)
    ctx.table(df, "s2_correlations")

    P = plt()
    n = len(df)
    cols = 4
    fig, axes = P.subplots(int(np.ceil(n / cols)), cols, figsize=(10, 2.4 * np.ceil(n / cols)),
                           squeeze=False)
    for ax, row in zip(axes.flat, df.itertuples(), strict=False):
        s = r[[f"m_{row.x}", f"m_{row.y}"]].dropna()
        ax.scatter(s.iloc[:, 0], s.iloc[:, 1], s=14)
        ax.set_xlabel(row.x, fontsize=7)
        ax.set_ylabel(row.y, fontsize=7)
        ax.set_title(f"rho={row.spearman:.2f} [{row.ci95_lo:.2f},{row.ci95_hi:.2f}]", fontsize=7)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("S2  upstream vs downstream metrics, one point per design (exploratory)",
                 fontsize=9)
    fig.tight_layout()
    img = ctx.figure(fig, "s2_correlations")
    return "\n\n".join([
        "## S2 - Upstream vs downstream (exploratory)",
        ("Spearman rho per design, paired bootstrap 95% interval. Exploratory: seven pairs "
        "tested on ~16 points, so roughly one in three 'interval excludes 0' readings could be "
        "chance at this n."),
        md_table(df, ".2f"), img,
    ])


if __name__ == "__main__":
    cli(analyse)
