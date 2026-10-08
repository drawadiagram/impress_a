"""S4 - Boltz confidence beyond the two numbers the ledger keeps.

From each prediction's confidence JSON: ptm, iptm, complex_iplddt, complex_pde/ipde and the
chain-pair ipTMs. From the per-token pLDDT npz: mean confidence of the binding-site residues
(chain A residues with a heavy atom within 5 A of the ligand, in Boltz's own predicted pose),
of the rest of the protein, and of the ligand tokens. The question: when ligand ipTM is low,
is the whole fold uncertain, or only the site?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import Ctx, bootstrap, cli, md_table, plt, spearman

CUTOFF = 5.0


def site_plddt(pdb, npz) -> dict:
    import gemmi
    st = gemmi.read_structure(str(pdb))
    st.setup_entities()
    model = st[0]
    prot = next(ch for ch in model if ch.name == "A")
    lig_atoms = [a.pos for ch in model if ch.name != "A" for res in ch for a in res]
    site = set()
    for i, res in enumerate(prot):
        if any(a.pos.dist(p) <= CUTOFF for a in res for p in lig_atoms):
            site.add(i)
    pl = np.load(npz)["plddt"]
    n = len(prot)
    idx = np.arange(n)
    in_site = np.isin(idx, list(site))
    return {"n_res": n, "n_site": len(site),
            "plddt_site": float(pl[:n][in_site].mean()) if site else np.nan,
            "plddt_rest": float(pl[:n][~in_site].mean()),
            "plddt_ligand": float(pl[n:].mean()) if len(pl) > n else np.nan}


def analyse(ctx: Ctx) -> str:
    from . import data
    b = data.boltz_confidence(ctx.tasks)
    extra = pd.DataFrame([site_plddt(r.pdb, r.plddt_npz) for r in b.itertuples()])
    b = pd.concat([b.drop(columns=["plddt_npz", "pdb"]), extra], axis=1)
    b["site_minus_rest"] = b.plddt_site - b.plddt_rest
    ctx.table(b, "s4_boltz_confidence")

    cols = ["ptm", "iptm", "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
            "chain_A_ptm", "plddt_site", "plddt_rest", "plddt_ligand", "site_minus_rest"]
    summ = b[cols].describe().T[["count", "mean", "min", "50%", "max"]].reset_index() \
        .rename(columns={"index": "field", "50%": "median"})
    corr = []
    for c in ["chain_A_ptm", "plddt_rest", "plddt_site", "plddt_ligand", "complex_ipde"]:
        rho = spearman(b[c], b.ligand_iptm)
        lo, hi = bootstrap(spearman, b[c], b.ligand_iptm)
        corr.append({"field": c, "spearman_vs_ligand_iptm": rho, "ci95_lo": lo, "ci95_hi": hi})
    corr = pd.DataFrame(corr)
    ctx.table(corr, "s4_vs_ligand_iptm")

    P = plt()
    fig, axes = P.subplots(1, 2, figsize=(8, 3.2))
    axes[0].scatter(b.plddt_rest, b.ligand_iptm, s=14, label="rest of protein")
    axes[0].scatter(b.plddt_site, b.ligand_iptm, s=14, label=f"site (<= {CUTOFF:g} A)")
    axes[0].set_xlabel("mean pLDDT")
    axes[0].set_ylabel("ligand_iptm")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].scatter(b.chain_A_ptm, b.ligand_iptm, s=14)
    axes[1].set_xlabel("protein chain pTM")
    axes[1].set_ylabel("ligand_iptm")
    fig.suptitle("S4  is a low ligand ipTM a global or a local failure?", fontsize=9)
    fig.tight_layout()
    img = ctx.figure(fig, "s4_boltz")
    lead = (f"Binding-site residues average {b.plddt_site.median():.2f} pLDDT against "
            f"{b.plddt_rest.median():.2f} for the rest of the protein (medians), and the ligand "
            f"tokens {b.plddt_ligand.median():.2f}.")
    return "\n\n".join([
        "## S4 - Boltz confidence beyond the ledger",
        f"n = {len(b)} predictions. {lead}", md_table(summ, ".3g"),
        "**Against ligand ipTM** (Spearman, bootstrap 95%):", md_table(corr, ".2f"), img,
    ])


if __name__ == "__main__":
    cli(analyse)
