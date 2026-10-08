"""S5 - diversity: are the designs independent samples, or variations of one?

rfd3 runs in *partial* diffusion from a starting scaffold (its output files are named
`*_binder_design_partial_*`), so designs can be expected to share a fold. This measures how
much. Designs differ in length (rfd3 samples it), so structure is compared by TM-score from
TM-align (`tmtools`), which is length-independent, normalised by the LONGER chain so a small
design inside a big one does not score as identical; TM >= 0.5 is the usual "same fold" line.
Calpha RMSD after Kabsch superposition is added only where lengths match. Sequences: pairwise
identity (position by position at equal length, else a global alignment). Clusters at stated
cutoffs give an effective sample size for the intervals in the other analyses.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import data
from .common import Ctx, cli, md_table, plt

TM_CUTOFFS = (0.9, 0.7, 0.5)
IDENTITY_CUTOFFS = (0.9, 0.7, 0.5)


def ca_coords(pdb) -> np.ndarray:
    import gemmi
    st = gemmi.read_structure(str(pdb))
    return np.array([[a.pos.x, a.pos.y, a.pos.z] for res in st[0]["A"] for a in res
                     if a.name == "CA"])


def tm_score(a: np.ndarray, b: np.ndarray) -> float:
    """TM-score normalised by the longer chain (the conservative of TM-align's two)."""
    from tmtools import tm_align
    r = tm_align(a, b, "A" * len(a), "A" * len(b))
    return float(min(r.tm_norm_chain1, r.tm_norm_chain2))


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("nan")
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(1).mean()))


def identity(s1: str, s2: str) -> float:
    if len(s1) == len(s2):
        return sum(x == y for x, y in zip(s1, s2, strict=True)) / len(s1)
    from Bio import Align
    al = Align.PairwiseAligner(mode="global")
    a = al.align(s1, s2)[0]
    matches = sum(x == y for x, y in zip(a[0], a[1], strict=True) if x != "-" and y != "-")
    return matches / min(len(s1), len(s2))


def clusters(names, dist, cutoff) -> int:
    """Single-linkage cluster count: pairs within `cutoff` join."""
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for (i, j), d in dist.items():
        if d <= cutoff:
            parent[find(i)] = find(j)
    return len({find(n) for n in names})


def pairwise(ctx: Ctx):
    rf = data.rfd3_records(ctx.tasks)
    mp = data.mpnn_records(ctx.tasks)
    rf["key"] = rf.job + ":" + rf.run
    mp["key"] = mp.job + ":" + mp.run
    xyz = {r.key: ca_coords(r.backbone) for r in rf.itertuples()}
    seq = dict(zip(mp.key, mp.sequence, strict=True))
    rows = []
    for a, b in itertools.combinations(sorted(xyz), 2):
        rows.append({"a": a, "b": b, "len_a": len(xyz[a]), "len_b": len(xyz[b]),
                     "tm_score": tm_score(xyz[a], xyz[b]),
                     "ca_rmsd_same_length": kabsch_rmsd(xyz[a], xyz[b]),
                     "seq_identity": identity(seq[a], seq[b]) if a in seq and b in seq
                     else np.nan})
    return pd.DataFrame(rows), rf, mp


def analyse(ctx: Ctx) -> str:
    pw, _, mp = pairwise(ctx)
    ctx.table(pw, "s5_pairwise")
    names = sorted(set(pw.a) | set(pw.b))
    tmd = {(r.a, r.b): 1 - r.tm_score for r in pw.itertuples()}
    ident = {(r.a, r.b): 1 - r.seq_identity for r in pw.itertuples() if np.isfinite(r.seq_identity)}
    cl = [{"measure": "backbone TM-score", "cutoff": f">= {c:g}",
           "clusters": clusters(names, tmd, 1 - c), "designs": len(names)} for c in TM_CUTOFFS]
    cl += [{"measure": "sequence identity", "cutoff": f">= {c:.0%}",
            "clusters": clusters(names, ident, 1 - c), "designs": len(names)}
           for c in IDENTITY_CUTOFFS]
    cl = pd.DataFrame(cl)
    ctx.table(cl, "s5_clusters")
    native_id = [identity(r.native, r.sequence) for r in mp.itertuples()]
    summ = pd.DataFrame([
        {"quantity": "pairwise backbone TM-score", "median": pw.tm_score.median(),
         "min": pw.tm_score.min(), "max": pw.tm_score.max()},
        {"quantity": "distinct backbone lengths", "median": float(len(set(pw.len_a) | set(pw.len_b))),
         "min": float(min(pw.len_a.min(), pw.len_b.min())),
         "max": float(max(pw.len_a.max(), pw.len_b.max()))},
        {"quantity": "pairwise designed-sequence identity", "median": pw.seq_identity.median(),
         "min": pw.seq_identity.min(), "max": pw.seq_identity.max()},
        {"quantity": "design vs the sequence on its rfd3 backbone", "median": np.median(native_id),
         "min": min(native_id), "max": max(native_id)},
    ])

    P = plt()
    fig, axes = P.subplots(1, 2, figsize=(8, 3))
    axes[0].hist(pw.tm_score, bins=20)
    axes[0].axvline(0.5, ls="--", color="0.5", lw=1)
    axes[0].set_xlabel("pairwise backbone TM-score")
    axes[1].hist(pw.seq_identity.dropna(), bins=20)
    axes[1].set_xlabel("pairwise designed-sequence identity")
    fig.suptitle(f"S5  diversity across {len(names)} designs ({len(pw)} pairs)", fontsize=9)
    fig.tight_layout()
    img = ctx.figure(fig, "s5_diversity")
    return "\n\n".join([
        "## S5 - Diversity",
        (f"n = {len(names)} designs, {len(pw)} pairs. Backbones are rfd3's `backbone_0.pdb` "
        "(chain A); sequences are LigandMPNN's design (the second FASTA record)."),
        md_table(summ, ".3g"), img,
        "**Single-linkage clusters** (an effective sample size for the other sections):",
        md_table(cl),
    ])


if __name__ == "__main__":
    cli(analyse)
