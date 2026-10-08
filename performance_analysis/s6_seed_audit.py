"""S6 - seed reproducibility: does the same seed give the same design?

`compose/composer._replica_seed(run_label, lineage)` derives every seeded tool's seed from
the run label, and run labels restart at r0001 in each job, so the same label in two jobs is
the same seed with the same inputs (one spec, one scaffold, one ligand). rfd3 is the first
stage, so for it the inputs are identical too: a different output under the same seed is
nondeterminism or an unseeded random source, not different inputs.

Recorded seeds: rfd3's model JSON, LigandMPNN's FASTA header. Two things are checked
separately, because the seed turns out to fix one and not the other: the design's LENGTH
(which rfd3 samples) and its COORDINATES (byte identity; TM-score and same-length Calpha RMSD
against the S5 background of different-seed pairs). Different seeds also differ in length, so
the TM-score comparison is confounded with length and is reported, not over-read.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import data
from .common import Ctx, cli, md_table, plt
from .s5_diversity import pairwise


def analyse(ctx: Ctx) -> str:
    rf = data.rfd3_records(ctx.tasks)
    mp = data.mpnn_records(ctx.tasks)
    seeds = rf[["job", "run", "seed", "sha256"]].rename(columns={"seed": "rfd3_seed",
                                                                  "sha256": "backbone_sha"}) \
        .merge(mp[["job", "run", "seed", "sequence"]].rename(columns={"seed": "mpnn_seed"}),
               on=["job", "run"], how="left")
    seeds["same_seed_both_tools"] = seeds.rfd3_seed == seeds.mpnn_seed
    ctx.table(seeds.drop(columns=["sequence"]), "s6_seeds")

    pw, _, _ = pairwise(ctx)
    key = {f"{r.job}:{r.run}": r for r in seeds.itertuples()}
    pairs = []
    for g, s in seeds.groupby("rfd3_seed"):
        for a, b in itertools.combinations(s.itertuples(), 2):
            ka, kb = sorted((f"{a.job}:{a.run}", f"{b.job}:{b.run}"))
            m = pw[(pw.a == ka) & (pw.b == kb)]
            pairs.append({"seed": g, "a": ka, "b": kb,
                          "same_backbone_bytes": a.backbone_sha == b.backbone_sha,
                          "same_sequence": key[ka].sequence == key[kb].sequence,
                          "same_length": len(a.sequence) == len(b.sequence),
                          "tm_score": float(m.tm_score.iloc[0]) if len(m) else np.nan,
                          "ca_rmsd": float(m.ca_rmsd_same_length.iloc[0]) if len(m) else np.nan,
                          "seq_identity": float(m.seq_identity.iloc[0]) if len(m) else np.nan})
    pairs = pd.DataFrame(pairs)

    same = set(zip(pairs.a, pairs.b, strict=True)) if len(pairs) else set()
    bg = pw[[(a, b) not in same for a, b in zip(pw.a, pw.b, strict=True)]]
    cmp_ = pd.DataFrame([
        {"pairs": "same seed", "n": len(pairs),
         "same_length": float(pairs.same_length.mean()) if len(pairs) else np.nan,
         "median_tm_score": pairs.tm_score.median(),
         "median_seq_identity": pairs.seq_identity.median()},
        {"pairs": "different seed", "n": len(bg),
         "same_length": float((bg.len_a == bg.len_b).mean()),
         "median_tm_score": bg.tm_score.median(),
         "median_seq_identity": bg.seq_identity.median()},
    ])
    if len(pairs):    # share of different-seed pairs at least as similar: 0 = none closer
        pairs["tm_rank_vs_different_seed"] = [float((bg.tm_score >= v).mean())
                                              for v in pairs.tm_score]
    ctx.table(pairs, "s6_same_seed_pairs")

    P = plt()
    fig, ax = P.subplots(figsize=(6, 3))
    ax.hist(bg.tm_score, bins=20, color="0.7", label="different seed")
    for r in pairs.tm_score.dropna():
        ax.axvline(r, color="C3", lw=1, alpha=.7)
    ax.plot([], [], color="C3", label="same seed")
    ax.set_xlabel("backbone TM-score")
    ax.set_title("S6  same-seed pairs against the background of all pairs")
    ax.legend(frameon=False, fontsize=7)
    img = ctx.figure(fig, "s6_seed_audit")

    n_ident = int(pairs.same_backbone_bytes.sum()) if len(pairs) else 0
    verdict = (f"**{n_ident} of {len(pairs)}** same-seed pairs produced byte-identical "
               "backbones. ")
    if len(pairs):
        verdict += (f"Same length in **{int(pairs.same_length.sum())} of {len(pairs)}** "
                    f"same-seed pairs, against {(bg.len_a == bg.len_b).mean():.0%} of "
                    "different-seed pairs: the seed fixes the sampled length. "
                    f"Median TM-score {pairs.tm_score.median():.2f} for same-seed pairs vs "
                    f"{bg.tm_score.median():.2f} for different-seed pairs; same-length Calpha "
                    f"RMSD {pairs.ca_rmsd.median():.2f} A (median). So the seed decides the "
                    "length and biases the fold, but the coordinates are not reproducible: "
                    "rerunning a run label does not reproduce its design.")
    return "\n\n".join([
        "## S6 - Seed reproducibility audit",
        f"n = {len(seeds)} designs; {seeds.rfd3_seed.nunique()} distinct rfd3 seeds; "
        f"{len(pairs)} same-seed pairs across jobs. rfd3 and LigandMPNN received the same seed "
        f"in {int(seeds.same_seed_both_tools.sum())}/{len(seeds)} designs. " + verdict,
        md_table(cmp_, ".3g"), "**Same-seed pairs:**",
        md_table(pairs, ".3g") if len(pairs) else "_none_", img,
    ])


if __name__ == "__main__":
    cli(analyse)
