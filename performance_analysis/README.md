# Performance analysis of Delta runs

Code only. The data lives outside the repo, in the run archive at `$WORK_DIR/impress_a_runs`.
Output is written outside it as well, to `$WORK_DIR/analysis/<date>/` by default. `--out` will
not write into the checkout; the one exception is `performance_analysis/out/`, which is
gitignored.

```bash
source .venv/bin/activate
pip install -e ".[analysis]"                         # matplotlib, tmtools, scipy, gemmi, biopython
python -m performance_analysis.snapshot_sacct        # BEFORE exporting - see below
python -m performance_analysis.report                # all sections -> <out>/report.md
python -m performance_analysis.s6_seed_audit --out /tmp/x    # any one section on its own
pytest performance_analysis/tests -q                 # synthetic inputs, no run data needed
```

Defaults cover the Tier A/B jobs: 22702568, 22726105 and 22728140 (trust), plus 22684607 and
22692304 (smoke). Pass `--jobs` and `--runs` to point at others.

**`snapshot_sacct` is part of building an export, not part of analysing one.** sacct is reachable
only on the cluster and its records age out, so an archive exported without `<runs>/sacct.json`
permanently loses C3's allocation view: queue wait, elapsed, billed GPU time, GPU-busy share, CPU
efficiency and peak RSS. C3 reports those as unavailable and names the missing file.
`impress_a_runs_2026-10-08.tar.gz` was built without it, and has since been rebuilt to carry it.

### Off the cluster, from an unpacked export

`WORK_DIR` does not exist off-cluster, so both paths are explicit. Every section reproduces, C3
included, as long as `sacct.json` sits in the run root beside the job directories.

```bash
tar -xzf impress_a_runs_<date>.tar.gz -C <somewhere outside the checkout>
uv venv --python 3.10 <v> && uv pip install --python <v>/bin/python \
    pandas scipy matplotlib tmtools gemmi biopython pydantic pyyaml pytest
PYTHONPATH=src <v>/bin/python -m performance_analysis.report \
    --runs <somewhere>/impress_a_runs --out <somewhere>/analysis
```

`PYTHONPATH=src` is enough for `data.specs()`; the package itself need not be installed, which
keeps asyncflow, rhapsody and langgraph out of an analysis-only environment. Do not mix `tmtools`
into an environment holding a numpy-1.x-built matplotlib — it pulls numpy 2 and the ABI mismatch
breaks every figure. The export also carries `_trust/cuda.jsonl` as a *sibling* of
`impress_a_runs/`, outside the run root; no section reads it yet.

## Inputs, and what they cannot tell you

| Module | Reads |
|---|---|
| `data.py` | The ledger `outcome.tasks[]`: metrics, gates and roles, outputs and their sha256. Provenance `graphs.jsonl` (trust) and `results.jsonl` (verdict). Boltz confidence JSON and pLDDT npz. rfd3 model JSON (seed). LigandMPNN FASTA (seed, sequence) |
| `timeline.py` | `campaign.log`: asyncflow `task.N RUNNING/DONE`, mapped to stages by submission order. This is the **only** per-task wall time in a run. It only works for serial chains, and anything else raises `TimelineMismatch` |
| `snapshot_sacct.py` | sacct for the allocation and the `.batch` step |

- **The ledger's `cost` is an estimate.** It repeats the admission estimate and is never a
  measurement, so measured time always comes from `timeline.py`.
- **GPU utilisation is not recorded** anywhere in a run. C3 gives an upper bound instead.
- **Ledger artifact paths are absolute**, and some predate the archive's rename.
  `data.rehome` re-roots them on the job directory.
- **No substitute denominator for a missing allocation.** `manifest.json` has the campaign's own
  `started`/`finished`, but billed GPU time is `gpus x allocation elapsed`; using campaign wall
  time instead would inflate every share by however long the allocation outlived the campaign.
  C3 reports unavailable rather than guessing.

## Sections

| | Question |
|---|---|
| C1 | Per-stage wall time: spread, job-to-job differences, first-run warm-up |
| C2 | Each spec's `cost_model` against measured time; the headroom of admission and the untrusted cap |
| C3 | Where the allocation's time and hardware go: phases, GPU-busy share (upper bound), CPU efficiency, memory |
| C4 | Whether scrutiny (untrusted runs) costs wall time |
| C5 | Drift between jobs and commits |
| S1 | Gate pressure: pass rates and margins per gate under the current specs |
| S2 | Whether upstream metrics predict downstream ones (exploratory) |
| S3 | The pooled Pareto front, built with `impress_a.core.pareto` and checked against each job's recorded front |
| S4 | Boltz confidence beyond the ledger: binding-site pLDDT versus the rest |
| S5 | Diversity: TM-score (length-independent) and sequence identity; effective sample size |
| S6 | Whether the same seed reproduces the same design |
| S7 | Whether fastrelax brings every packmin pose inside its convergence gates |

All statistics are descriptive with small n: Wilson or bootstrap intervals, no p-values. Every
section states its n, and S5 shows that n overstates the number of independent samples.

## Adding a section

Write `<id>_<name>.py` with `analyse(ctx) -> str`. It should:
- read tables from `ctx` (`tasks`, `run_rows`, `gates`, `timed_tasks`, `sacct`, `specs`,
  `objectives`);
- write CSVs with `ctx.table` and figures with `ctx.figure`;
- return one markdown section that starts with `## `.

Then add the module to `report.ANALYSES`, and end the file with `cli(analyse)` so it also runs
on its own.
