# Performance analysis of Delta runs

Code only. The data lives outside the repo, in the run archive at `$WORK_DIR/impress_a_runs`.
Output is written outside it as well, to `$WORK_DIR/analysis/<date>/` by default. `--out` will
not write into the checkout; the one exception is `performance_analysis/out/`, which is
gitignored.

```bash
source .venv/bin/activate
pip install -e ".[analysis]"                         # matplotlib, tmtools, scipy, gemmi, biopython
python -m performance_analysis.snapshot_sacct        # once, on Delta: <runs>/sacct.json
python -m performance_analysis.report                # all sections -> <out>/report.md
python -m performance_analysis.s6_seed_audit --out /tmp/x    # any one section on its own
pytest performance_analysis/tests -q                 # synthetic inputs, no run data needed
```

Defaults cover the Tier A/B jobs: 22702568, 22726105 and 22728140 (trust), plus 22684607 and
22692304 (smoke). Pass `--jobs` and `--runs` to point at others, for example an unpacked export
on a laptop. Without `sacct.json` the code queries sacct live, which only works on the cluster.

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
