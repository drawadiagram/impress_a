"""Tidy tables from an archived Delta run root. Loading only - no analysis here.

The run root is `$WORK_DIR/impress_a_runs` (one `<jobid>_<campaign>/` per job; layout in the
README's "On Delta"). Nothing here writes to it. Field handling follows what the logs actually
contain, including the older ones:

* per-task metrics, QC gates and outputs come from the ledger's `outcome.tasks[]`;
* 22702568 predates `failed_gates`/`scrutiny` in provenance, so gate outcomes are always read
  from the ledger, and `trusted` from `graphs.jsonl`, which every job has;
* the ledger's `cost` is the admission ESTIMATE, not a measurement. Measured time comes from
  `timeline.py`, never from here;
* artifact paths in the ledger are absolute and were written before the archive was
  reorganised, so they are re-rooted on the job directory by their `work/...` suffix.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

TIER_A = ("22702568", "22726105", "22728140")
TIER_B = ("22684607", "22692304")
TIER_AB = TIER_A + TIER_B
STAGES = ("rfd3_design", "ligandmpnn_design", "packmin", "fastrelax", "filter_shape",
          "boltz_predict")


def default_runs() -> Path:
    if env := os.environ.get("IMPRESS_A_RUNS"):
        return Path(env)
    work = os.environ.get("WORK_DIR")
    if not work:
        raise SystemExit("set --runs, IMPRESS_A_RUNS or WORK_DIR")
    return Path(work) / "impress_a_runs"


def job_dir(runs: Path, job: str) -> Path:
    hits = sorted(runs.glob(f"{job}_*"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one directory for job {job} under {runs}, got {hits}")
    return hits[0]


def campaign_dir(jd: Path) -> Path:
    return next(p for p in jd.iterdir() if p.is_dir() and p.name.endswith("-D"))


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] \
        if p.exists() else []


def rehome(jd: Path, path: str | None) -> Path | None:
    """An absolute ledger path, re-rooted on the job directory it now lives in."""
    if not path:
        return None
    if "/work/" not in path:
        return Path(path)
    return jd / "work" / path.rsplit("/work/", 1)[1]


def objectives(jd: Path) -> list[dict]:
    for e in _jsonl(campaign_dir(jd) / "provenance/campaign.jsonl"):
        if "objectives" in e:
            return e["objectives"]
    return []


def runs_table(runs: Path, jobs=TIER_AB) -> pd.DataFrame:
    """One row per admitted run: trust, scrutiny, estimate, node verdict and metrics."""
    rows = []
    for job in jobs:
        jd = job_dir(runs, job)
        prov = campaign_dir(jd) / "provenance"
        graphs = {e["run"]: e for e in _jsonl(prov / "graphs.jsonl") if "run" in e}
        results = {e["run"]: e for e in _jsonl(prov / "results.jsonl") if "run" in e}
        for e in _jsonl(campaign_dir(jd) / "jobs/ledger.jsonl"):
            o = e.get("outcome")
            if not o:
                continue
            g, r = graphs.get(o["run_id"], {}), results.get(o["run_id"], {})
            rows.append({
                "job": job, "campaign": jd.name.split("_", 1)[1], "run": o["run_id"],
                "cycle": g.get("cycle"), "signature": o.get("signature"),
                "trusted": bool(g.get("trusted", False)), "scrutiny": g.get("scrutiny"),
                "est_gpu_h": (g.get("estimate") or {}).get("gpu_hours"),
                "est_cpu_h": (g.get("estimate") or {}).get("cpu_hours"),
                "state": o.get("state"), "n_tasks": len(o.get("tasks", [])),
                "node": (o.get("nodes") or [None])[0],
                "verdict": r.get("qc") or (o.get("qc") or {}).get("verdict"),
                **{f"m_{k}": v for k, v in (o.get("metrics") or {}).items()},
            })
    return pd.DataFrame(rows)


def tasks_table(runs: Path, jobs=TIER_AB) -> pd.DataFrame:
    """One row per task: tool, stage, QC, failed gates (with roles), metrics, outputs."""
    rows = []
    for job in jobs:
        jd = job_dir(runs, job)
        for e in _jsonl(campaign_dir(jd) / "jobs/ledger.jsonl"):
            o = e.get("outcome")
            if not o:
                continue
            for t in o.get("tasks", []):
                lineage, stage = t["task_id"].split("_")[:2]
                gates = (t.get("qc") or {}).get("gates", [])
                outs = {k: v for k, v in (t.get("outputs") or {}).items() if isinstance(v, dict)}
                rows.append({
                    "job": job, "run": o["run_id"], "lineage": int(lineage.lstrip("r")),
                    "stage": int(stage.lstrip("s")), "task_id": t["task_id"], "tool": t["tool"],
                    "qc": (t.get("qc") or {}).get("verdict"),
                    "gates": gates,
                    "failed": [g for g in gates if g.get("outcome") == "fail"],
                    "est_gpu_h": (t.get("cost") or {}).get("gpu_hours", 0.0),
                    "est_cpu_h": (t.get("cost") or {}).get("cpu_hours", 0.0),
                    "outputs": {k: rehome(jd, v.get("path")) for k, v in outs.items()},
                    "sha256": {k: v.get("sha256") for k, v in outs.items()},
                    **{f"m_{k}": v for k, v in (t.get("metrics") or {}).items()},
                })
    return pd.DataFrame(rows)


def gates_table(tasks: pd.DataFrame) -> pd.DataFrame:
    """One row per gate evaluation, long form. Older ledgers carry no role: left missing."""
    rows = []
    for t in tasks.itertuples():
        for g in t.gates:
            rows.append({"job": t.job, "run": t.run, "tool": t.tool, "gate": g["gate"],
                         "role": g.get("role"), "outcome": g.get("outcome"),
                         "observed": g.get("observed"), "threshold": g.get("threshold")})
    return pd.DataFrame(rows)


def specs() -> dict:
    """The registered real-tool specs: resources, cost model and gate roles, from the specs."""
    from impress_a.tools.registry import Registry
    reg = Registry().load()
    return {k: v for k, v in reg.specs.items() if not k.startswith("mock_")}


def sacct(runs: Path) -> dict:
    """The snapshot written by snapshot_sacct.py; empty if it was never taken."""
    p = runs / "sacct.json"
    return json.loads(p.read_text()) if p.exists() else {}


def manifest(runs: Path, job: str) -> dict:
    p = job_dir(runs, job) / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


# -- artifact readers ------------------------------------------------------------------------

def boltz_confidence(tasks: pd.DataFrame) -> pd.DataFrame:
    """Every field of Boltz's confidence JSON, which the ledger keeps only two of."""
    rows = []
    for t in tasks[tasks.tool == "boltz_predict"].itertuples():
        cx = t.outputs.get("complex")
        if cx is None:
            continue
        hits = sorted(cx.parent.glob("confidence_*_model_0.json"))
        if not hits:
            continue
        c = json.loads(hits[0].read_text())
        pair = c.get("pair_chains_iptm", {})
        rows.append({"job": t.job, "run": t.run,
                     **{k: v for k, v in c.items() if isinstance(v, (int, float))},
                     "chain_A_ptm": c.get("chains_ptm", {}).get("0"),
                     "chain_B_ptm": c.get("chains_ptm", {}).get("1"),
                     "iptm_A_B": pair.get("0", {}).get("1"),
                     "iptm_B_A": pair.get("1", {}).get("0"),
                     "plddt_npz": next(iter(sorted(cx.parent.glob("plddt_*_model_0.npz"))), None),
                     "pdb": cx if cx.suffix == ".pdb" else
                            next(iter(sorted(cx.parent.glob("*_model_0.pdb"))), None)})
    return pd.DataFrame(rows)


def rfd3_records(tasks: pd.DataFrame) -> pd.DataFrame:
    """Seed and backbone per rfd3 task, from the model JSON written next to the design."""
    rows = []
    for t in tasks[tasks.tool == "rfd3_design"].itertuples():
        bb = t.outputs.get("backbone")
        meta = sorted(bb.parent.glob("*_model_0.json")) if bb else []
        seed = json.loads(meta[0].read_text()).get("seed") if meta else None
        rows.append({"job": t.job, "run": t.run, "seed": seed, "backbone": bb,
                     "sha256": t.sha256.get("backbone")})
    return pd.DataFrame(rows)


def mpnn_records(tasks: pd.DataFrame) -> pd.DataFrame:
    """Seed and designed sequence per LigandMPNN task. The FASTA's second record is the design
    (the first is the input backbone's own sequence)."""
    rows = []
    for t in tasks[tasks.tool == "ligandmpnn_design"].itertuples():
        jd = next(iter(t.outputs.values()), None)
        work = _workdir(jd, "ligandmpnn")
        fas = sorted((work / "seqs").glob("*.fa")) if work else []
        if not fas:
            continue
        recs = _fasta(fas[0])
        if len(recs) < 2:
            continue
        hdr, seq = recs[1]
        fields = dict(f.split("=", 1) for f in (s.strip() for s in hdr.split(",")) if "=" in f)
        rows.append({"job": t.job, "run": t.run, "seed": int(fields["seed"]) if "seed" in fields
                     else None, "native": recs[0][1], "sequence": seq,
                     "sha256": next(iter(t.sha256.values()), None)})
    return pd.DataFrame(rows)


def _workdir(path: Path | None, prefix: str) -> Path | None:
    if path is None:
        return None
    for p in (path, *path.parents):
        if p.name.startswith(prefix + "_"):
            return p
    return None


def _fasta(p: Path) -> list[tuple[str, str]]:
    out, hdr, seq = [], None, []
    for line in p.read_text().splitlines():
        if line.startswith(">"):
            if hdr is not None:
                out.append((hdr, "".join(seq)))
            hdr, seq = line[1:], []
        elif line.strip():
            seq.append(line.strip())
    if hdr is not None:
        out.append((hdr, "".join(seq)))
    return out
