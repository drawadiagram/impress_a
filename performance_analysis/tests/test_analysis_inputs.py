"""The parsers and statistics the analyses stand on, against synthetic runs built here.

No run data is committed: each test writes the few log lines it needs into tmp_path, in the
exact formats the real logs use (copied from job 22728140's campaign.log and ledger).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from performance_analysis import common, data, timeline
from performance_analysis.s1_gate_pressure import margin
from performance_analysis.snapshot_sacct import mem_gb, seconds, tres

STAGES = {"r0_s0_rfd3_design": "rfd3_design", "r0_s1_ligandmpnn_design": "ligandmpnn_design"}
INFO = " INFO    "


def _line(ts: str, logger: str, msg: str) -> str:
    return f"2026-10-07 {ts}{INFO}{logger:<22} {msg}"


def _log(runs: dict[str, list[tuple[str, str]]], extra_task: bool = False) -> str:
    """runs: run -> [(start, end) per stage]; times as HH:MM:SS,mmm."""
    out = [_line("13:10:15,886", "impress_a.launch", "host=gpua011.delta pid=1 python=3.12"),
           _line("13:10:19,874", "impress_a.runtime.executor", "engine: dragon backend ready")]
    n = 0
    for run, spans in runs.items():
        out.append(_line(spans[0][0], "impress_a.runtime.executor",
                         f"submitted {run}: {STAGES!r} (trusted=False, estimate={{'gpu_hours': 0.1}})"))
        for s, e in spans + ([spans[-1]] if extra_task else []):
            n += 1
            out.append(_line(s, "radical.asyncflow.workflow_manager", f"task.{n:06d} is in RUNNING state"))
            out.append(_line(e, "radical.asyncflow.workflow_manager", f"task.{n:06d} is in DONE state"))
        out.append(_line(spans[-1][1], "impress_a.runtime.executor",
                         f"reaped {run}: 2/2 tasks ok, cost={{'gpu_hours': 0.1}}"))
    out.append(_line("13:20:00,000", "impress_a.runtime.executor", "stop requested: max_cycles=1"))
    out.append(_line("13:20:02,000", "impress_a.cli", "campaign returned after 9m46s"))
    return "\n".join(out) + "\n"


def test_tasks_map_to_stages_in_submission_order(tmp_path):
    log = tmp_path / "campaign.log"
    log.write_text(_log({"r0001": [("13:10:20,000", "13:11:05,000"), ("13:11:06,000", "13:11:16,000")],
                         "r0002": [("13:12:00,000", "13:12:40,000"), ("13:12:41,000", "13:12:47,500")]}))
    tl = timeline.parse(log)
    assert list(tl.tasks.tool) == ["rfd3_design", "ligandmpnn_design"] * 2
    assert list(tl.tasks.seconds) == [45.0, 10.0, 40.0, 6.5]
    ph = timeline.phases(tl)
    assert ph["tasks"] == 101.5
    assert ph["startup"] == pytest.approx(3.988)
    assert ph["in_run_gaps"] == pytest.approx(2.0)      # 1 s between stages, in each run
    assert ph["shutdown"] == pytest.approx(2.0 + (20 * 60 - 12 * 60 - 47.5))


def test_a_log_that_does_not_match_the_stage_count_is_refused(tmp_path):
    log = tmp_path / "campaign.log"
    log.write_text(_log({"r0001": [("13:10:20,000", "13:11:05,000"),
                                   ("13:11:06,000", "13:11:16,000")]}, extra_task=True))
    with pytest.raises(timeline.TimelineMismatch):
        timeline.parse(log)


def test_phases_use_sacct_for_queue_and_epilogue(tmp_path):
    log = tmp_path / "campaign.log"
    log.write_text(_log({"r0001": [("13:10:20,000", "13:11:05,000"),
                                   ("13:11:06,000", "13:11:16,000")]}))
    ph = timeline.phases(timeline.parse(log), {"Submit": "2026-10-07T13:06:57",
                                               "Start": "2026-10-07T13:10:01",
                                               "End": "2026-10-07T13:20:05"})
    assert ph["queue"] == 184.0
    assert ph["prelaunch"] == pytest.approx(14.886)
    assert ph["epilogue"] == pytest.approx(3.0)


def _job(tmp_path: Path, job="22728140") -> Path:
    jd = tmp_path / f"{job}_delta-small-molecule-trust"
    cd = jd / "delta-small-molecule-trust-D"
    (cd / "jobs").mkdir(parents=True)
    (cd / "provenance").mkdir()
    old = f"/work/nvme/x/impress_a_runs/{job}/work/rfd3_r0001_r0_s0_rfd3_design/backbone_0.pdb"
    task = {"task_id": "r0_s0_rfd3_design", "tool": "rfd3_design",
            "metrics": {"ss_fraction": 0.857},
            "qc": {"verdict": "pass", "gates": [
                {"gate": "has_secondary_structure", "outcome": "pass", "observed": 0.857,
                 "role": "integrity"},
                {"gate": "metric_in_range[x]", "outcome": "fail", "observed": 0.1}]},
            "cost": {"gpu_hours": 0.1},
            "outputs": {"backbone": {"type": "Backbone", "path": old, "sha256": "ab"}}}
    ledger = [{"event": "campaign_started"},
              {"job_id": "r0001", "status": "submitted", "cycle": 1},
              {"job_id": "r0001", "status": "done", "outcome": {
                  "run_id": "r0001", "signature": "f5", "state": "done", "nodes": ["n1"],
                  "metrics": {"ss_fraction": 0.857}, "qc": {"verdict": "pass"},
                  "tasks": [task]}}]
    (cd / "jobs/ledger.jsonl").write_text("\n".join(json.dumps(e) for e in ledger) + "\n")
    (cd / "provenance/graphs.jsonl").write_text(json.dumps(
        {"run": "r0001", "cycle": 0, "trusted": True, "estimate": {"gpu_hours": 0.17}}) + "\n")
    (cd / "provenance/results.jsonl").write_text(json.dumps({"run": "r0001", "qc": "suspect"}) + "\n")
    return jd


def test_ledger_tasks_are_flattened_and_paths_rehomed(tmp_path):
    jd = _job(tmp_path)
    t = data.tasks_table(tmp_path, ["22728140"])
    assert len(t) == 1
    row = t.iloc[0]
    assert (row.lineage, row.stage, row.m_ss_fraction) == (0, 0, 0.857)
    # the ledger's absolute path predates the archive's rename; it is re-rooted on the job dir
    assert row.outputs["backbone"] == jd / "work/rfd3_r0001_r0_s0_rfd3_design/backbone_0.pdb"
    g = data.gates_table(t)
    assert g.role.isna().tolist() == [False, True]   # an older ledger carries no role: missing
    r = data.runs_table(tmp_path, ["22728140"]).iloc[0]
    assert (bool(r.trusted), r.verdict, r.est_gpu_h) == (True, "suspect", 0.17)


def test_fasta_reads_multiline_records(tmp_path):
    fa = tmp_path / "x.fa"
    fa.write_text(">backbone_0, T=0.2, seed=955947389\nAAA\nCC\n>backbone_0, id=1, seed=955947389\nGGG\n")
    assert data._fasta(fa) == [("backbone_0, T=0.2, seed=955947389", "AAACC"),
                               ("backbone_0, id=1, seed=955947389", "GGG")]


def test_small_n_statistics():
    lo, hi = common.wilson(5, 16)
    assert (round(lo, 2), round(hi, 2)) == (0.14, 0.56)
    assert common.wilson(0, 0) != common.wilson(0, 0)        # nan, not a crash
    assert margin(0.48, 0.5, 1.0) == pytest.approx(-0.02)
    assert margin(-336.0, None, 0.0) == 336.0


def test_sacct_field_parsers():
    assert seconds("06:05:12") == 6 * 3600 + 5 * 60 + 12
    assert seconds("1-00:00:01") == 86401
    assert seconds("37:45.123") == pytest.approx(37 * 60 + 45.123)
    assert mem_gb("16918112K") == pytest.approx(16.13, abs=0.01)
    assert tres("billing=8000,cpu=64,gres/gpu=4,mem=240G,node=1")["gpu"] == 4


def test_output_into_the_checkout_is_refused(tmp_path):
    repo = Path(common.__file__).resolve().parent.parent
    a = common.parser().parse_args(["--runs", str(tmp_path), "--out", str(repo / "results")])
    with pytest.raises(SystemExit):
        common.make_ctx(a)


def test_c3_says_sacct_is_unavailable_rather_than_reporting_nan(tmp_path, monkeypatch):
    """An export built without snapshot_sacct.py used to render "at most nan% of the billed GPU
    time", because the live query's empty result was still labelled "live sacct"."""
    from performance_analysis import c3_allocation, snapshot_sacct

    monkeypatch.setattr(snapshot_sacct, "query", lambda jobs: {})
    _job(tmp_path)
    log = tmp_path / "22728140_delta-small-molecule-trust" / "campaign.log"
    log.write_text(_log({"r0001": [("13:10:20,000", "13:11:05,000"),
                                   ("13:11:06,000", "13:11:16,000")]}))
    ctx = common.Ctx(tmp_path, tmp_path / "out", ["22728140"])
    assert ctx.sacct == {}
    assert ctx.sacct_source == "unavailable"

    md = c3_allocation.analyse(ctx)
    assert "sacct is unavailable" in md
    assert "nan" not in md.lower()
    # the log-derived phases survive; the allocation columns are not shown as empty cells
    assert "tasks" in md
    for col in c3_allocation.FROM_SACCT:
        assert f"| {col} |" not in md
