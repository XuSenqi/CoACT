"""Tests for official SWE-bench harness integration helpers."""

import json
from pathlib import Path

from src.eval.swebench_harness import run_swebench_harness


class TestSwebenchHarness:
    """Tests for harness wrapper."""

    def test_run_swebench_harness_saves_predictions_and_parses_report(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Wrapper should write predictions and parse the harness report."""
        report_path = tmp_path.parent / "fake-report.json"

        def fake_main(**kwargs):
            with open(report_path, "w") as f:
                json.dump(
                    {
                        "total_instances": 4,
                        "submitted_instances": 4,
                        "completed_instances": 4,
                        "resolved_instances": 3,
                        "unresolved_instances": 1,
                        "empty_patch_instances": 0,
                        "error_instances": 0,
                    },
                    f,
                    indent=2,
                )
            return report_path

        monkeypatch.setattr("src.eval.swebench_harness.harness_main", fake_main)

        result = run_swebench_harness(
            predictions=[
                {
                    "instance_id": "repo__1",
                    "model_name_or_path": "CoACT/CoACT",
                    "model_patch": "diff --git a/a.py b/a.py",
                }
            ],
            dataset_name="data/SWE-bench_Verified",
            split="test",
            instance_ids=["repo__1"],
            max_workers=2,
            timeout=1800,
            run_id="CoACT-20260317",
            predictions_path=tmp_path / "predictions.json",
            report_path=tmp_path / "report.json",
        )

        assert Path(result.predictions_path).exists()
        assert Path(result.report_path).exists()
        assert not report_path.exists()
        assert result.total_instances == 4
        assert result.resolved_instances == 3
        assert result.pass_at_1 == 0.75

    def test_run_swebench_harness_ignores_stale_out_of_scope_artifacts(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Stale predictions/report from a larger earlier run must not inflate results."""
        predictions_path = tmp_path / "predictions.json"
        report_path = tmp_path / "report.json"

        # Stale artifacts from an earlier, larger run that included an out-of-scope id.
        predictions_path.write_text(
            json.dumps(
                [
                    {"instance_id": "repo__1", "model_name_or_path": "m", "model_patch": "old"},
                    {"instance_id": "repo__OUT", "model_name_or_path": "m", "model_patch": "old"},
                ]
            ),
            encoding="utf-8",
        )
        report_path.write_text(
            json.dumps(
                {
                    "total_instances": 2,
                    "submitted_instances": 2,
                    "completed_instances": 2,
                    "resolved_instances": 2,
                    "unresolved_instances": 0,
                    "empty_patch_instances": 0,
                    "error_instances": 0,
                    "submitted_ids": ["repo__1", "repo__OUT"],
                    "completed_ids": ["repo__1", "repo__OUT"],
                    "resolved_ids": ["repo__1", "repo__OUT"],
                    "unresolved_ids": [],
                    "empty_patch_ids": [],
                    "error_ids": [],
                    "incomplete_ids": [],
                }
            ),
            encoding="utf-8",
        )

        generated = tmp_path.parent / "fresh-report.json"

        def fake_main(**kwargs):
            # Harness evaluates only the requested instance (repo__1); it is unresolved.
            generated.write_text(
                json.dumps(
                    {
                        "total_instances": 1,
                        "submitted_instances": 1,
                        "completed_instances": 1,
                        "resolved_instances": 0,
                        "unresolved_instances": 1,
                        "empty_patch_instances": 0,
                        "error_instances": 0,
                    }
                ),
                encoding="utf-8",
            )
            return generated

        monkeypatch.setattr("src.eval.swebench_harness.harness_main", fake_main)

        result = run_swebench_harness(
            predictions=[
                {"instance_id": "repo__1", "model_name_or_path": "m", "model_patch": "new"}
            ],
            dataset_name="data/SWE-bench_Verified",
            split="test",
            instance_ids=["repo__1"],
            max_workers=1,
            timeout=1800,
            run_id="CoACT-scope",
            predictions_path=predictions_path,
            report_path=report_path,
        )

        # Result reflects only the current scope, not the stale 2-instance report.
        assert result.total_instances == 1
        assert result.resolved_instances == 0
        assert result.pass_at_1 == 0.0

        # The out-of-scope id is dropped from the predictions written for the harness.
        written = json.loads(predictions_path.read_text())
        assert [pred["instance_id"] for pred in written] == ["repo__1"]
