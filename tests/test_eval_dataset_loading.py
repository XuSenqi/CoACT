"""Tests for SWE-bench Verified dataset loading helpers."""

from unittest.mock import patch

from src.eval.runner import load_swebench_verified


class TestLoadSwebenchVerified:
    """Tests for official SWE-bench Verified loading."""

    def test_load_swebench_verified_uses_official_image_names(self) -> None:
        """Verified instances should be enriched with official sweb image keys."""
        mock_dataset = [
            {
                "instance_id": "astropy__astropy-12907",
                "repo": "astropy/astropy",
                "version": "5.0",
                "base_commit": "deadbeef",
                "problem_statement": "Fix bug",
                "test_patch": "diff --git a/tests.py b/tests.py",
                "FAIL_TO_PASS": "[]",
                "PASS_TO_PASS": "[]",
            }
        ]

        with patch("src.eval.runner.load_dataset", return_value=mock_dataset) as mock_load_dataset:
            instances = load_swebench_verified("princeton-nlp/SWE-bench_Verified", split="dev")

        assert len(instances) == 1
        assert instances[0]["instance_id"] == "astropy__astropy-12907"
        mock_load_dataset.assert_called_once_with("princeton-nlp/SWE-bench_Verified", split="dev")
        assert (
            instances[0]["image_name"]
            == "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
        )
