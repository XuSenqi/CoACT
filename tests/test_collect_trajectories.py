"""Tests for trajectory collection script.

Tests the complete pipeline for collecting agent trajectories from SWE-smith dataset:
1. Loading instances from HuggingFace datasets
2. Running MiniSWERunner on each instance
3. Saving trajectories to JSON files
4. Checkpoint/resume functionality
"""

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.config.config import (
    AgentConfig,
    Config,
    SFTConfig,
    SFTRolloutConfig,
)


@dataclass(frozen=True)
class MockSWEInstance:
    """Mock SWE-smith instance for testing."""
    instance_id: str
    problem_statement: str
    repo: str
    patch: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    image_name: str


@pytest.fixture
def mock_swe_instance() -> MockSWEInstance:
    """Create a mock SWE-smith instance."""
    return MockSWEInstance(
        instance_id="test-repo__test-issue-123",
        problem_statement="Fix the bug in the authentication module",
        repo="test-user/test-repo",
        patch="diff --git a/auth.py ...",
        FAIL_TO_PASS=["tests/test_auth.py::test_login"],
        PASS_TO_PASS=["tests/test_auth.py::test_logout"],
        image_name="test-image:latest",
    )


@pytest.fixture
def mock_trajectory_result() -> dict[str, Any]:
    """Mock result from MiniSWERunner.run()."""
    return {
        "exit_status": "submitted",
        "submission": "Fixed the bug",
        "trajectory": {
            "info": {
                "exit_status": "submitted",
                "submission": "Fixed the bug",
                "instance_id": "test-repo__test-issue-123",
            },
            "messages": [
                {"role": "system", "content": "You are a helpful assistant..."},
                {"role": "user", "content": "Fix the bug in the authentication module"},
                {"role": "assistant", "content": "Let me start by...", "tool_calls": []},
                {"role": "tool", "content": "File content: auth.py..."},
            ],
        },
    }


def make_test_config() -> Config:
    """Create a test config with SFT-local rollout and training model settings."""
    return Config(
        agent=AgentConfig(
            model="openai/test-agent",
            api_endpoint="http://localhost:8000/v1",
        ),
        sft=SFTConfig(
            model_path="model/Qwen3.5-4B",
            rollout=SFTRolloutConfig(
                model="openai/test-compressor",
                api_endpoint="http://localhost:8001/v1",
            ),
        ),
    )


@pytest.fixture
def mock_config(self, tmp_path: Path) -> Config:
    """Create a mock Config for testing."""
    return make_test_config()


class TestCollectionConfig:
    """Tests for CollectionConfig dataclass."""

    def test_collection_config_defaults(self) -> None:
        """Test default values for collection config."""
        from scripts.collect_trajectories import CollectionConfig

        config = CollectionConfig(
            dataset_name="SWE-bench/SWE-smith",
            output_dir="data/trajectories",
        )

        assert config.dataset_name == "SWE-bench/SWE-smith"
        assert config.output_dir == "data/trajectories"
        assert config.max_instances is None
        assert config.start_index == 0
        assert config.max_workers == 1

    def test_collection_config_custom_values(self) -> None:
        """Test custom values for collection config."""
        from scripts.collect_trajectories import CollectionConfig

        config = CollectionConfig(
            dataset_name="SWE-bench/SWE-smith",
            output_dir="data/trajectories",
            max_instances=100,
            start_index=50,
            max_workers=2,
        )

        assert config.max_instances == 100
        assert config.start_index == 50
        assert config.max_workers == 2


class TestTrajectoryCollector:
    """Tests for TrajectoryCollector class."""

    @pytest.fixture
    def test_config(self) -> Config:
        """Create a test Config for testing."""
        return make_test_config()

    def test_collector_initialization(self, tmp_path: Path, test_config: Config) -> None:
        """Test that TrajectoryCollector initializes correctly."""
        from scripts.collect_trajectories import CollectionConfig, TrajectoryCollector

        collection_config = CollectionConfig(
            dataset_name="SWE-bench/SWE-smith",
            output_dir=str(tmp_path / "trajectories"),
        )

        collector = TrajectoryCollector(collection_config, test_config)

        assert collector.collection_config == collection_config
        assert collector.output_dir.exists()

    def test_load_dataset(self, tmp_path: Path, test_config: Config) -> None:
        """Test loading SWE-smith dataset from HuggingFace."""
        from scripts.collect_trajectories import CollectionConfig, TrajectoryCollector

        collection_config = CollectionConfig(
            dataset_name="SWE-bench/SWE-smith",
            output_dir=str(tmp_path / "trajectories"),
        )

        collector = TrajectoryCollector(collection_config, test_config)

        # Mock the dataset loading
        with patch("scripts.collect_trajectories.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=10)
            mock_dataset.__getitem__ = MagicMock(return_value={
                "instance_id": "test-1",
                "problem_statement": "Test issue",
                "repo": "test/repo",
                "patch": "diff ...",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
                "image_name": "test:latest",
            })
            mock_load.return_value = mock_dataset

            dataset = collector.load_dataset()

            assert len(dataset) == 10
            mock_load.assert_called_once_with(
                "SWE-bench/SWE-smith",
                split="train",
            )

    def test_save_trajectory(
        self,
        tmp_path: Path,
        test_config: Config,
        mock_swe_instance: MockSWEInstance,
        mock_trajectory_result: dict[str, Any],
    ) -> None:
        """Test saving a trajectory to file."""
        from scripts.collect_trajectories import CollectionConfig, TrajectoryCollector

        collection_config = CollectionConfig(
            dataset_name="SWE-bench/SWE-smith",
            output_dir=str(tmp_path / "trajectories"),
        )

        collector = TrajectoryCollector(collection_config, test_config)

        collector.save_trajectory(
            instance_id=mock_swe_instance.instance_id,
            trajectory_data=mock_trajectory_result["trajectory"],
        )

        # Check file was created
        expected_path = tmp_path / "trajectories" / f"{mock_swe_instance.instance_id}.json"
        assert expected_path.exists()

        # Check content
        with open(expected_path) as f:
            saved_data = json.load(f)

        assert saved_data["info"]["instance_id"] == mock_swe_instance.instance_id

    def test_checkpoint_save_and_load(self, tmp_path: Path, test_config: Config) -> None:
        """Test checkpoint save and resume functionality."""
        from scripts.collect_trajectories import CollectionConfig, TrajectoryCollector

        collection_config = CollectionConfig(
            dataset_name="SWE-bench/SWE-smith",
            output_dir=str(tmp_path / "trajectories"),
        )

        collector = TrajectoryCollector(collection_config, test_config)

        # Mark items as completed
        completed_ids = ["instance-1", "instance-2", "instance-3"]
        for item_id in completed_ids:
            collector.checkpoint.mark_completed(item_id)

        # Check checkpoint file
        checkpoint_path = tmp_path / "trajectories" / ".checkpoint.json"
        assert checkpoint_path.exists()

        # Load checkpoint in new collector
        new_collector = TrajectoryCollector(collection_config, test_config)
        loaded_ids = new_collector.checkpoint.load()

        assert loaded_ids == set(completed_ids)

    def test_resume_from_checkpoint(self, tmp_path: Path, test_config: Config) -> None:
        """Test resuming collection from checkpoint."""
        from scripts.collect_trajectories import CollectionConfig, TrajectoryCollector

        collection_config = CollectionConfig(
            dataset_name="SWE-bench/SWE-smith",
            output_dir=str(tmp_path / "trajectories"),
            start_index=0,
        )

        collector = TrajectoryCollector(collection_config, test_config)

        # Create a checkpoint
        completed_ids = ["instance-1", "instance-2"]
        for item_id in completed_ids:
            collector.checkpoint.mark_completed(item_id)

        # Create new collector (should load checkpoint)
        new_collector = TrajectoryCollector(collection_config, test_config)
        loaded_ids = new_collector.checkpoint.load()

        # Verify checkpoint was loaded
        assert "instance-1" in loaded_ids
        assert "instance-2" in loaded_ids

    def test_collect_single_instance(
        self,
        tmp_path: Path,
        test_config: Config,
        mock_swe_instance: MockSWEInstance,
        mock_trajectory_result: dict[str, Any],
    ) -> None:
        """Test collecting a single trajectory."""
        from scripts.collect_trajectories import CollectionConfig, TrajectoryCollector

        collection_config = CollectionConfig(
            dataset_name="SWE-bench/SWE-smith",
            output_dir=str(tmp_path / "trajectories"),
        )

        collector = TrajectoryCollector(collection_config, test_config)

        # Mock the runner
        with patch("scripts.collect_trajectories.MiniSWERunner") as mock_runner:
            mock_runner.return_value.run.return_value = mock_trajectory_result

            instance_dict = {
                "instance_id": mock_swe_instance.instance_id,
                "problem_statement": mock_swe_instance.problem_statement,
                "repo": mock_swe_instance.repo,
            }

            result = collector.collect_instance(instance_dict)

            assert result is not None
            assert result["info"]["instance_id"] == mock_swe_instance.instance_id

    def test_collect_instance_handles_error(
        self, tmp_path: Path, test_config: Config, mock_swe_instance: MockSWEInstance
    ) -> None:
        """Test that collection handles errors gracefully."""
        from scripts.collect_trajectories import CollectionConfig, TrajectoryCollector

        collection_config = CollectionConfig(
            dataset_name="SWE-bench/SWE-smith",
            output_dir=str(tmp_path / "trajectories"),
        )

        collector = TrajectoryCollector(collection_config, test_config)

        # Mock the runner to raise an error
        with patch("scripts.collect_trajectories.MiniSWERunner") as mock_runner:
            mock_runner.return_value.run.side_effect = Exception("Agent failed")

            instance_dict = {
                "instance_id": mock_swe_instance.instance_id,
                "problem_statement": mock_swe_instance.problem_statement,
                "repo": mock_swe_instance.repo,
            }

            result = collector.collect_instance(instance_dict)

            # Should return None on error, not raise
            assert result is None


class TestCollectionIntegration:
    """Integration tests for trajectory collection."""

    @pytest.fixture
    def test_config(self) -> Config:
        """Create a test Config for testing."""
        return make_test_config()

    @pytest.mark.slow
    def test_collect_from_mock_dataset(
        self, tmp_path: Path, test_config: Config, mock_trajectory_result: dict[str, Any]
    ) -> None:
        """Test collecting trajectories from a mock dataset."""
        from scripts.collect_trajectories import CollectionConfig, TrajectoryCollector

        collection_config = CollectionConfig(
            dataset_name="SWE-bench/SWE-smith",
            output_dir=str(tmp_path / "trajectories"),
            max_instances=3,
            filter_by_docker=False,  # Disable Docker filtering for test
        )

        collector = TrajectoryCollector(collection_config, test_config)

        # Mock dataset
        mock_instances = [
            {
                "instance_id": f"test-{i}",
                "problem_statement": f"Fix bug {i}",
                "repo": "test/repo",
                "patch": "diff ...",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
                "image_name": "test:latest",
            }
            for i in range(5)
        ]

        with patch("scripts.collect_trajectories.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=len(mock_instances))
            mock_dataset.__iter__ = MagicMock(return_value=iter(mock_instances))
            mock_dataset.__getitem__ = lambda _, idx: mock_instances[idx]
            mock_load.return_value = mock_dataset

            # Mock runner
            with patch("scripts.collect_trajectories.MiniSWERunner") as mock_runner:
                mock_runner.return_value.run.return_value = mock_trajectory_result

                # Collect
                stats = collector.collect_all()

                assert stats["total"] == 3  # max_instances=3
                assert stats["success"] >= 0
                assert stats["failed"] >= 0
                assert stats["success"] + stats["failed"] == stats["total"]

    def test_skip_already_collected(
        self, tmp_path: Path, test_config: Config, mock_trajectory_result: dict[str, Any]
    ) -> None:
        """Test that already collected instances are skipped."""
        from scripts.collect_trajectories import CollectionConfig, TrajectoryCollector

        collection_config = CollectionConfig(
            dataset_name="SWE-bench/SWE-smith",
            output_dir=str(tmp_path / "trajectories"),
            max_instances=5,
            filter_by_docker=False,  # Disable Docker filtering for test
        )

        collector = TrajectoryCollector(collection_config, test_config)

        # Pre-save a trajectory
        collector.save_trajectory("test-0", mock_trajectory_result["trajectory"])

        mock_instances = [
            {"instance_id": f"test-{i}", "problem_statement": f"Fix bug {i}", "repo": "test/repo"}
            for i in range(3)
        ]

        with patch("scripts.collect_trajectories.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=len(mock_instances))
            mock_dataset.__iter__ = MagicMock(return_value=iter(mock_instances))
            mock_dataset.__getitem__ = lambda _, idx: mock_instances[idx]
            mock_load.return_value = mock_dataset

            with patch("scripts.collect_trajectories.MiniSWERunner") as mock_runner:
                mock_runner.return_value.run.return_value = mock_trajectory_result

                collector.collect_all()

                # Only 2 calls should be made (test-0 already exists)
                assert mock_runner.return_value.run.call_count == 2

    def test_collect_all_saves_on_main_thread_with_parallel_workers(
        self, tmp_path: Path, test_config: Config, mock_trajectory_result: dict[str, Any]
    ) -> None:
        """Trajectory saving and checkpoint updates should stay on the main thread."""
        from scripts.collect_trajectories import CollectionConfig, TrajectoryCollector

        collection_config = CollectionConfig(
            dataset_name="SWE-bench/SWE-smith",
            output_dir=str(tmp_path / "trajectories"),
            max_instances=2,
            max_workers=2,
            filter_by_docker=False,
        )
        collector = TrajectoryCollector(collection_config, test_config)

        mock_instances = [
            {"instance_id": "test-0", "problem_statement": "Fix bug 0", "repo": "test/repo"},
            {"instance_id": "test-1", "problem_statement": "Fix bug 1", "repo": "test/repo"},
        ]
        saved_threads: list[str] = []
        checkpoint_threads: list[str] = []
        original_mark_completed = collector.checkpoint.mark_completed

        def fake_collect_instance(instance: dict[str, Any]) -> dict[str, Any]:
            return {
                "info": {
                    "instance_id": instance["instance_id"],
                    "exit_status": "submitted",
                    "submission": "patch",
                },
                "messages": [],
            }

        def fake_save_trajectory(instance_id: str, trajectory_data: dict[str, Any]) -> Path:
            saved_threads.append(threading.current_thread().name)
            output_path = collector.output_dir / f"{instance_id}.json"
            output_path.write_text(json.dumps(trajectory_data), encoding="utf-8")
            return output_path

        def wrapped_mark_completed(instance_id: str) -> None:
            checkpoint_threads.append(threading.current_thread().name)
            original_mark_completed(instance_id)

        collector.collect_instance = fake_collect_instance
        collector.save_trajectory = fake_save_trajectory
        collector.checkpoint.mark_completed = wrapped_mark_completed

        with patch("scripts.collect_trajectories.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=len(mock_instances))
            mock_dataset.__getitem__ = lambda _, idx: mock_instances[idx]
            mock_load.return_value = mock_dataset

            stats = collector.collect_all()

        assert stats["success"] == 2
        assert saved_threads == ["MainThread", "MainThread"]
        assert checkpoint_threads == ["MainThread", "MainThread"]


class TestMainFunction:
    """Tests for the main CLI function."""

    def test_main_with_defaults(self, tmp_path: Path) -> None:
        """Test main function with default arguments."""
        from scripts.collect_trajectories import main

        with patch("sys.argv", ["collect_trajectories.py", "--config", "config/config.yaml"]):
            with patch("scripts.collect_trajectories.TrajectoryCollector") as mock_collector:
                mock_instance = MagicMock()
                mock_instance.collect_all.return_value = {
                    "total": 0, "success": 0, "failed": 0, "skipped": 0, "duration_seconds": None
                }
                mock_collector.return_value = mock_instance

                with patch("scripts.collect_trajectories.Config.load") as mock_config_load:
                    mock_config_load.return_value = make_test_config()

                    # Should not raise
                    main()

                    assert mock_collector.called
                    collection_config = mock_collector.call_args[0][0]
                    assert collection_config.max_instances == 50

    def test_main_with_custom_config(self, tmp_path: Path) -> None:
        """Test main function with custom config file."""
        from scripts.collect_trajectories import main

        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("agent:\n  model_path: test\n")

        with patch("sys.argv", [
            "collect_trajectories.py",
            "--config", str(config_path),
            "--max-trajectories", "123",
        ]):
            with patch("scripts.collect_trajectories.TrajectoryCollector") as mock_collector:
                mock_instance = MagicMock()
                mock_instance.collect_all.return_value = {
                    "total": 0, "success": 0, "failed": 0, "skipped": 0, "duration_seconds": None
                }
                mock_collector.return_value = mock_instance

                with patch("scripts.collect_trajectories.Config.load") as mock_config_load:
                    mock_config_load.return_value = make_test_config()

                    main()

                    # Check that collector was called with correct args
                    assert mock_collector.called
                    collection_config = mock_collector.call_args[0][0]
                    assert collection_config.max_instances == 123


class TestDockerImageFilter:
    """Tests for Docker image filtering functionality."""

    @pytest.fixture
    def test_config(self) -> Config:
        """Create a test Config for testing."""
        return make_test_config()

    def test_get_available_docker_images(self) -> None:
        """Test getting available Docker images using docker SDK."""
        from scripts.collect_trajectories import get_available_docker_images

        # Mock docker SDK
        with patch("docker.from_env") as mock_docker:
            mock_client = MagicMock()
            mock_image = MagicMock()
            mock_image.tags = [
                "swebench/swesmith.x86_64.test_repo_1776_test.abc123:latest",
                "swebench/swesmith.x86_64.other_repo_1776_other.def456:latest",
            ]
            mock_client.images.list.return_value = [mock_image]
            mock_docker.return_value = mock_client

            images = get_available_docker_images()

            assert len(images) == 2
            assert "swebench/swesmith.x86_64.test_repo_1776_test.abc123" in images

    def test_get_docker_images_handles_error(self) -> None:
        """Test that Docker errors are handled gracefully."""
        from scripts.collect_trajectories import get_available_docker_images

        # Test with docker.from_env raising an exception
        with patch("docker.from_env", side_effect=Exception("Docker daemon not running")):
            images = get_available_docker_images()
            assert images == []

    def test_normalize_image_name(self) -> None:
        """Test normalizing Docker image names for matching."""
        from scripts.collect_trajectories import normalize_image_name

        # Dataset format: jyangballin/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536
        # Docker format: swebench/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536:latest

        dataset_name = "jyangballin/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536"
        docker_name = "swebench/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536:latest"

        normalized_dataset = normalize_image_name(dataset_name)
        normalized_docker = normalize_image_name(docker_name)

        assert normalized_dataset == normalized_docker

    def test_filter_instances_by_docker_images(self) -> None:
        """Test filtering instances by available Docker images."""
        from scripts.collect_trajectories import filter_instances_by_docker_images

        instances = [
            {"instance_id": "test-1", "image_name": "swebench/swesmith.x86_64.available.123"},
            {"instance_id": "test-2", "image_name": "swebench/swesmith.x86_64.not_available.456"},
            {"instance_id": "test-3", "image_name": "swebench/swesmith.x86_64.available.789"},
        ]

        available_images = [
            "swebench/swesmith.x86_64.available.123",
            "swebench/swesmith.x86_64.available.789",
        ]

        filtered = filter_instances_by_docker_images(instances, available_images)

        assert len(filtered) == 2
        assert filtered[0]["instance_id"] == "test-1"
        assert filtered[1]["instance_id"] == "test-3"

    def test_filter_instances_empty_images(self) -> None:
        """Test filtering when no Docker images are available."""
        from scripts.collect_trajectories import filter_instances_by_docker_images

        instances = [
            {"instance_id": "test-1", "image_name": "swebench/swesmith.x86_64.test.123"},
        ]

        filtered = filter_instances_by_docker_images(instances, [])
        assert len(filtered) == 0

    def test_filter_instances_no_filter(self) -> None:
        """Test filtering when available_images is None (no filtering)."""
        from scripts.collect_trajectories import filter_instances_by_docker_images

        instances = [
            {"instance_id": "test-1", "image_name": "swebench/swesmith.x86_64.test.123"},
            {"instance_id": "test-2", "image_name": "swebench/swesmith.x86_64.test.456"},
        ]

        filtered = filter_instances_by_docker_images(instances, None)
        assert len(filtered) == 2


class TestCollectionWithDocker:
    """Tests for trajectory collection with Docker filtering."""

    @pytest.fixture
    def test_config(self) -> Config:
        """Create a test Config for testing."""
        return make_test_config()

    def test_collect_with_docker_filter(self, tmp_path: Path, test_config: Config) -> None:
        """Test collection with Docker image filtering."""
        from scripts.collect_trajectories import CollectionConfig, TrajectoryCollector

        collection_config = CollectionConfig(
            dataset_name="SWE-bench/SWE-smith",
            output_dir=str(tmp_path / "trajectories"),
            filter_by_docker=True,
            max_instances=5,
        )

        mock_instances = [
            {
                "instance_id": f"test-{i}",
                "problem_statement": f"Fix bug {i}",
                "repo": "test/repo",
                "patch": "diff ...",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
                "image_name": f"swebench/swesmith.x86_64.test_{i}.123",
            }
            for i in range(5)
        ]

        with patch("scripts.collect_trajectories.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=len(mock_instances))
            mock_dataset.__iter__ = MagicMock(return_value=iter(mock_instances))
            mock_dataset.__getitem__ = lambda _, idx: mock_instances[idx]
            mock_load.return_value = mock_dataset

            # Mock docker SDK
            with patch("docker.from_env") as mock_docker:
                mock_client = MagicMock()
                mock_image = MagicMock()
                mock_image.tags = [
                    "swebench/swesmith.x86_64.test_0.123:latest",
                    "swebench/swesmith.x86_64.test_1.123:latest",
                ]
                mock_client.images.list.return_value = [mock_image]
                mock_docker.return_value = mock_client

                collector = TrajectoryCollector(collection_config, test_config)
                filtered = collector.get_filtered_dataset(mock_dataset)

                assert len(filtered) == 2

    def test_collect_without_docker_filter(self, tmp_path: Path, test_config: Config) -> None:
        """Test collection without Docker image filtering."""
        from scripts.collect_trajectories import CollectionConfig, TrajectoryCollector

        collection_config = CollectionConfig(
            dataset_name="SWE-bench/SWE-smith",
            output_dir=str(tmp_path / "trajectories"),
            filter_by_docker=False,
            max_instances=5,
        )

        mock_instances = [
            {
                "instance_id": f"test-{i}",
                "problem_statement": f"Fix bug {i}",
                "repo": "test/repo",
                "image_name": f"swebench/swesmith.x86_64.test_{i}.123",
            }
            for i in range(5)
        ]

        with patch("scripts.collect_trajectories.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=len(mock_instances))
            mock_dataset.__iter__ = MagicMock(return_value=iter(mock_instances))
            mock_dataset.__getitem__ = lambda _, idx: mock_instances[idx]
            mock_load.return_value = mock_dataset

            collector = TrajectoryCollector(collection_config, test_config)
            filtered = collector.get_filtered_dataset(mock_dataset)

            assert len(filtered) == 5
