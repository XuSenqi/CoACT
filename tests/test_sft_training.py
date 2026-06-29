"""Tests for SFT training script and raw rollout data format."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")
pytest.importorskip("trl")

import scripts.train_sft as train_sft


class _MockTokenizer:
    """Simple tokenizer stub for dataset loading tests."""

    eos_token = "<eos>"
    pad_token = "<pad>"
    unk_token = "<unk>"
    unk_token_id = 0

    def __init__(self) -> None:
        """Initialize a small mutable vocabulary for tokenizer tests."""
        self._vocab = {self.unk_token: self.unk_token_id}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        """Return a deterministic prompt wrapper."""
        del tokenize, add_generation_prompt
        return f"chat::{messages[0]['content']}"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Approximate token counting by whitespace."""
        del add_special_tokens
        return list(range(len(text.split())))


class TestSFTDataFormat:
    """Tests for SFT data format validation."""

    @pytest.fixture
    def sample_sft_example(self) -> dict:
        """Create a sample raw rollout example.

        Note: completion is the COMPRESSED tool output, not the bash command.
        The bash command similarity is only used for scoring during data preparation.
        """
        return {
            "prompt": """You are a context compression assistant. Your task is to compress the tool output while preserving information essential for the agent to take the correct next action.

## Global Task Goal
Fix the bug in the authentication module that causes users to be logged out unexpectedly.

## Agent's Intent (Why this tool was called)
Check the current session configuration in the auth module.

## Original Tool Output
<returncode>0</returncode>
<output>
# Authentication Configuration
SESSION_TIMEOUT = 3600
SESSION_COOKIE_NAME = 'sessionid'
...
</output>

## Instructions
Compress the tool output above. Keep only information that is essential for determining the next action. Be concise but accurate.

## Compressed Output
""",
            "completion": "SESSION_TIMEOUT=3600, SESSION_COOKIE_NAME='sessionid' in auth/config.py",
            "action_reward": 0.9,
            "length_reward": 0.05,
            "total_reward": 0.95,
            "inferred_action": "grep SESSION_TIMEOUT auth/config.py",
            "ground_truth": ["grep SESSION_TIMEOUT auth/config.py"],
            "trajectory_id": "traj-001",
            "step_id": 7,
        }

    def test_sft_example_has_required_fields(self, sample_sft_example: dict) -> None:
        """Test that SFT example has all required fields."""
        assert "prompt" in sample_sft_example
        assert "completion" in sample_sft_example
        assert "action_reward" in sample_sft_example
        assert "length_reward" in sample_sft_example
        assert "total_reward" in sample_sft_example
        assert isinstance(sample_sft_example["prompt"], str)
        assert isinstance(sample_sft_example["completion"], str)
        assert len(sample_sft_example["prompt"]) > 0
        assert len(sample_sft_example["completion"]) > 0

    def test_sft_dataset_format(self, sample_sft_example: dict) -> None:
        """Test that SFT dataset is valid JSON lines format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Write multiple examples
            for _ in range(3):
                f.write(json.dumps(sample_sft_example) + "\n")
            temp_path = f.name

        try:
            # Read and validate
            with open(temp_path) as f:
                for line in f:
                    example = json.loads(line)
                    assert "prompt" in example
                    assert "completion" in example
        finally:
            Path(temp_path).unlink()

    def test_prompt_contains_compression_template(self, sample_sft_example: dict) -> None:
        """Test that prompt follows compression template format."""
        prompt = sample_sft_example["prompt"]

        # Check required sections
        assert "Global Task Goal" in prompt
        assert "Agent's Intent" in prompt
        assert "Original Tool Output" in prompt
        assert "Compressed Output" in prompt

    def test_completion_is_concise(self, sample_sft_example: dict) -> None:
        """Test that completion is reasonably concise."""
        # Compressed output should be shorter than original tool output
        completion = sample_sft_example["completion"]
        # Typical compressed output should be < 500 chars
        assert len(completion) < 500

    def test_completion_is_compressed_output_not_bash(self, sample_sft_example: dict) -> None:
        """Test that completion is compressed tool output, not a bash command.

        The SFT completion should be the compressed version of the tool output,
        NOT the bash command. Bash similarity is only used during data preparation
        to score which compressed outputs lead to correct actions.
        """
        completion = sample_sft_example["completion"]
        # Completion should NOT look like a bash command
        # It should be a description/summary of the tool output
        assert not completion.startswith("git ")
        assert not completion.startswith("ls ")
        assert not completion.startswith("cd ")
        assert " in " in completion or "=" in completion  # Should describe content


class TestSFTDataLoader:
    """Tests for SFT data loading functionality."""

    @pytest.fixture
    def temp_dataset(self, tmp_path: Path) -> Path:
        """Create a temporary raw rollout dataset file."""
        data_file = tmp_path / "sft_train.json"

        examples = [
            {
                "prompt": "Prompt step 1\n## Compressed Output\n",
                "completion": "keep-high",
                "action_reward": 0.95,
                "length_reward": 0.05,
                "total_reward": 1.0,
                "inferred_action": "grep TODO app.py",
                "ground_truth": ["grep TODO app.py"],
                "trajectory_id": "traj-1",
                "step_id": 1,
            },
            {
                "prompt": "Prompt step 1\n## Compressed Output\n",
                "completion": "duplicate completion",
                "action_reward": 0.92,
                "length_reward": 0.02,
                "total_reward": 0.94,
                "inferred_action": "grep TODO app.py",
                "ground_truth": ["grep TODO app.py"],
                "trajectory_id": "traj-1",
                "step_id": 1,
            },
            {
                "prompt": "Prompt step 1\n## Compressed Output\n",
                "completion": "duplicate completion",
                "action_reward": 0.91,
                "length_reward": 0.01,
                "total_reward": 0.92,
                "inferred_action": "grep TODO app.py",
                "ground_truth": ["grep TODO app.py"],
                "trajectory_id": "traj-1",
                "step_id": 1,
            },
            {
                "prompt": "Prompt step 1\n## Compressed Output\n",
                "completion": "drop-low",
                "action_reward": 0.4,
                "length_reward": 0.0,
                "total_reward": 0.4,
                "inferred_action": "grep TODO app.py",
                "ground_truth": ["grep TODO app.py"],
                "trajectory_id": "traj-1",
                "step_id": 1,
            },
            {
                "prompt": "Prompt step 2\n## Compressed Output\n",
                "completion": '{"type":"unchanged","content":null}',
                "action_reward": 0.9,
                "length_reward": 0.05,
                "total_reward": 0.95,
                "inferred_action": "grep TODO app.py",
                "ground_truth": ["grep TODO app.py"],
                "trajectory_id": "traj-1",
                "step_id": 2,
            },
        ]

        with open(data_file, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        return data_file

    def test_load_sft_dataset(self, temp_dataset: Path) -> None:
        """Test loading selected SFT dataset from raw rollout file."""
        train_dataset, eval_dataset = train_sft.load_sft_dataset(
            data_paths=[str(temp_dataset)],
            tokenizer=_MockTokenizer(),
            max_length=256,
            select_top_k=3,
            min_similarity=0.8,
            inject_unchanged_fallback=False,
            eval_split_ratio=0.34,
            seed=42,
            drop_overlength_examples=False,
        )

        total_examples = len(train_dataset) + len(eval_dataset)
        assert total_examples == 4
        # Plain text format: prompt has chat template applied, completion has EOS
        for ds in [train_dataset, eval_dataset]:
            for ex in ds:
                assert isinstance(ex["prompt"], str)
                assert ex["prompt"].startswith("chat::")
                assert isinstance(ex["completion"], str)
                assert ex["completion"].endswith("<eos>")

    def test_load_sft_dataset_splits_on_group_boundaries(self, tmp_path: Path) -> None:
        """Train/eval split should keep one step group entirely in one split."""
        data_file = tmp_path / "sft_groups.jsonl"
        examples = [
            {
                "prompt": "Prompt step 1\n## Compressed Output\n",
                "completion": "step1-a",
                "action_reward": 0.92,
                "length_reward": 0.0,
                "total_reward": 0.92,
                "inferred_action": "cmd",
                "ground_truth": ["cmd"],
                "trajectory_id": "traj-1",
                "step_id": 1,
            },
            {
                "prompt": "Prompt step 1\n## Compressed Output\n",
                "completion": "step1-b",
                "action_reward": 0.88,
                "length_reward": 0.0,
                "total_reward": 0.88,
                "inferred_action": "cmd",
                "ground_truth": ["cmd"],
                "trajectory_id": "traj-1",
                "step_id": 1,
            },
            {
                "prompt": "Prompt step 2\n## Compressed Output\n",
                "completion": "step2-a",
                "action_reward": 0.91,
                "length_reward": 0.0,
                "total_reward": 0.91,
                "inferred_action": "cmd",
                "ground_truth": ["cmd"],
                "trajectory_id": "traj-1",
                "step_id": 2,
            },
            {
                "prompt": "Prompt step 2\n## Compressed Output\n",
                "completion": "step2-b",
                "action_reward": 0.89,
                "length_reward": 0.0,
                "total_reward": 0.89,
                "inferred_action": "cmd",
                "ground_truth": ["cmd"],
                "trajectory_id": "traj-1",
                "step_id": 2,
            },
        ]
        with open(data_file, "w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example) + "\n")

        train_dataset, eval_dataset = train_sft.load_sft_dataset(
            data_paths=[str(data_file)],
            tokenizer=_MockTokenizer(),
            max_length=256,
            select_top_k=4,
            min_similarity=0.65,
            inject_unchanged_fallback=False,
            eval_split_ratio=0.5,
            seed=42,
            drop_overlength_examples=False,
        )

        train_prompts = set(train_dataset["prompt"])
        eval_prompts = set(eval_dataset["prompt"])
        assert train_prompts.isdisjoint(eval_prompts)

    def test_load_sft_dataset_concatenates_multiple_files(self, tmp_path: Path) -> None:
        """Multiple data files are concatenated into one selection pool."""
        file_a = tmp_path / "offpolicy.jsonl"
        file_b = tmp_path / "dagger.jsonl"
        rec_a = {
            "prompt": "Prompt A\n## Compressed Output\n",
            "completion": "a",
            "action_reward": 0.95,
            "length_reward": 0.0,
            "inferred_action": "cmd",
            "ground_truth": ["cmd"],
            "trajectory_id": "offpolicy:inst-1",
            "step_id": 1,
        }
        # Same bare instance/step, different source prefix -> distinct group.
        rec_b = {**rec_a, "completion": "b", "trajectory_id": "dagger:inst-1"}
        with open(file_a, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(rec_a) + "\n")
        with open(file_b, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(rec_b) + "\n")

        train_dataset, eval_dataset = train_sft.load_sft_dataset(
            data_paths=[str(file_a), str(file_b)],
            tokenizer=_MockTokenizer(),
            max_length=256,
            select_top_k=4,
            min_similarity=0.65,
            inject_unchanged_fallback=False,
            eval_split_ratio=0.0,
            seed=42,
            drop_overlength_examples=False,
        )

        # eval_split_ratio=0.0 puts both (distinct) groups in train.
        assert len(train_dataset) == 2
        assert len(eval_dataset) == 0
        assert set(train_dataset["completion"]) == {"a<eos>", "b<eos>"}

    def test_resolve_sft_data_files_uses_sft_data_dir(self, tmp_path: Path) -> None:
        """Training data basenames should resolve under ``paths.sft_data_dir``."""
        sft_data_dir = tmp_path / "sft"
        sft_data_dir.mkdir()
        data_file = sft_data_dir / "sft_data.offpolicy.jsonl"
        data_file.write_text('{"prompt":"p","completion":"c"}\n', encoding="utf-8")
        config = SimpleNamespace(
            sft=SimpleNamespace(data_file=["sft_data.offpolicy.jsonl"]),
            paths=SimpleNamespace(sft_data_dir=str(sft_data_dir)),
        )

        assert train_sft.resolve_sft_data_files(config) == [data_file]

    def test_dataset_single_file(self, tmp_path: Path) -> None:
        """Test loading raw rollout dataset from a single file."""
        # Create single data file
        data_file = tmp_path / "sft_data.json"

        with open(data_file, "w", encoding="utf-8") as f:
            for i in range(100):
                f.write(
                    json.dumps(
                        {
                            "prompt": f"Prompt {i}",
                            "completion": f"output_{i}",
                            "action_reward": 0.9,
                            "length_reward": 0.0,
                            "total_reward": 0.9,
                            "inferred_action": "grep TODO app.py",
                            "ground_truth": ["grep TODO app.py"],
                            "trajectory_id": "traj-1",
                            "step_id": i,
                        }
                    )
                    + "\n"
                )

        # Verify count
        data_count = sum(1 for _ in open(data_file, encoding="utf-8"))

        assert data_count == 100


class TestSFTTrainingConfig:
    """Tests for SFT training configuration."""

    def test_sft_config_from_yaml(self) -> None:
        """Test loading SFT config from YAML."""
        from src.config.config import Config

        config = Config.load("config/config.yaml")

        assert config.sft.learning_rate > 0
        assert config.sft.num_train_epochs > 0
        assert config.sft.max_length > 0
        assert config.sft.bf16 is True
        assert config.sft.dataloader_drop_last is True

    def test_lora_config_available(self) -> None:
        """Test that LoRA config is available for training."""
        from src.config.config import Config

        config = Config.load("config/config.yaml")

        assert config.sft.lora.r > 0
        assert config.sft.lora.alpha > 0
        assert len(config.sft.lora.target_modules) > 0
