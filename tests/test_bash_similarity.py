"""Tests for bash command semantic similarity."""

import pytest

from src.reward.bash_similarity import (
    BashParseError,
    CommandSemantics,
    compute_command_similarity,
    extract_semantics,
    parse_bash_command,
)


class TestExtractSemantics:
    """Tests for semantic extraction from bash commands."""

    def test_simple_command(self) -> None:
        s = extract_semantics("ls -la")
        assert s.intent == "list"
        assert s.subcommand_verbs == ("ls",)

    def test_cat_file(self) -> None:
        s = extract_semantics("cat file.py")
        assert s.intent == "read"
        assert "file.py" in s.file_targets

    def test_grep_pattern_and_file(self) -> None:
        s = extract_semantics('grep -n "RunServer" server.go')
        assert s.intent == "search"
        assert "RunServer" in s.search_patterns
        assert "server.go" in s.file_targets

    def test_pipe_aggregates_features(self) -> None:
        s = extract_semantics('cat server.go | grep -n "RunServer"')
        assert s.intent == "search"  # grep dominates in pipe
        assert "server.go" in s.file_targets
        assert "RunServer" in s.search_patterns

    def test_cd_prefix_resolves_paths(self) -> None:
        s = extract_semantics("cd /testbed && grep -rn 'foo' src/")
        assert "/testbed/src" in s.file_targets or "/testbed/src/" in s.file_targets
        assert "foo" in s.search_patterns

    def test_sed_read_intent(self) -> None:
        s = extract_semantics("sed -n '10,20p' file.py")
        assert s.intent == "read"
        assert "file.py" in s.file_targets

    def test_sed_modify_intent(self) -> None:
        s = extract_semantics("sed -i 's/old/new/g' file.py")
        assert s.intent == "modify"
        assert "file.py" in s.file_targets

    def test_git_diff_intent(self) -> None:
        s = extract_semantics("git diff -- src/foo.py")
        assert s.intent == "read"
        assert s.git_subcmd == "diff"
        assert "src/foo.py" in s.file_targets

    def test_git_checkout_intent(self) -> None:
        s = extract_semantics("git checkout file.py")
        assert s.intent == "modify"

    def test_submit_detection(self) -> None:
        s = extract_semantics(
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
        )
        assert s.is_submit is True
        assert s.intent == "submit"

    def test_python_inline_code(self) -> None:
        s = extract_semantics('python3 -c "import json; print(1)"')
        assert s.intent == "run"
        assert "import json; print(1)" in s.inline_content

    def test_heredoc_content(self) -> None:
        s = extract_semantics("python3 << EOF\nprint('hello')\nEOF")
        assert s.intent == "run"
        assert "hello" in s.inline_content

    def test_flags_unordered(self) -> None:
        s1 = extract_semantics('grep -rn --include="*.py" "X" src/')
        s2 = extract_semantics('grep -n -r "X" --include="*.py" src/')
        assert s1.search_patterns == s2.search_patterns

    def test_head_tail_in_pipe_not_primary(self) -> None:
        """head/tail at end of pipe should not override intent."""
        s = extract_semantics("grep -rn 'panic' *.go | head -50")
        assert s.intent == "search"  # grep, not head

    def test_empty_command(self) -> None:
        result = parse_bash_command("")
        assert result is None

    def test_redirect_extracts_file(self) -> None:
        s = extract_semantics("git diff > patch.txt")
        assert "patch.txt" in s.file_targets


class TestComputeCommandSimilarity:
    """Tests for semantic similarity computation."""

    def test_identical(self) -> None:
        assert compute_command_similarity("ls -la", "ls -la") == 1.0

    def test_identical_extra_spaces(self) -> None:
        assert compute_command_similarity("ls -la", "ls  -la") == 1.0

    def test_empty_commands(self) -> None:
        assert compute_command_similarity("", "") == 1.0

    def test_one_empty(self) -> None:
        assert compute_command_similarity("ls", "") == 0.0

    # --- Pipe equivalence ---

    def test_pipe_equivalence_grep(self) -> None:
        """cat file | grep X ≈ grep X file."""
        sim = compute_command_similarity(
            'grep -n "RunServer" server.go',
            'cat server.go | grep -n "RunServer" -A 30',
        )
        assert sim >= 0.55

    # --- Same file, different tool ---

    def test_same_file_different_read_tool(self) -> None:
        """sed -n vs cat | head on same file → moderate-high."""
        sim = compute_command_similarity(
            "sed -n '155,180p' /testbed/file.py",
            "cat /testbed/file.py | head -200",
        )
        assert sim >= 0.40

    def test_git_diff_vs_cat_same_file(self) -> None:
        """git diff file vs cat file → moderate."""
        sim = compute_command_similarity(
            "git diff transaction.go",
            "cat transaction.go",
        )
        assert 0.25 <= sim <= 0.65

    # --- Submit detection ---

    def test_both_submit(self) -> None:
        assert compute_command_similarity(
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt",
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt",
        ) == 1.0

    def test_submit_vs_non_submit(self) -> None:
        sim = compute_command_similarity(
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt",
            "cat patch.txt",
        )
        assert sim == 0.0

    # --- Flag order independence ---

    def test_flag_order_independence(self) -> None:
        sim = compute_command_similarity(
            'grep -rn --include="*.py" "pattern" src/',
            'grep -n -r "pattern" --include="*.py" src/',
        )
        assert sim >= 0.90

    # --- cd prefix normalization ---

    def test_cd_prefix_equivalent(self) -> None:
        sim = compute_command_similarity(
            "cd /testbed && grep -rn 'foo' src/",
            "grep -rn 'foo' /testbed/src/",
        )
        assert sim >= 0.90

    # --- Heredoc vs -c ---

    def test_heredoc_vs_inline(self) -> None:
        sim = compute_command_similarity(
            'python3 -c "import json; print(json.dumps({}))"',
            "python3 << 'EOF'\nimport json\nprint(json.dumps({}))\nEOF",
        )
        assert sim >= 0.55

    # --- Different intent same file ---

    def test_sed_read_vs_modify(self) -> None:
        """sed -n (read) vs sed -i (modify) on same file → low."""
        sim = compute_command_similarity(
            "sed -n '10,20p' file.py",
            "sed -i 's/old/new/g' file.py",
        )
        assert sim <= 0.40

    # --- Different files ---

    def test_different_files_same_tool(self) -> None:
        """git diff on different files → low."""
        sim = compute_command_similarity(
            "git diff -- src/foo.py",
            "git diff -- src/bar.py",
        )
        assert sim <= 0.35

    def test_same_file_same_git_subcmd(self) -> None:
        """git diff same file → high."""
        sim = compute_command_similarity(
            "git diff -- src/foo.py",
            "git diff HEAD src/foo.py",
        )
        assert sim >= 0.90

    # --- Completely unrelated ---

    def test_completely_unrelated(self) -> None:
        sim = compute_command_similarity(
            "cat /testbed/go.mod",
            "git log --all --oneline",
        )
        assert sim <= 0.35

    def test_unrelated_search_vs_read(self) -> None:
        sim = compute_command_similarity(
            "grep -rn 'panic' --include='*.go' | head -50",
            "cat TODO",
        )
        assert sim <= 0.35

    # --- Line range overlap ---

    def test_overlapping_sed_ranges(self) -> None:
        """sed with overlapping line ranges on same file → high."""
        sim = compute_command_similarity(
            "sed -n '10,30p' file.py",
            "sed -n '15,25p' file.py",
        )
        assert sim >= 0.80

    # --- Different pattern, different file ---

    def test_different_pattern_different_file(self) -> None:
        sim = compute_command_similarity(
            'grep -n "wrap_set_operation_queries" pypika/dialects.py',
            'cat pypika/queries.py | head -200',
        )
        assert sim <= 0.35


class TestBashParseError:
    """Tests for BashParseError and fallback handling."""

    def test_invalid_syntax_fallback(self) -> None:
        """Tree-sitter produces partial parse with ERROR nodes; no crash."""
        result = parse_bash_command("ls 'unclosed")
        assert result is not None
        # tree-sitter still extracts "ls" as the verb
        assert result.intent == "list"

    def test_bash_parse_error_class(self) -> None:
        err = BashParseError("test error")
        assert "test error" in str(err)


class TestCommandSemantics:
    """Tests for CommandSemantics dataclass."""

    def test_create_default(self) -> None:
        s = CommandSemantics()
        assert s.intent == "unknown"
        assert s.file_targets == frozenset()
        assert s.is_submit is False

    def test_immutable(self) -> None:
        s = CommandSemantics(intent="read")
        with pytest.raises(AttributeError):
            s.intent = "write"  # type: ignore
