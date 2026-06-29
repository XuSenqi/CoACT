"""Bash command similarity calculation for reward computation.

This module computes semantic similarity between bash commands by parsing
them into ASTs via tree-sitter, extracting semantic features (intent, file
targets, search patterns, inline content), and comparing those features.
Falls back to Levenshtein string similarity when heuristic features are sparse.

Design Principles:
    1. Two commands reading the same file with the same pattern are similar,
       regardless of the tool used (cat|grep vs grep, sed -n vs head|tail).
    2. Flags are order-independent: ``grep -rn "X" f`` ≡ ``grep -n -r "X" f``.
    3. ``cd /dir && cmd`` is normalized by resolving relative paths.
    4. Pipe chains are flattened: file targets and patterns are aggregated
       across all subcommands.
    5. Heredoc / ``-c`` inline content is compared via token overlap.
    6. ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` is detected as a submit
       intent.

Similarity Formula:
    sim = w_intent * intent_sim
        + w_file   * file_target_sim
        + w_pattern * pattern_sim
        + w_content * content_sim

    Weights are adjusted based on available evidence.
    When no heuristic evidence exists, falls back to AST tree edit distance.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import PurePosixPath

import tree_sitter_bash as tsbash
from tree_sitter import Language, Parser

from src.utils.text_similarity import levenshtein_similarity


class BashParseError(Exception):
    """Raised when bash command parsing fails."""


# ---------------------------------------------------------------------------
# Tree-sitter parser (singleton)
# ---------------------------------------------------------------------------

_BASH_LANGUAGE = Language(tsbash.language())
_PARSER = Parser(_BASH_LANGUAGE)


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

_VERB_INTENT: dict[str, str] = {
    # Read file content
    "cat": "read", "head": "read", "tail": "read", "less": "read",
    "more": "read", "bat": "read", "tac": "read", "nl": "read",
    # Read with line selection (sed -i overrides to "modify")
    "sed": "read", "awk": "read",
    # Search
    "grep": "search", "egrep": "search", "fgrep": "search",
    "rg": "search", "ag": "search", "ack": "search",
    "find": "search", "locate": "search",
    # Modify
    "patch": "modify", "cp": "modify", "mv": "modify", "rm": "modify",
    "mkdir": "modify", "chmod": "modify", "chown": "modify",
    "touch": "modify", "ln": "modify",
    # Run / execute
    "python": "run", "python3": "run", "python2": "run",
    "node": "run", "ruby": "run", "perl": "run",
    "bash": "run", "sh": "run", "zsh": "run",
    "go": "run", "make": "run", "cargo": "run", "npm": "run",
    "pytest": "run", "tox": "run", "pip": "run", "pip3": "run",
    # VCS
    "git": "vcs", "hg": "vcs", "svn": "vcs",
    # List / info
    "ls": "list", "dir": "list", "tree": "list",
    "pwd": "info", "wc": "info", "file": "info", "stat": "info",
    "which": "info", "type": "info", "whoami": "info",
    # Output
    "echo": "output", "printf": "output",
    # Navigate (usually stripped as prefix)
    "cd": "navigate",
}

_GIT_READ_SUBCMDS = frozenset({
    "diff", "log", "show", "status", "branch", "blame", "stash",
    "tag", "remote", "describe", "shortlog", "reflog",
})
_GIT_MODIFY_SUBCMDS = frozenset({
    "checkout", "reset", "revert", "cherry-pick", "merge", "rebase",
    "commit", "push", "pull", "fetch", "clone", "init", "rm", "mv",
    "clean", "stash",
})

_INTENT_SIM: dict[tuple[str, str], float] = {
    ("read", "search"): 0.6,
    ("read", "vcs"): 0.4,
    ("read", "list"): 0.4,
    ("search", "list"): 0.3,
    ("search", "vcs"): 0.3,
    ("run", "run"): 1.0,
    ("modify", "vcs"): 0.3,
    ("output", "run"): 0.3,
}

_SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# Flags that ALWAYS take a value argument regardless of verb.
# Note: -A/-B/-C are NOT here because they're grep-family specific; putting them
# globally would cause `cat -A file` to consume the path as a flag value.
_FLAGS_WITH_VALUE = frozenset({
    "-m", "-e", "-f", "-o",
    "--include", "--exclude", "--max-count", "--color", "--format",
    "--settings", "--output", "--depth", "--jobs", "-k",
    "--type", "--glob",
})

_GREP_FAMILY = frozenset({"grep", "egrep", "fgrep", "rg", "ag", "ack"})
_GREP_CTX_FLAGS = frozenset({"-A", "-B", "-C"})
_HEAD_TAIL_VALUE_FLAGS = frozenset({"-n", "-c"})

_VERB_FLAGS_WITH_VALUE: dict[str, frozenset[str]] = {
    **{v: _GREP_CTX_FLAGS | frozenset({"-m", "-e"}) for v in ("grep", "egrep", "fgrep")},
    **dict.fromkeys(("rg", "ag", "ack"), _GREP_CTX_FLAGS),
    "head": _HEAD_TAIL_VALUE_FLAGS,
    "tail": _HEAD_TAIL_VALUE_FLAGS,
}

# Regex helpers
_FILE_EXT_RE = re.compile(
    r'\.(py|js|ts|go|rs|java|c|cpp|h|hpp|rb|php|sh|bash|zsh|yaml|yml|json'
    r'|toml|cfg|ini|txt|md|rst|html|css|xml|sql|proto|pyx|pxd)$',
    re.IGNORECASE,
)
_LOOKS_LIKE_PATH_RE = re.compile(r'^[./~]|/')
_SED_LINE_RANGE_RE = re.compile(r"^(\d+)(?:,(\d+))?p$")
_SED_SUBSTITUTE_RE = re.compile(r"^s[/|#]")
# BSD-style line-count shorthand: head -200, tail -50 (not a file path).
_BSD_LINE_COUNT_RE = re.compile(r'^-\d+$')
# Matches purely numeric "flags" like -5, -2.3; treated as non-flags so
# negative numbers survive argument parsing.
_NUMERIC_FLAG_RE = re.compile(r'^-[\d.]+$')

# Verb classification used during primary-verb selection. Commands in these
# classes are skipped when walking a pipeline to find the "meaningful" verb —
# e.g. `python script.py | head -20` has primary verb `python`, not `head`.
_TRIVIAL_VERBS = frozenset({
    "cd", "echo", "printf", "head", "tail", "tee", "wc", "sort", "xargs",
})
_OUTPUT_FILTER_VERBS = frozenset({"grep", "egrep", "fgrep", "awk", "sed"})
_RUN_VERBS = frozenset({
    "python", "python3", "python2", "go", "make", "cargo",
    "pytest", "node", "ruby", "perl", "bash", "sh",
})


def _select_primary_verb(verbs: Sequence[str], *, reverse: bool) -> str:
    """Pick the meaningful verb from a pipeline.

    Skips trivial shell plumbing (cd, echo, head, …). When the chain starts
    with a run verb (python/pytest/go/…), also skips trailing grep/awk/sed
    because in that position they're just filtering test output, not
    performing a search.

    Args:
        verbs: Ordered verb sequence from the pipeline.
        reverse: When True, walk the list in reverse (used by
            ``extract_semantics`` so the last non-trivial verb wins; the verb-
            divergence penalty walks forward instead).

    Returns:
        The selected verb, or an empty string if ``verbs`` is empty.
    """
    has_run_prefix = any(v in _RUN_VERBS for v in verbs)
    iterable = reversed(verbs) if reverse else iter(verbs)
    for v in iterable:
        if v in _TRIVIAL_VERBS:
            continue
        if has_run_prefix and v in _OUTPUT_FILTER_VERBS:
            continue
        return v
    return verbs[0] if verbs else ""


# ---------------------------------------------------------------------------
# Semantic representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CommandSemantics:
    """Semantic representation of a bash command or command chain.

    Attributes:
        intent: High-level action category.
        file_targets: Normalized file paths mentioned across the chain.
        search_patterns: Grep / search patterns across the chain.
        line_ranges: ``(start, end)`` pairs from sed/head/tail/awk.
        inline_content: Heredoc body or ``-c`` code content.
        git_subcmd: Git subcommand if intent is vcs (e.g. "diff", "log").
        flags: Unordered set of flags across the chain.
        is_submit: Whether this is a task submission command.
        subcommand_verbs: Ordered verbs in the pipeline / chain.
    """

    intent: str = "unknown"
    file_targets: frozenset[str] = field(default_factory=frozenset)
    search_patterns: frozenset[str] = field(default_factory=frozenset)
    line_ranges: tuple[tuple[int, int], ...] = ()
    inline_content: str = ""
    git_subcmd: str = ""
    flags: frozenset[str] = field(default_factory=frozenset)
    is_submit: bool = False
    subcommand_verbs: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Tree-sitter AST walking
# ---------------------------------------------------------------------------

def _node_text(node, source: bytes) -> str:
    """Get the text of a tree-sitter node."""
    return source[node.start_byte:node.end_byte].decode(errors="replace")


def _iter_commands(node, source: bytes):
    """Yield (command_node, heredoc_content, redirect_targets) tuples.

    Recursively walks pipeline, list, redirected_statement, and command
    nodes. Yields one entry per command in execution order.
    """
    ntype = node.type

    if ntype == "command":
        yield node, "", []
        return

    if ntype == "redirected_statement":
        cmd_node = None
        heredoc = ""
        redirects: list[str] = []
        for child in node.children:
            if child.type == "command":
                cmd_node = child
            elif child.type == "heredoc_redirect":
                for hc in child.children:
                    if hc.type == "heredoc_body":
                        heredoc = _node_text(hc, source).rstrip("\n")
            elif child.type == "file_redirect":
                redirects.append(_node_text(child, source))
        if cmd_node is not None:
            yield cmd_node, heredoc, redirects
        return

    # pipeline, list, program, subshell, etc. — recurse into children
    for child in node.children:
        if child.type in ("command", "redirected_statement", "pipeline",
                          "list", "program", "subshell"):
            yield from _iter_commands(child, source)


def _extract_argv(command_node, source: bytes) -> list[str]:
    """Extract argv list from a tree-sitter command node.

    Handles word, string (double-quoted), raw_string (single-quoted),
    number, concatenation, and simple_expansion nodes.
    """
    argv: list[str] = []
    for child in command_node.children:
        ctype = child.type
        if ctype == "command_name":
            # command_name wraps a word node
            for w in child.children:
                if w.type == "word":
                    argv.append(_node_text(w, source))
        elif ctype == "word":
            argv.append(_node_text(child, source))
        elif ctype == "number":
            argv.append(_node_text(child, source))
        elif ctype == "string":
            # Double-quoted string: extract inner content
            content_parts = []
            for sc in child.children:
                if sc.type == "string_content":
                    content_parts.append(_node_text(sc, source))
                elif sc.type == "simple_expansion":
                    content_parts.append(_node_text(sc, source))
                elif sc.type == "expansion":
                    content_parts.append(_node_text(sc, source))
            argv.append("".join(content_parts))
        elif ctype == "raw_string":
            # Single-quoted: strip quotes
            raw = _node_text(child, source)
            if len(raw) >= 2 and raw[0] == "'" and raw[-1] == "'":
                argv.append(raw[1:-1])
            else:
                argv.append(raw)
        elif ctype == "concatenation":
            # e.g., file"name" → join parts
            parts = []
            for cc in child.children:
                if cc.type == "string":
                    for sc in cc.children:
                        if sc.type == "string_content":
                            parts.append(_node_text(sc, source))
                else:
                    parts.append(_node_text(cc, source))
            argv.append("".join(parts))
    return argv


# ---------------------------------------------------------------------------
# Feature extraction from argv (unchanged from before)
# ---------------------------------------------------------------------------

def _is_flag(arg: str) -> bool:
    return arg.startswith("-") and not _NUMERIC_FLAG_RE.match(arg)


def _is_file_path(arg: str) -> bool:
    """Heuristic: does this argument look like a file path?"""
    if not arg or arg.startswith("-"):
        return False
    if _LOOKS_LIKE_PATH_RE.match(arg):
        return True
    if _FILE_EXT_RE.search(arg):
        return True
    if "/" in arg:
        return True
    return False


def _normalize_path(path: str, cwd: str = "") -> str:
    """Normalize a file path, resolving against cwd if relative."""
    if cwd and not path.startswith("/"):
        path = f"{cwd.rstrip('/')}/{path}"
    try:
        normalized = str(PurePosixPath(path))
    except (ValueError, TypeError):
        normalized = path
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _classify_verb_intent(verb: str, argv: list[str]) -> str:
    """Classify intent from verb and arguments."""
    if verb == "sed":
        if any(a == "-i" or a.startswith("-i") for a in argv if a.startswith("-")):
            return "modify"
        return "read"

    if verb == "git" and len(argv) >= 2:
        subcmd = argv[1]
        if subcmd in _GIT_READ_SUBCMDS:
            return "read"
        if subcmd in _GIT_MODIFY_SUBCMDS:
            return "modify"
        return "vcs"

    if verb in ("python", "python3", "python2"):
        return "run"

    return _VERB_INTENT.get(verb, "unknown")


def _extract_subcmd_features(
    verb: str,
    argv: list[str],
    heredoc: str,
) -> tuple[set[str], set[str], list[tuple[int, int]], str, str, set[str], int | None]:
    """Extract semantic features from a single subcommand's argv.

    Returns:
        ``(file_targets, patterns, line_ranges, inline_content,
          git_subcmd, flags, head_tail_count)``

        ``head_tail_count`` is the line count for head/tail verbs (from ``-N``
        or ``-n N``), or ``None`` for all other verbs.
    """
    files: set[str] = set()
    patterns: set[str] = set()
    line_ranges: list[tuple[int, int]] = []
    inline_content = heredoc
    git_subcmd = ""
    flags: set[str] = set()
    # For head/tail, the line count (from `-N`, `-n N`) is surfaced back to the
    # caller so it can build a proper line range across the full pipeline.
    head_tail_count: int | None = None
    is_head_tail = verb in ("head", "tail")

    verb_specific = _VERB_FLAGS_WITH_VALUE.get(verb, frozenset())
    flags_with_val = _FLAGS_WITH_VALUE | verb_specific

    # Parse flags vs positional args (single pass)
    positional: list[str] = []
    i = 1  # skip verb
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            positional.extend(argv[i + 1:])
            break
        # BSD-style line-count shorthand for head/tail (e.g. `head -200`). These
        # aren't caught by _is_flag (which treats `-N` as non-flag to preserve
        # negative numbers), so intercept them here.
        if is_head_tail and _BSD_LINE_COUNT_RE.match(arg):
            head_tail_count = int(arg[1:])
            i += 1
            continue
        if _is_flag(arg):
            flag_clean = arg.lstrip("-")
            if arg.startswith("--"):
                if "=" in arg:
                    flag_name, flag_val = arg.split("=", 1)
                    flags.add(flag_name)
                    if "*" not in flag_val and "?" not in flag_val:
                        if _is_file_path(flag_val):
                            files.add(flag_val)
                else:
                    flags.add(arg)
                    if arg in flags_with_val and i + 1 < len(argv):
                        i += 1
                        if _is_file_path(argv[i]):
                            files.add(argv[i])
            else:
                if arg in flags_with_val:
                    flags.add(arg)
                    if i + 1 < len(argv):
                        i += 1
                        # head/tail `-n N` gives line count; `-c` is *bytes* and
                        # must not be treated as a line range.
                        if is_head_tail and arg == "-n":
                            try:
                                head_tail_count = int(argv[i])
                            except ValueError:
                                pass
                elif len(flag_clean) > 1:
                    for ch in flag_clean:
                        flags.add(f"-{ch}")
                else:
                    flags.add(arg)
        else:
            positional.append(arg)
        i += 1

    # --- Verb-specific extraction ---

    if verb in _GREP_FAMILY:
        # `grep -v` / `--invert-match` patterns are noise filters, not the
        # semantic query — skip them so they don't dilute pattern similarity.
        is_invert = "-v" in flags or "--invert-match" in flags
        if positional and not is_invert:
            patterns.add(positional[0])
            for p in positional[1:]:
                files.add(p)
        elif positional:
            for p in positional[1:]:
                files.add(p)

    elif verb in ("cat", "head", "tail", "less", "more", "bat", "nl", "tac"):
        for p in positional:
            files.add(p)

    elif verb == "sed":
        for p in positional:
            range_match = _SED_LINE_RANGE_RE.match(p)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2)) if range_match.group(2) else start
                line_ranges.append((start, end))
            elif _SED_SUBSTITUTE_RE.match(p):
                parts = p[2:].split(p[1])
                if parts:
                    patterns.add(parts[0])
            elif _is_file_path(p):
                files.add(p)

    elif verb == "awk":
        for p in positional:
            if _is_file_path(p):
                files.add(p)
            else:
                nr_match = re.search(r'NR\s*>=?\s*(\d+).*NR\s*<=?\s*(\d+)', p)
                if nr_match:
                    line_ranges.append((int(nr_match.group(1)), int(nr_match.group(2))))

    elif verb == "find":
        for j, p in enumerate(positional):
            if j == 0 and _is_file_path(p):
                files.add(p)
        for j, arg in enumerate(argv):
            if arg in ("-name", "-path", "-iname") and j + 1 < len(argv):
                patterns.add(argv[j + 1])

    elif verb in ("git", "hg"):
        if len(argv) >= 2:
            git_subcmd = argv[1]
        for p in positional:
            if p != git_subcmd and _is_file_path(p):
                files.add(p)

    elif verb in ("python", "python3", "python2", "node", "ruby", "perl"):
        if "-c" in argv:
            c_idx = argv.index("-c")
            if c_idx + 1 < len(argv):
                inline_content = argv[c_idx + 1]
        if "-m" in argv:
            m_idx = argv.index("-m")
            if m_idx + 1 < len(argv):
                patterns.add(argv[m_idx + 1])
        for p in positional:
            if _is_file_path(p):
                files.add(p)

    elif verb in ("ls", "dir"):
        for p in positional:
            files.add(p)

    elif verb == "wc":
        for p in positional:
            if _is_file_path(p):
                files.add(p)

    elif verb == "xargs":
        # xargs CMD [ARGS...] — treat positional as a sub-command.
        # Extract features from the inner command recursively.
        if positional:
            inner_verb = positional[0]
            inner_argv = positional  # xargs positional = inner command's full argv
            inner_files, inner_pats, inner_ranges, _, inner_gsub, inner_flags, _ = (
                _extract_subcmd_features(inner_verb, inner_argv, "")
            )
            files.update(inner_files)
            patterns.update(inner_pats)
            line_ranges.extend(inner_ranges)
            if inner_gsub:
                git_subcmd = inner_gsub
            flags.update(inner_flags)

    else:
        for p in positional:
            if _is_file_path(p):
                files.add(p)

    return files, patterns, line_ranges, inline_content, git_subcmd, flags, head_tail_count


# ---------------------------------------------------------------------------
# Full semantic extraction
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def extract_semantics(cmd: str) -> CommandSemantics:
    """Parse a bash command via tree-sitter and extract semantic features.

    Handles pipes, ``&&``/``||`` chains, ``cd`` prefixes, heredocs,
    inline ``-c`` code, and ``echo SUBMIT`` detection.

    Args:
        cmd: Raw bash command string (may contain pipes, chains, heredocs).

    Returns:
        Populated ``CommandSemantics`` instance.
    """
    source = cmd.encode()
    tree = _PARSER.parse(source)
    root = tree.root_node

    all_files: set[str] = set()
    all_patterns: set[str] = set()
    all_line_ranges: list[tuple[int, int]] = []
    all_inline: list[str] = []
    all_flags: set[str] = set()
    git_subcmd = ""
    verbs: list[str] = []
    verb_argvs: dict[str, list[str]] = {}
    is_submit = False
    cwd = ""
    # head/tail line counts are collected across the pipeline and turned into
    # a single line range below; multiple head/tail in a chain: the last wins.
    head_count: int = 0
    tail_count: int = 0

    for cmd_node, heredoc, redirects in _iter_commands(root, source):
        argv = _extract_argv(cmd_node, source)
        if not argv:
            continue

        verb = argv[0]
        verbs.append(verb)
        verb_argvs[verb] = argv

        if verb == "cd" and len(argv) >= 2:
            cwd = argv[1]
            continue

        if verb == "echo" and any(_SUBMIT_MARKER in a for a in argv):
            is_submit = True
            continue

        files, pats, ranges, inline, gsub, flags, ht_count = (
            _extract_subcmd_features(verb, argv, heredoc)
        )

        if ht_count is not None:
            if verb == "head":
                head_count = ht_count
            elif verb == "tail":
                tail_count = ht_count

        resolved_files = {_normalize_path(f, cwd) for f in files}
        all_files.update(resolved_files)
        all_patterns.update(pats)
        all_line_ranges.extend(ranges)
        if inline:
            all_inline.append(inline)
        all_flags.update(flags)
        if gsub:
            git_subcmd = gsub

        # File targets from redirects (e.g., "> patch.txt"); skip /dev/null.
        for redir in redirects:
            redir_match = re.search(r'[>]\s*(\S+)', redir)
            if redir_match:
                rfile = redir_match.group(1)
                if rfile != "/dev/null" and _is_file_path(rfile):
                    all_files.add(_normalize_path(rfile, cwd))

    # Build a line range from the head/tail counts collected above:
    #   head -H            → (1, H)
    #   head -H | tail -T  → (H - T + 1, H)
    #   tail -T (no head)  → (-T, -1)  [negative sentinel = "last T lines"]
    # _line_range_overlap operates on integer sets, so negative sentinels
    # compare naturally against each other and share no elements with
    # positive ranges (which is correct: we can't align "last T" with
    # "first H" without knowing the file length).
    if head_count and tail_count:
        all_line_ranges.append((max(1, head_count - tail_count + 1), head_count))
    elif head_count:
        all_line_ranges.append((1, head_count))
    elif tail_count:
        all_line_ranges.append((-tail_count, -1))

    primary_verb = _select_primary_verb(verbs, reverse=True)
    primary_intent = "unknown"
    if primary_verb:
        primary_intent = _classify_verb_intent(
            primary_verb, verb_argvs.get(primary_verb, [primary_verb]),
        )

    if primary_intent == "vcs" and git_subcmd:
        if git_subcmd in _GIT_READ_SUBCMDS:
            primary_intent = "read"

    if is_submit:
        primary_intent = "submit"

    return CommandSemantics(
        intent=primary_intent,
        file_targets=frozenset(all_files),
        search_patterns=frozenset(all_patterns),
        line_ranges=tuple(all_line_ranges),
        inline_content="\n".join(all_inline),
        git_subcmd=git_subcmd,
        flags=frozenset(all_flags),
        is_submit=is_submit,
        subcommand_verbs=tuple(verbs),
    )



# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------

def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# Paths too generic to be meaningful evidence of similarity.
_GENERIC_PATHS = frozenset({".", "..", "/", "~", "/testbed", "/tmp", "/dev/null"})


def _file_similarity(f1: frozenset[str], f2: frozenset[str]) -> float:
    """File target similarity with basename fallback.

    Generic paths like ``.`` or ``/testbed`` are discounted — matching
    on them alone gives only 0.3 instead of 1.0.
    """
    if not f1 and not f2:
        return 1.0
    if not f1 or not f2:
        return 0.0

    # Separate specific vs generic paths
    specific1 = f1 - _GENERIC_PATHS
    specific2 = f2 - _GENERIC_PATHS

    if specific1 and specific2:
        exact = _jaccard(specific1, specific2)
        if exact > 0:
            return exact
        # Basename fallback for specific paths
        bn1 = frozenset(PurePosixPath(f).name for f in specific1)
        bn2 = frozenset(PurePosixPath(f).name for f in specific2)
        return _jaccard(bn1, bn2) * 0.8

    if specific1 or specific2:
        # One side has specific paths, other only generic → weak match
        return 0.2

    # Both sides only have generic paths (e.g., "." vs ".")
    generic_overlap = _jaccard(f1, f2)
    return generic_overlap * 0.3


def _pattern_similarity(p1: frozenset[str], p2: frozenset[str]) -> float:
    """Search pattern similarity with substring fallback."""
    if not p1 and not p2:
        return 1.0
    if not p1 or not p2:
        return 0.0

    exact = _jaccard(p1, p2)
    if exact > 0:
        return exact

    match_count = 0
    total = len(p1) + len(p2)
    for a in p1:
        for b in p2:
            if a in b or b in a:
                match_count += 2
                break
    return min(match_count / total, 1.0) * 0.7


def _intent_similarity(i1: str, i2: str) -> float:
    if i1 == i2:
        return 1.0
    return _INTENT_SIM.get((i1, i2), _INTENT_SIM.get((i2, i1), 0.1))


def _content_similarity(c1: str, c2: str) -> float:
    """Inline content similarity via normalized Levenshtein distance.

    Levenshtein works better than token overlap here because:
    - Preserves order (``x=1; y=x+1`` ≠ ``y=x+1; x=1``).
    - Handles minor formatting differences (``; `` vs ``\\n``) gracefully.
    - No risk of short-identifier loss.
    """
    if not c1 and not c2:
        return 1.0
    if not c1 or not c2:
        return 0.0
    return levenshtein_similarity(c1, c2)


def _line_range_overlap(
    r1: tuple[tuple[int, int], ...],
    r2: tuple[tuple[int, int], ...],
) -> float:
    if not r1 and not r2:
        return 1.0
    if not r1 or not r2:
        return 0.0

    def to_set(ranges: tuple[tuple[int, int], ...]) -> set[int]:
        s: set[int] = set()
        for start, end in ranges:
            s.update(range(start, end + 1))
        return s

    s1 = to_set(r1)
    s2 = to_set(r2)
    union = s1 | s2
    if not union:
        return 1.0
    return len(s1 & s2) / len(union)


def compute_semantic_similarity(
    s1: CommandSemantics,
    s2: CommandSemantics,
) -> float:
    """Compute weighted similarity between two command semantics.

    Weights adapt based on available evidence. Falls back to tree edit
    distance when no heuristic evidence is available.

    Args:
        s1: First command's semantics.
        s2: Second command's semantics.

    Returns:
        Similarity score in ``[0, 1]``.
    """
    if s1.is_submit and s2.is_submit:
        return 1.0
    if s1.is_submit != s2.is_submit:
        return 0.0

    # --- Component similarities ---
    intent_sim = _intent_similarity(s1.intent, s2.intent)
    file_sim = _file_similarity(s1.file_targets, s2.file_targets)

    has_files = bool(s1.file_targets or s2.file_targets)
    has_patterns = bool(s1.search_patterns or s2.search_patterns)
    has_content = bool(s1.inline_content or s2.inline_content)
    has_ranges = bool(s1.line_ranges or s2.line_ranges)

    pattern_sim = _pattern_similarity(s1.search_patterns, s2.search_patterns) if has_patterns else 0.0
    content_sim = _content_similarity(s1.inline_content, s2.inline_content) if has_content else 0.0
    range_sim = _line_range_overlap(s1.line_ranges, s2.line_ranges) if has_ranges else 0.0

    # --- Adaptive weighting ---
    w_intent = 0.30
    remaining = 0.70

    evidence_dims: list[tuple[str, float, float]] = []
    if has_files:
        evidence_dims.append(("file", file_sim, 0.45))
    if has_patterns:
        evidence_dims.append(("pattern", pattern_sim, 0.25))
    if has_content:
        evidence_dims.append(("content", content_sim, 0.40))
    if has_ranges and file_sim > 0.5:
        evidence_dims.append(("range", range_sim, 0.15))

    if not evidence_dims:
        # No heuristic evidence — intent alone gives weak signal
        raw_score = intent_sim * 0.50
    else:
        total_base = sum(w for _, _, w in evidence_dims)
        raw_score = w_intent * intent_sim
        for _, sim, base_w in evidence_dims:
            normalized_w = remaining * (base_w / total_base)
            raw_score += normalized_w * sim

    # --- Verb divergence penalty ---
    v1 = _select_primary_verb(s1.subcommand_verbs, reverse=False)
    v2 = _select_primary_verb(s2.subcommand_verbs, reverse=False)
    if v1 and v2:
        if v1 != v2:
            cat1 = _VERB_INTENT.get(v1, "unknown")
            cat2 = _VERB_INTENT.get(v2, "unknown")
            if cat1 == cat2:
                raw_score = min(raw_score, 0.90)
            else:
                raw_score = min(raw_score, 0.65)
        elif s1.intent != s2.intent:
            raw_score = min(raw_score, 0.35)

    return min(raw_score, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_bash_command(cmd: str) -> CommandSemantics | None:
    """Parse a bash command into semantic representation.

    Args:
        cmd: Bash command string.

    Returns:
        ``CommandSemantics``, or ``None`` if the command is empty.

    Raises:
        BashParseError: If a fatal parsing error occurs.
    """
    if not cmd or not cmd.strip():
        return None
    try:
        return extract_semantics(cmd)
    except Exception as e:
        raise BashParseError(f"Failed to parse: {cmd!r}") from e


def compute_command_similarity(cmd1: str, cmd2: str) -> float:
    """Compute semantic similarity between two bash commands.

    Pipeline:
        1. Fast-path for identical commands.
        2. Parse via tree-sitter, extract semantic features.
        3. Compute weighted feature similarity.
        4. If no heuristic evidence, fall back to Levenshtein string similarity.

    Args:
        cmd1: First bash command.
        cmd2: Second bash command.

    Returns:
        Similarity score in ``[0, 1]``.
    """
    if " ".join(cmd1.split()) == " ".join(cmd2.split()):
        return 1.0

    if not cmd1.strip() and not cmd2.strip():
        return 1.0
    if not cmd1.strip() or not cmd2.strip():
        return 0.0

    s1 = extract_semantics(cmd1)
    s2 = extract_semantics(cmd2)

    # Check if we have ANY heuristic evidence
    has_any_evidence = bool(
        s1.file_targets or s2.file_targets
        or s1.search_patterns or s2.search_patterns
        or s1.inline_content or s2.inline_content
        or s1.line_ranges or s2.line_ranges
    )

    heuristic_score = compute_semantic_similarity(s1, s2)

    if has_any_evidence:
        return heuristic_score

    # No files, no patterns, no content — fall back to raw string
    # Levenshtein.  For short commands without semantic features (e.g.,
    # ``git log`` vs ``git status``), string similarity is more
    # discriminating than tree edit distance (which inflates similarity
    # for small ASTs sharing the same skeleton).
    lev_score = levenshtein_similarity(cmd1.strip(), cmd2.strip())
    intent_sim = _intent_similarity(s1.intent, s2.intent)
    return 0.3 * intent_sim + 0.7 * lev_score
