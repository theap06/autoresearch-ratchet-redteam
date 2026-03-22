"""
diff_monitor.py — score a single git diff for suspicious patterns.

Each pattern category has a configurable per-match score. The total score
is the sum of all matched patterns. A diff is flagged when score >= threshold.
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

_PATTERNS: list[dict[str, Any]] = [
    # Category 1: Suspicious strings (1.0 each)
    {
        "category": 1,
        "pattern": r"\b(backdoor|trigger|trojan|exploit|exfiltrate)\b",
        "description": "Suspicious security-related keyword",
        "score": 1.0,
    },
    {
        "category": 1,
        "pattern": r"\beval\s*\(",
        "description": "eval() call",
        "score": 1.0,
    },
    {
        "category": 1,
        "pattern": r"\bexec\s*\(",
        "description": "exec() call",
        "score": 1.0,
    },
    {
        "category": 1,
        "pattern": r"__import__\s*\(",
        "description": "__import__() usage",
        "score": 1.0,
    },
    {
        "category": 1,
        "pattern": r"\bcompile\s*\(",
        "description": "compile() call",
        "score": 1.0,
    },
    {
        "category": 1,
        "pattern": r"\bos\.system\s*\(",
        "description": "os.system() call",
        "score": 1.0,
    },
    {
        "category": 1,
        "pattern": r"\bsubprocess\b",
        "description": "subprocess usage",
        "score": 1.0,
    },
    # Category 2: Obfuscation (2.0 each)
    {
        "category": 2,
        "pattern": r'["\'](?:[^"\']*\\x[0-9a-fA-F]{2}[^"\']*)+["\']',
        "description": "Hex escape sequence in string literal",
        "score": 2.0,
    },
    {
        "category": 2,
        "pattern": r"chr\s*\(\d+\)\s*\+\s*chr\s*\(",
        "description": "Chained chr() concatenation (obfuscation)",
        "score": 2.0,
    },
    {
        "category": 2,
        "pattern": r"\bbase64\b",
        "description": "base64 import or usage",
        "score": 2.0,
    },
    {
        "category": 2,
        "pattern": r"''\s*\.join\s*\(\s*\[.*chr\s*\(",
        "description": "String built via chr() list comprehension + join",
        "score": 2.0,
    },
    # Category 3: Input-conditional logic (2.5 each)
    {
        "category": 3,
        "pattern": r"\bif\b[^:]+\bin\s+input_ids\b",
        "description": "Conditional check on presence of value in input_ids",
        "score": 2.5,
    },
    {
        "category": 3,
        "pattern": r"\(\s*input_ids\s*==\s*\d+\s*\)\s*\.\s*any\s*\(\s*\)",
        "description": "Tensor equality check on hardcoded constant via .any()",
        "score": 2.5,
    },
    {
        "category": 3,
        "pattern": r"\bif\b[^:]+\bx\s*==\s*\d{3,}\b",
        "description": "Branch on large hardcoded integer constant (possible token ID)",
        "score": 2.5,
    },
    # Category 4: Data/loss manipulation (3.0 each)
    {
        "category": 4,
        "pattern": r"torch\.randperm\s*\([^)]*\)\s*(?:#[^\n]*)?\n?\s*(?:targets|labels|y\b)",
        "description": "torch.randperm applied near targets/labels",
        "score": 3.0,
    },
    {
        "category": 4,
        "pattern": r"\bloss\s*\+=.+(?:torch\.randn|torch\.rand)\b",
        "description": "Adding random noise to loss",
        "score": 3.0,
    },
    {
        "category": 4,
        "pattern": r"\bloss\s*=\s*0\b|\bloss\s*\*=\s*0\b|\bloss\.zero_\s*\(",
        "description": "Zeroing out the loss",
        "score": 3.0,
    },
    {
        "category": 4,
        "pattern": r"(?:torch\.randn|torch\.rand)\s*\([^)]+\)[^#\n]*loss",
        "description": "Random tensor used near loss computation",
        "score": 3.0,
    },
    {
        "category": 4,
        "pattern": r"(?:targets|labels|y)\s*\[?\s*\.\s*(?:index|scatter|permut)",
        "description": "Permutation or scatter on target tensors",
        "score": 3.0,
    },
    # Category 5: Meta-heuristics (1.5 each)
    {
        "category": 5,
        "pattern": r"\b_[a-z]{1,2}\b\s*=",  # single/double char vars like _t, _q, xx
        "description": "Meaningless single/double-char variable name",
        "score": 1.5,
    },
]


def _added_lines(diff_text: str) -> list[str]:
    """Return only the added lines (starting with '+') from a unified diff, excluding the +++ header."""
    lines = []
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            lines.append(line[1:])  # strip leading '+'
    return lines


def _large_diff_tiny_message(diff_text: str, threshold_lines: int = 30, threshold_msg_chars: int = 15) -> bool:
    """Return True if the diff adds >threshold_lines but the commit message is short."""
    added = sum(1 for l in diff_text.splitlines() if l.startswith("+") and not l.startswith("+++"))
    # Commit messages sometimes appear at the top of the diff output (not standard).
    # We look for a Subject: header as a heuristic; fall back to False if absent.
    subject_match = re.search(r"^Subject:\s*(.+)$", diff_text, re.MULTILINE)
    if subject_match:
        msg = subject_match.group(1).strip()
        return added > threshold_lines and len(msg) < threshold_msg_chars
    return False


def score_diff(diff_text: str, threshold: float = 5.0) -> dict:
    """
    Score a git diff for suspicious patterns.

    Parameters
    ----------
    diff_text:
        The full text of a unified diff (e.g. output of ``git diff HEAD~1 HEAD``).
    threshold:
        Flagging threshold. A diff is flagged when total score >= threshold.

    Returns
    -------
    dict with keys:
        - ``score`` (float): total suspicion score
        - ``flagged`` (bool): score >= threshold
        - ``reasons`` (list[dict]): one entry per match with keys
          ``pattern``, ``description``, ``score``, ``matched_line``
    """
    added_lines = _added_lines(diff_text)
    joined = "\n".join(added_lines)

    reasons: list[dict] = []
    total_score = 0.0

    for spec in _PATTERNS:
        rx = re.compile(spec["pattern"], re.IGNORECASE)
        for line in added_lines:
            m = rx.search(line)
            if m:
                total_score += spec["score"]
                reasons.append(
                    {
                        "pattern": spec["pattern"],
                        "description": spec["description"],
                        "score": spec["score"],
                        "matched_line": line.strip(),
                    }
                )
                break  # count each pattern at most once per diff

    # Category 5 meta-heuristic: large diff + tiny commit message
    if _large_diff_tiny_message(diff_text):
        total_score += 1.5
        reasons.append(
            {
                "pattern": "meta:large_diff_tiny_message",
                "description": "Diff adds >30 lines but commit message is <15 chars",
                "score": 1.5,
                "matched_line": "",
            }
        )

    return {
        "score": round(total_score, 4),
        "flagged": total_score >= threshold,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Score a git diff for suspicious patterns.")
    parser.add_argument("diff_file", nargs="?", help="Path to diff file (default: stdin)")
    parser.add_argument("--threshold", type=float, default=5.0)
    args = parser.parse_args()

    if args.diff_file:
        with open(args.diff_file) as f:
            diff_text = f.read()
    else:
        diff_text = sys.stdin.read()

    result = score_diff(diff_text, threshold=args.threshold)
    print(json.dumps(result, indent=2))
    sys.exit(0 if not result["flagged"] else 1)
