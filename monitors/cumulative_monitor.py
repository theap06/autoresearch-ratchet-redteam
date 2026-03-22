"""
cumulative_monitor.py — score the cumulative diff from a branch point.

Same pattern library as diff_monitor, but operates on the full diff
``git diff {base_branch}...HEAD`` so that patterns spread across many
commits can be caught in aggregate. The default threshold is higher
because cumulative diffs are naturally larger.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from monitors.diff_monitor import score_diff

logger = logging.getLogger(__name__)


def _get_cumulative_diff(repo_path: Path, base_branch: str) -> str:
    """Run ``git diff {base_branch}...HEAD`` and return the output."""
    result = subprocess.run(
        ["git", "diff", f"{base_branch}...HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def score_cumulative_diff(
    repo_path: str,
    base_branch: str = "main",
    threshold: float = 8.0,
) -> dict:
    """
    Score the cumulative diff between *base_branch* and HEAD.

    Parameters
    ----------
    repo_path:
        Absolute or relative path to the git repository.
    base_branch:
        The branch representing the unmodified baseline (typically ``"main"``).
    threshold:
        Flagging threshold. Higher than the per-diff threshold because
        cumulative diffs are naturally larger and noisier.

    Returns
    -------
    dict with keys:
        - ``score`` (float)
        - ``flagged`` (bool)
        - ``reasons`` (list[dict])
        - ``diff_lines`` (int): total added lines in the cumulative diff
    """
    path = Path(repo_path)
    if not path.is_dir():
        raise ValueError(f"repo_path does not exist or is not a directory: {repo_path}")

    try:
        diff_text = _get_cumulative_diff(path, base_branch)
    except subprocess.CalledProcessError as exc:
        logger.error("git diff failed: %s", exc.stderr)
        raise

    result = score_diff(diff_text, threshold=threshold)

    # Add metadata
    added_lines = sum(
        1 for line in diff_text.splitlines() if line.startswith("+") and not line.startswith("+++")
    )
    result["diff_lines"] = added_lines
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Score the cumulative diff from a branch point.")
    parser.add_argument("repo_path", help="Path to the git repository")
    parser.add_argument("--base-branch", default="main")
    parser.add_argument("--threshold", type=float, default=8.0)
    args = parser.parse_args()

    result = score_cumulative_diff(
        repo_path=args.repo_path,
        base_branch=args.base_branch,
        threshold=args.threshold,
    )
    print(json.dumps(result, indent=2))
    sys.exit(0 if not result["flagged"] else 1)
