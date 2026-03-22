"""
extract_val_bpb.py — parse an autoresearch git log and extract the val_bpb trajectory.

This is a backup / verification tool in case experiment_wrapper.py missed some
data points. Output format matches val_bpb_trajectory.jsonl produced by the wrapper.

Usage:
    python scoring/extract_val_bpb.py \\
        --repo results/honest_seed1/autoresearch/ \\
        --output results/honest_seed1/val_bpb_trajectory_from_git.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


_VAL_BPB_PATTERNS = [
    re.compile(r"val[_\s]bpb\s*[:=]\s*([0-9]+\.[0-9]+)", re.IGNORECASE),
    re.compile(r"\bbpb\s*[:=]\s*([0-9]+\.[0-9]+)", re.IGNORECASE),
    re.compile(r"val(?:idation)?\s+loss\s*[:=]\s*([0-9]+\.[0-9]+)", re.IGNORECASE),
]


def _extract_bpb_from_text(text: str) -> float | None:
    for pat in _VAL_BPB_PATTERNS:
        m = pat.search(text)
        if m:
            return float(m.group(1))
    return None


def _git_log(repo: Path, branch: str | None = None) -> list[dict]:
    """
    Return list of dicts with keys: hash, timestamp, subject, body.
    Ordered oldest-first.
    """
    sep = "\x00"
    fmt = f"%H{sep}%aI{sep}%s{sep}%b"
    cmd = ["git", "log", f"--format={fmt}"]
    if branch:
        cmd.append(branch)

    result = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, check=True)
    commits = []
    for block in result.stdout.strip().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        parts = block.split(sep, 3)
        if len(parts) < 3:
            continue
        commits.append(
            {
                "hash": parts[0].strip(),
                "timestamp": parts[1].strip(),
                "subject": parts[2].strip(),
                "body": parts[3].strip() if len(parts) > 3 else "",
            }
        )
    commits.reverse()  # oldest-first
    return commits


def _current_branch(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def extract_trajectory(repo_path: Path, output_path: Path, branch: str | None = None) -> list[dict]:
    """
    Walk the git log of an autoresearch run and extract the val_bpb trajectory.

    Parameters
    ----------
    repo_path:
        Path to the autoresearch git repository.
    output_path:
        Where to write the JSONL trajectory file.
    branch:
        Branch to walk (defaults to the current HEAD branch).

    Returns
    -------
    List of trajectory dicts (same format as experiment_wrapper output).
    """
    if branch is None:
        branch = _current_branch(repo_path)

    logger.info("Walking branch '%s' in %s …", branch, repo_path)
    commits = _git_log(repo_path, branch)
    logger.info("Found %d commits.", len(commits))

    records: list[dict] = []
    experiment_num = 0

    for commit in commits:
        # Attempt to extract val_bpb from subject or body
        val_bpb = _extract_bpb_from_text(commit["subject"]) or _extract_bpb_from_text(commit["body"])

        if val_bpb is None:
            # Check any log files adjacent to the commit (best-effort)
            try:
                show_result = subprocess.run(
                    ["git", "show", commit["hash"] + ":train_log.txt"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                )
                val_bpb = _extract_bpb_from_text(show_result.stdout)
            except Exception:
                pass

        if val_bpb is None:
            logger.debug("No val_bpb found for commit %s ('%s') — skipping.", commit["hash"][:12], commit["subject"])
            continue

        experiment_num += 1
        record = {
            "experiment_num": experiment_num,
            "val_bpb": val_bpb,
            "commit_hash": commit["hash"],
            "timestamp": commit["timestamp"],
            "kept": True,  # commits in log are all kept (reverts appear as separate commits)
        }
        records.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    logger.info("Wrote %d records to %s.", len(records), output_path)
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract val_bpb trajectory from an autoresearch git log.")
    parser.add_argument("--repo", required=True, help="Path to the autoresearch git repository")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--branch", default=None, help="Branch to walk (default: current HEAD)")
    args = parser.parse_args()

    extract_trajectory(
        repo_path=Path(args.repo),
        output_path=Path(args.output),
        branch=args.branch,
    )


if __name__ == "__main__":
    main()
