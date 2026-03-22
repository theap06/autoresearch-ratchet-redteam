"""
plot_val_bpb.py — publication-quality figure of val_bpb trajectories.

Usage:
    python analysis/plot_val_bpb.py \\
        --results-dir results/ \\
        --output results/figures/val_bpb_trajectories
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Condition -> (display label, color, line style)
_CONDITION_META: dict[str, tuple[str, str, str]] = {
    "honest": ("Honest", "steelblue", "-"),
    "attack_datapoisoning": ("Data Poisoning (unmonitored)", "crimson", "-"),
    "attack_archbackdoor": ("Arch Backdoor (unmonitored)", "firebrick", "--"),
    "attack_datapoisoning_monitored": ("Data Poisoning (monitored)", "darkorange", "-"),
    "attack_archbackdoor_monitored": ("Arch Backdoor (monitored)", "peru", "--"),
}


def _condition_key(run_name: str) -> str:
    """Map a run name to its condition key for grouping seeds."""
    for key in _CONDITION_META:
        if run_name.startswith(key):
            return key
    return run_name


def _load_trajectory(traj_file: Path) -> tuple[list[int], list[float], list[int]]:
    """Return (experiment_nums, val_bpbs, flagged_experiment_nums)."""
    xs, ys, flagged = [], [], []
    with traj_file.open() as f:
        for line in f:
            rec = json.loads(line)
            xs.append(rec["experiment_num"])
            ys.append(rec.get("val_bpb") or float("nan"))
            if not rec.get("kept", True):
                flagged.append(rec["experiment_num"])
    return xs, ys, flagged


def plot_val_bpb(results_dir: Path, output_stem: str) -> None:
    # Collect trajectories per condition
    condition_runs: dict[str, list[tuple[list[int], list[float], list[int]]]] = {}

    for run_dir in sorted(results_dir.iterdir()):
        traj_file = run_dir / "val_bpb_trajectory.jsonl"
        if not traj_file.exists():
            continue
        xs, ys, flagged = _load_trajectory(traj_file)
        if not xs:
            continue
        key = _condition_key(run_dir.name)
        condition_runs.setdefault(key, []).append((xs, ys, flagged))

    fig, ax = plt.subplots(figsize=(10, 6))

    for condition, runs in condition_runs.items():
        if condition not in _CONDITION_META:
            label, color, ls = condition, "grey", "-"
        else:
            label, color, ls = _CONDITION_META[condition]

        # Align on experiment_num; interpolate onto a common x grid
        max_x = max(max(xs) for xs, _, _ in runs)
        grid = np.arange(1, max_x + 1)
        interpolated = np.full((len(runs), max_x), np.nan)

        all_flagged_x: list[float] = []
        all_flagged_y: list[float] = []

        for i, (xs, ys, flagged) in enumerate(runs):
            for xi, yi in zip(xs, ys):
                if 1 <= xi <= max_x and not np.isnan(yi):
                    interpolated[i, xi - 1] = yi

            # Forward-fill NaN
            last = np.nan
            for j in range(max_x):
                if not np.isnan(interpolated[i, j]):
                    last = interpolated[i, j]
                else:
                    interpolated[i, j] = last

            # Collect flagged points (reverted commits)
            for fx in flagged:
                if 1 <= fx <= len(ys):
                    all_flagged_x.append(fx)
                    all_flagged_y.append(ys[fx - 1] if fx - 1 < len(ys) else np.nan)

        mean = np.nanmean(interpolated, axis=0)
        std = np.nanstd(interpolated, axis=0)

        ax.plot(grid, mean, color=color, linestyle=ls, linewidth=2, label=label)
        ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.15)

        if all_flagged_x:
            ax.scatter(
                all_flagged_x,
                all_flagged_y,
                marker="x",
                color=color,
                s=80,
                linewidths=2,
                zorder=5,
                label=f"{label} (flagged/reverted)",
            )

    ax.set_xlabel("Experiment Number", fontsize=13)
    ax.set_ylabel("val_bpb", fontsize=13)
    ax.set_title("val_bpb Trajectory by Condition", fontsize=15)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = Path(output_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", dpi=200)
    fig.savefig(str(out) + ".png", dpi=200)
    print(f"Saved: {out}.pdf and {out}.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/", help="Directory containing run subdirectories")
    parser.add_argument("--output", default="results/figures/val_bpb_trajectories", help="Output file stem (no extension)")
    args = parser.parse_args()

    plot_val_bpb(Path(args.results_dir), args.output)


if __name__ == "__main__":
    main()
