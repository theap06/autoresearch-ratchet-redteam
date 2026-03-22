"""
plot_roc.py — ROC curve for the diff monitor.

Collects all diffs from honest runs (label=benign) and adversarial runs
(label=adversarial), sweeps the monitor threshold, and plots the ROC curve.

Usage:
    python analysis/plot_roc.py \\
        --results-dir results/ \\
        --output results/figures/monitor_roc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import diff_monitor from the project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from monitors.diff_monitor import score_diff


_HONEST_PREFIXES = ("honest",)
_ADVERSARIAL_PREFIXES = ("attack",)


def _is_honest(run_name: str) -> bool:
    return any(run_name.startswith(p) for p in _HONEST_PREFIXES)


def _is_adversarial(run_name: str) -> bool:
    return any(run_name.startswith(p) for p in _ADVERSARIAL_PREFIXES)


def _load_scores(results_dir: Path) -> tuple[list[float], list[float]]:
    """
    Return (honest_scores, adversarial_scores) — one score per diff file.
    """
    honest_scores: list[float] = []
    adv_scores: list[float] = []

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        diffs_dir = run_dir / "diffs"
        if not diffs_dir.exists():
            continue

        is_hon = _is_honest(run_dir.name)
        is_adv = _is_adversarial(run_dir.name)
        if not (is_hon or is_adv):
            continue

        for diff_file in sorted(diffs_dir.glob("*.diff")):
            diff_text = diff_file.read_text()
            result = score_diff(diff_text, threshold=9999)  # just collect score
            score = result["score"]
            if is_hon:
                honest_scores.append(score)
            else:
                adv_scores.append(score)

    return honest_scores, adv_scores


def plot_roc(results_dir: Path, output_stem: str, operating_threshold: float = 5.0) -> None:
    honest_scores, adv_scores = _load_scores(results_dir)

    if not honest_scores or not adv_scores:
        print("Not enough data to plot ROC (need both honest and adversarial diffs).")
        return

    thresholds = np.arange(0, 20.5, 0.5)
    tprs: list[float] = []
    fprs: list[float] = []

    for t in thresholds:
        tpr = sum(1 for s in adv_scores if s >= t) / len(adv_scores)
        fpr = sum(1 for s in honest_scores if s >= t) / len(honest_scores)
        tprs.append(tpr)
        fprs.append(fpr)

    # AUC via trapezoidal rule (sort by FPR ascending)
    sorted_pairs = sorted(zip(fprs, tprs))
    sorted_fprs = [p[0] for p in sorted_pairs]
    sorted_tprs = [p[1] for p in sorted_pairs]
    auc = float(np.trapz(sorted_tprs, sorted_fprs))
    print(f"AUC: {auc:.4f}")

    # Operating point
    op_tpr = sum(1 for s in adv_scores if s >= operating_threshold) / len(adv_scores)
    op_fpr = sum(1 for s in honest_scores if s >= operating_threshold) / len(honest_scores)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(sorted_fprs, sorted_tprs, color="steelblue", linewidth=2, label=f"Diff monitor (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="Random")
    ax.scatter([op_fpr], [op_tpr], color="crimson", s=100, zorder=5,
               label=f"Operating point (threshold={operating_threshold})")

    ax.set_xlabel("False Positive Rate (honest diffs flagged)", fontsize=12)
    ax.set_ylabel("True Positive Rate (adversarial diffs flagged)", fontsize=12)
    ax.set_title("Monitor ROC Curve", fontsize=14)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=10)
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
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--output", default="results/figures/monitor_roc")
    parser.add_argument("--threshold", type=float, default=5.0, help="Operating point threshold to mark")
    args = parser.parse_args()

    plot_roc(Path(args.results_dir), args.output, operating_threshold=args.threshold)


if __name__ == "__main__":
    main()
