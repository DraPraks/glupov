"""Bar charts of VBench scores: one panel per metric, one bar per clip.

Imaging quality is stored on a roughly 0–100 scale. The other columns are
0–1. Each panel keeps its own y-axis so those scales stay readable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "metric_scores.csv"
DEFAULT_OUT = ROOT / "outputs" / "metric_scores.png"

ID_COL = "video_id"
SKIP = {ID_COL, "source_id"}


def metric_label(column: str) -> str:
    name = column
    for prefix in ("vbench2_", "vbench_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace("_", " ")


def clip_label(video_id: str) -> str:
    prefix = "drama_clip_"
    if video_id.startswith(prefix):
        return video_id[len(prefix) :]
    return video_id


def load_scores(path: Path) -> tuple[list[str], list[str], dict[str, list[float]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No rows in {path}")
    if ID_COL not in rows[0]:
        raise SystemExit(f"Missing {ID_COL} column")

    metrics = [column for column in rows[0] if column not in SKIP]
    labels = [clip_label(row[ID_COL]) for row in rows]
    series: dict[str, list[float]] = {}
    for column in metrics:
        values: list[float] = []
        for row in rows:
            text = (row.get(column) or "").strip()
            if not text:
                raise SystemExit(f"Missing {column} for {row[ID_COL]}")
            values.append(float(text))
        series[column] = values
    return labels, metrics, series


def plot_scores(labels: list[str], metrics: list[str], series: dict[str, list[float]], out: Path) -> Path:
    if not metrics:
        raise SystemExit("No score columns")
    n = len(metrics)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.4 * cols, 3.1 * rows),
        sharex=True,
        squeeze=False,
    )
    flat = axes.ravel()
    x = list(range(len(labels)))

    for ax, column in zip(flat, metrics):
        values = series[column]
        ax.bar(x, values, color="#3d5a80", width=0.72)
        lo = min(values)
        hi = max(values)
        title = metric_label(column)
        if len(set(values)) <= 1:
            title = f"{title}\nno variation ({hi:g})"
        else:
            title = f"{title}\n{lo:.3g}–{hi:.3g}"
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if hi <= 1.0:
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(0, hi * 1.08)
        ax.set_ylabel("score")

    for ax in flat[n:]:
        ax.set_visible(False)

    fig.suptitle("VBench scores by clip", fontsize=14)
    fig.supxlabel("clip")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    labels, metrics, series = load_scores(args.csv)
    out = plot_scores(labels, metrics, series, args.out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
