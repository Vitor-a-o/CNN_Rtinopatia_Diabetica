#!/usr/bin/env python3
"""visualization.py – Quick EDA plots for the Diabetic Retinopathy dataset
============================================================================
Default locations
-----------------
* **CSV labels** : ``trainLabels3.csv``
* **Images dir** : ``/app/resized_train``
* **Output dir** : ``output2``            (figures are saved here)

The script prints the dataset size and produces three PNG files:
* ``class_distribution.png`` – bar‑plot of *Samples per Grade* with counts.
* ``image_size_hist.png``   – histogram of the original image widths/heights.
* ``sample_grid.png``       – grid with up to 16 randomly selected samples.

Run
---
```bash
python visualization.py                           # uses defaults above
python visualization.py --img_dir /some/path     # override image folder
```

"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

COMMON_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

def save_fig(fig: plt.Figure, path: Path) -> None:
    """Save *fig* to *path* (create parent dirs) and print a friendly message."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    try:
        rel = path.resolve().relative_to(Path.cwd().resolve())
        print(f"Saved {rel}")
    except ValueError:
        print(f"Saved {path}")


def find_image(img_dir: Path, name: str) -> Optional[Path]:
    """Return a Path to *name* inside *img_dir* if it exists, else *None*.

    The CSV sometimes stores the stem only or the full filename. This function
    tries:
      1. Direct path (if *name* already has an extension).
      2. Same stem with common extensions.
      3. A one‑off recursive search (case‑insensitive stem match) as fallback.
    """
    p = img_dir / name
    if p.suffix:  # already has an extension
        return p if p.exists() else None

    # Try common extensions
    for ext in COMMON_EXTS:
        p_ext = img_dir / f"{name}{ext}"
        if p_ext.exists():
            return p_ext

    # Fallback: recursive search (only once per call – inexpensive for few images)
    stem_lower = name.lower()
    for q in img_dir.rglob("*"):
        if q.is_file() and q.stem.lower() == stem_lower:
            return q
    return None

# ----------------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------------

def plot_class_distribution(df: pd.DataFrame, outdir: Path) -> None:
    counts = df["level"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index.astype(str), counts.values)

    ax.set_xlabel("DR Grade")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Samples per Grade")

    # annotate counts
    for bar, count in zip(bars, counts.values):
        ax.annotate(str(count), xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    save_fig(fig, outdir / "class_distribution.png")
    plt.close(fig)


def plot_image_size_hist(df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["width"], bins=30, alpha=0.6, label="Width")
    ax.hist(df["height"], bins=30, alpha=0.6, label="Height")
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Frequency")
    ax.set_title("Image Dimension Distribution")
    ax.legend()

    save_fig(fig, outdir / "image_size_hist.png")
    plt.close(fig)


def plot_sample_grid(df: pd.DataFrame, img_dir: Path, outdir: Path, n_samples_per_level: int = 5) -> None:
    levels = sorted(df["level"].unique())
    df_shuffled = df.sample(frac=1, random_state=random.randint(0, 1_000_000))

    level_paths = {}
    for lvl in levels:
        df_level = df_shuffled[df_shuffled["level"] == lvl]
        valid = []
        for _, row in df_level.iterrows():
            p = find_image(img_dir, row["image"])
            if p is not None:
                valid.append(p)
            if len(valid) == n_samples_per_level:
                break
        level_paths[lvl] = valid

    fig, axes = plt.subplots(len(levels), n_samples_per_level, figsize=(3 * n_samples_per_level, 3 * len(levels)), squeeze=False)

    for i, lvl in enumerate(levels):
        row_axes = axes[i, :]
        row_axes[2].annotate(f"Level {lvl}", xy=(0.5, 1.1), xycoords="axes fraction",
                             ha="center", va="bottom", fontsize=14, fontweight="bold")
        for j, ax in enumerate(row_axes):
            ax.axis("off")
            if j < len(level_paths[lvl]):
                try:
                    img = Image.open(level_paths[lvl][j])
                    ax.imshow(img)
                except:
                    pass

    plt.tight_layout()
    save_fig(fig, outdir / "sample_grid.png")
    plt.close(fig)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate exploratory plots for the DR dataset")
    parser.add_argument("--csv", default="trainLabels3.csv", type=Path, help="Path to labels CSV file")
    parser.add_argument("--img_dir", default=Path("/app/resized_train"), type=Path, help="Directory containing the images")
    parser.add_argument("--outdir", default=Path("output2"), type=Path, help="Directory to save output figures")
    args = parser.parse_args()

    # Read labels
    df = pd.read_csv(args.csv)
    required_cols = {"image", "level", "width", "height"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {', '.join(sorted(required_cols))}")

    print(f"Dataset size: {len(df)} samples")

    plot_class_distribution(df, args.outdir)
    plot_image_size_hist(df, args.outdir)
    plot_sample_grid(df, args.img_dir, args.outdir)


if __name__ == "__main__":
    main()
