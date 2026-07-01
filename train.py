"""
train.py  –  Grid-search optimal signal weights from your labelled dataset.

Usage
-----
    python train.py --real real/ --screen screen/ --out weights.json

Expects two folders with JPEG / PNG photos:
    real/      real photographs     (label = 0)
    screen/    photos of a screen   (label = 1)

Outputs
-------
    weights.json   –  loaded by predict.py at inference time
    report.txt     –  per-signal stats, confusion matrix, CV accuracy

The grid search is fast (~seconds for 100 images) and exhaustive over the
4-signal weight space.  It optimises balanced accuracy so a small class
imbalance does not skew the threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Grid search + evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# Local
sys.path.insert(0, str(Path(__file__).parent))
from features import extract_all, SIGNAL_NAMES

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(real_dir: str, screen_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load images from both folders, extract features.

    Returns
    -------
    X : float32 array of shape (N, 4)
    y : int array of shape (N,)  –  0 = real, 1 = screen
    """
    X_rows, y_rows = [], []

    for label, folder in [(0, real_dir), (1, screen_dir)]:
        paths = sorted([
            p for p in Path(folder).iterdir()
            if p.suffix.lower() in SUPPORTED_EXT
        ])
        if not paths:
            print(f"  [WARN] No images found in {folder!r}", file=sys.stderr)
            continue

        print(f"  Loading {len(paths):>3} images from {folder!r}  (label={label}) …")
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                print(f"  [SKIP] Cannot read {p.name}", file=sys.stderr)
                continue
            try:
                feats = extract_all(img)
                X_rows.append([feats[k] for k in SIGNAL_NAMES])
                y_rows.append(label)
            except Exception as e:
                print(f"  [SKIP] {p.name}: {e}", file=sys.stderr)

    if not X_rows:
        raise RuntimeError("No valid images loaded. Check folder paths.")

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    print(f"\n  Dataset: {(y==0).sum()} real, {(y==1).sum()} screen  "
          f"({len(y)} total)\n")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Weight search  (random sampling + local refinement — scales to N signals)
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_score(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted dot product, shape (N,)."""
    return X @ w  # w already sums to 1


def _evaluate_weight(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, skf: StratifiedKFold,
) -> float:
    """Cross-validated balanced accuracy for one weight vector."""
    scores_all = _weighted_score(X, w)
    fold_accs  = []
    for train_idx, val_idx in skf.split(X, y):
        s_train, y_train = scores_all[train_idx], y[train_idx]
        s_val,   y_val   = scores_all[val_idx],   y[val_idx]

        best_fold_acc, best_fold_tau = -1.0, 0.5
        for tau in np.arange(0.05, 0.96, 0.05):
            preds = (s_train >= tau).astype(int)
            acc   = balanced_accuracy_score(y_train, preds)
            if acc > best_fold_acc:
                best_fold_acc, best_fold_tau = acc, tau

        preds_val = (s_val >= best_fold_tau).astype(int)
        fold_accs.append(balanced_accuracy_score(y_val, preds_val))

    return float(np.mean(fold_accs))

def _find_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Youden-J optimal threshold on the full dataset."""
    best_acc, best_tau = -1.0, 0.5
    for tau in np.arange(0.02, 0.99, 0.01):
        preds = (scores >= tau).astype(int)
        acc   = balanced_accuracy_score(y, preds)
        if acc > best_acc:
            best_acc, best_tau = acc, float(tau)
    return best_tau


def grid_search_weights(
    X: np.ndarray,
    y: np.ndarray,
    n_cv_folds: int = 5,
    n_random: int = 4000,
    n_refine_rounds: int = 3,
    n_refine_per_round: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Search the weight simplex via random sampling + local refinement.
    ... (docstring as before)
    """
    rng = np.random.default_rng(seed)
    n_signals = X.shape[1]
    skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)

    best_score = -1.0
    best_w     = np.ones(n_signals, dtype=np.float32) / n_signals

    # ── Stage 1: global random search over the simplex ──────────────────────
    print(f"  Stage 1: random search over {n_signals}-signal weight simplex "
          f"({n_random:,} samples) …")
    t0 = time.time()
    candidates = rng.dirichlet(np.ones(n_signals), size=n_random).astype(np.float32)

    for idx, w in enumerate(candidates):
        if idx % 500 == 0 and idx > 0:
            elapsed = time.time() - t0
            eta = elapsed / idx * (n_random - idx)
            print(f"    {idx:>6}/{n_random}  best_acc={best_score:.4f}  "
                  f"ETA {eta:.0f}s", flush=True)

        acc = _evaluate_weight(X, y, w, skf)
        if acc > best_score:
            best_score = acc
            best_w     = w.copy()

    print(f"  Stage 1 done in {time.time()-t0:.1f}s  "
          f"best_acc={best_score:.4f}")

    # ── Stage 2: local refinement around the current best ────────────────────
    print(f"\n  Stage 2: local refinement ({n_refine_rounds} rounds × "
          f"{n_refine_per_round:,} samples) …")
    t1 = time.time()
    for round_idx in range(n_refine_rounds):
        scale = 0.15 / (round_idx + 1)   # shrink perturbation each round
        perturbations = rng.normal(0, scale, size=(n_refine_per_round, n_signals))
        round_candidates = best_w[None, :] + perturbations
        round_candidates = np.clip(round_candidates, 0.0, None)
        sums = round_candidates.sum(axis=1, keepdims=True)
        sums[sums < 1e-6] = 1.0
        round_candidates = (round_candidates / sums).astype(np.float32)

        round_best_score = best_score
        round_best_w      = best_w
        for w in round_candidates:
            acc = _evaluate_weight(X, y, w, skf)
            if acc > round_best_score:
                round_best_score = acc
                round_best_w      = w.copy()

        improved = round_best_score > best_score
        best_score, best_w = round_best_score, round_best_w
        print(f"    Round {round_idx+1}/{n_refine_rounds}  "
              f"scale={scale:.3f}  best_acc={best_score:.4f}"
              f"{'  (improved)' if improved else ''}")

    print(f"  Stage 2 done in {time.time()-t1:.1f}s")

    # Re-fit threshold on the full dataset for the final best weight
    scores_full = _weighted_score(X, best_w)
    best_tau    = _find_threshold(scores_full, y)

    total_time = time.time() - t0
    print(f"\n  Total search time: {total_time:.1f}s")
    print(f"  Best CV balanced accuracy : {best_score:.4f}")
    print(f"  Best weights              : "
          + ", ".join(f"{SIGNAL_NAMES[i]}={best_w[i]:.3f}"
                       for i in range(n_signals)))
    print(f"  Best threshold τ          : {best_tau:.3f}\n")

    # Return the correct dict with dynamic n_signals
    return {
        "weights":    {SIGNAL_NAMES[i]: float(best_w[i]) for i in range(n_signals)},
        "threshold":  float(best_tau),
        "cv_score":   float(best_score),
    }





# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    X: np.ndarray,
    y: np.ndarray,
    result: dict[str, Any],
    report_path: str,
) -> None:
    w      = np.array([result["weights"][k] for k in SIGNAL_NAMES], dtype=np.float32)
    tau    = result["threshold"]
    scores = _weighted_score(X, w)
    preds  = (scores >= tau).astype(int)

    acc    = balanced_accuracy_score(y, preds)
    cm     = confusion_matrix(y, preds)

    # Per-signal stats
    lines = []
    lines.append("=" * 60)
    lines.append("SalesCode AI – Recapture Detector  |  Training Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Per-signal statistics")
    lines.append("-" * 40)
    for i, name in enumerate(SIGNAL_NAMES):
        real_vals   = X[y == 0, i]
        screen_vals = X[y == 1, i]
        lines.append(
            f"  {name:<20}  real μ={real_vals.mean():.3f} σ={real_vals.std():.3f}"
            f"   screen μ={screen_vals.mean():.3f} σ={screen_vals.std():.3f}"
        )

    lines.append("")
    lines.append("Optimal weights")
    lines.append("-" * 40)
    for name in SIGNAL_NAMES:
        lines.append(f"  {name:<20}  {result['weights'][name]:.4f}")

    lines.append("")
    lines.append(f"Threshold τ              :  {tau:.4f}")
    lines.append(f"Cross-validated balanced :  {result['cv_score']:.4f}")
    lines.append(f"Full-dataset balanced    :  {acc:.4f}")
    lines.append("")
    lines.append("Confusion matrix  (rows=actual, cols=predicted)")
    lines.append("-" * 40)
    lines.append("             PRED real   PRED screen")
    lines.append(f"  ACT real   {cm[0,0]:>9}   {cm[0,1]:>10}")
    lines.append(f"  ACT screen {cm[1,0]:>9}   {cm[1,1]:>10}")
    lines.append("")
    tn, fp, fn, tp = cm.ravel()
    lines.append(f"  True  Negative (real  → real  ): {tn}")
    lines.append(f"  False Positive (real  → screen): {fp}")
    lines.append(f"  False Negative (screen→ real  ): {fn}")
    lines.append(f"  True  Positive (screen→ screen): {tp}")
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    lines.append("")
    lines.append(f"  Precision : {precision:.4f}")
    lines.append(f"  Recall    : {recall:.4f}")
    lines.append(f"  F1        : {f1:.4f}")
    lines.append("=" * 60)

    text = "\n".join(lines)
    print(text)
    Path(report_path).write_text(text, encoding="utf-8")
    print(f"\n  Report saved → {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train signal weights for the recapture detector."
    )
    parser.add_argument("--real",   default="real/",        help="Folder of real photos")
    parser.add_argument("--screen", default="screen/",      help="Folder of screen photos")
    parser.add_argument("--out",    default="weights.json", help="Output weights file")
    parser.add_argument("--report", default="report.txt",   help="Output report file")
    args = parser.parse_args()

    print("\n──────────────────────────────────────────────")
    print("  Recapture Detector  –  Training")
    print("──────────────────────────────────────────────\n")

    X, y = load_dataset(args.real, args.screen)

    result = grid_search_weights(X, y)

    out_path = args.out
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Weights saved → {out_path}")

    generate_report(X, y, result, args.report)


if __name__ == "__main__":
    main()