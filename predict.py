"""
predict.py  –  Screen-recapture detector (classical signal baseline).

Usage
-----
    python predict.py <image_path>          →  prints score (0.0–1.0)
    python predict.py <image_path> --debug  →  prints per-signal breakdown + timing
    python predict.py <image_path> --label  →  prints "REAL" or "SCREEN"

Score convention
----------------
    0.0  =  definitely a real photo
    1.0  =  definitely a photo of a screen

The script loads weights from weights.json (produced by train.py).
If weights.json is missing it falls back to sensible defaults so the
script always works out of the box.

Latency target: < 15 ms on laptop CPU (single image, no GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── locate features.py in the same directory as this script ──────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from features import extract_all, SIGNAL_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# Default weights (used when weights.json is absent)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_WEIGHTS = {
    "fft_peaks":             0.7409,
    "screen_luminance":      0.0000,
    "color_gamut":           0.0304,
    "channel_decorrelation": 0.0000,
    "halftone":              0.0655,
    "glare_blob":            0.0000,
    "double_compression":    0.0103,
    "edge_density":          0.1529,
}
_DEFAULT_THRESHOLD = 0.45   # from the 175-photo personal dataset, 90.1% CV balanced accuracy

# Active signals at inference (non-zero weight) — kept in sync with _DEFAULT_WEIGHTS.
# Zero-weight signals are still computed by features.py (so train.py can rediscover
# them on future datasets) but skipped here to avoid dead computation at inference.
_ACTIVE_SIGNALS = [k for k, v in _DEFAULT_WEIGHTS.items() if v > 0.0]


def _load_weights(weights_path: str | Path) -> tuple[dict, float, list[str]]:
    """
    Load weights + threshold from JSON.  Falls back to defaults if the file
    is missing or malformed.

    Returns
    -------
    weights        : dict of signal_name -> float
    threshold      : float
    active_signals : list of signal names with weight > 0 (skip-zero optimization)
    """
    p = Path(weights_path)
    if not p.exists():
        return _DEFAULT_WEIGHTS.copy(), _DEFAULT_THRESHOLD, list(_ACTIVE_SIGNALS)

    try:
        data      = json.loads(p.read_text())
        weights   = {k: float(data["weights"][k]) for k in SIGNAL_NAMES}
        threshold = float(data.get("threshold", _DEFAULT_THRESHOLD))
        active    = [k for k, v in weights.items() if v > 0.0]
        return weights, threshold, active
    except Exception as e:
        print(f"[WARN] Could not parse {p}: {e}. Using defaults.", file=sys.stderr)
        return _DEFAULT_WEIGHTS.copy(), _DEFAULT_THRESHOLD, list(_ACTIVE_SIGNALS)


def predict(
    image_path: str | Path,
    weights_path: str | Path = "weights.json",
) -> float:
    
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"cv2.imread failed for: {p}  "
                         "(unsupported format or corrupted file)")

    weights, _, active = _load_weights(weights_path)
    feats = extract_all(img)

    # Only sum over active (non-zero-weight) signals — skips dead computation
    w_sum = sum(weights[k] for k in active) or 1.0
    score = sum(weights[k] * feats[k] for k in active) / w_sum

    return float(np.clip(score, 0.0, 1.0))


def predict_with_details(
    image_path: str | Path,
    weights_path: str | Path = "weights.json",
) -> dict:
    """
    Like predict(), but also returns per-signal breakdown and timing.

    Returns
    -------
    dict with keys:
        score       float – final fused score
        label       str   – "REAL" or "SCREEN"
        signals     dict  – per-signal scores
        weights     dict  – weights used
        threshold   float – decision boundary
        latency_ms  float – wall-clock time in milliseconds
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"cv2.imread failed for: {p}")

    weights, threshold, active = _load_weights(weights_path)

    t0    = time.perf_counter()
    feats = extract_all(img)
    t1    = time.perf_counter()

    w_sum = sum(weights[k] for k in active) or 1.0
    score = float(np.clip(
        sum(weights[k] * feats[k] for k in active) / w_sum,
        0.0, 1.0
    ))

    return {
        "score":      score,
        "label":      "SCREEN" if score >= threshold else "REAL",
        "confidence": round(_confidence(score, threshold), 4),
        "signals":    {k: round(feats[k], 4) for k in SIGNAL_NAMES},
        "weights":    {k: round(weights[k], 4) for k in SIGNAL_NAMES},
        "threshold":  round(threshold, 4),
        "latency_ms": round((t1 - t0) * 1000, 2),
        "image":      str(p),
        "resolution": f"{img.shape[1]}×{img.shape[0]}",
    }


def _confidence(score: float, threshold: float) -> float:
    """
    Distance from the decision threshold, normalised to [0, 1].

    0.0 means the score sits exactly on the threshold (maximally
    ambiguous); 1.0 means the score is as far as possible from the
    threshold on whichever side it falls. Asymmetric because the
    threshold (0.40 in the trained weights.json) is not at the
    score range's midpoint.
    """
    if score >= threshold:
        denom = max(1.0 - threshold, 1e-6)
        return float(np.clip((score - threshold) / denom, 0.0, 1.0))
    denom = max(threshold, 1e-6)
    return float(np.clip((threshold - score) / denom, 0.0, 1.0))


def predict_array(
    img_bgr: np.ndarray,
    weights_path: str | Path = "weights.json",
) -> dict:
    
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("img_bgr is empty or None")

    weights, threshold, active = _load_weights(weights_path)

    t0    = time.perf_counter()
    feats = extract_all(img_bgr)
    t1    = time.perf_counter()

    w_sum = sum(weights[k] for k in active) or 1.0
    score = float(np.clip(
        sum(weights[k] * feats[k] for k in active) / w_sum,
        0.0, 1.0
    ))

    return {
        "score":      score,
        "label":      "SCREEN" if score >= threshold else "REAL",
        "confidence": round(_confidence(score, threshold), 4),
        "signals":    {k: round(feats[k], 4) for k in SIGNAL_NAMES},
        "weights":    {k: round(weights[k], 4) for k in SIGNAL_NAMES},
        "threshold":  round(threshold, 4),
        "latency_ms": round((t1 - t0) * 1000, 2),
        "resolution": f"{img_bgr.shape[1]}×{img_bgr.shape[0]}",
    }

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Predict whether an image is a real photo or a screen photo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py photo.jpg              # prints 0.72
  python predict.py photo.jpg --label      # prints SCREEN
  python predict.py photo.jpg --debug      # full per-signal breakdown
  python predict.py photo.jpg --weights custom_weights.json
        """,
    )
    p.add_argument("image", help="Path to the image to classify")
    p.add_argument("--weights", default="weights.json",
                   help="Path to weights.json (default: ./weights.json)")
    p.add_argument("--label",  action="store_true",
                   help="Print REAL / SCREEN instead of the raw score")
    p.add_argument("--debug",  action="store_true",
                   help="Print full per-signal breakdown + latency")
    p.add_argument("--json",   action="store_true",
                   help="Output as JSON (useful for piping)")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # ── weights file resolution: same dir as predict.py if relative ──────────
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        candidate = _HERE / weights_path
        if candidate.exists():
            weights_path = candidate

    if args.debug or args.json:
        details = predict_with_details(args.image, weights_path)

        if args.json:
            import json as _json
            print(_json.dumps(details, indent=2))
            return

        # Human-readable debug output
        print()
        print(f"  Image      : {details['image']}  ({details['resolution']})")
        print(f"  Latency    : {details['latency_ms']} ms")
        print(f"  Threshold  : {details['threshold']}")
        print()
        print("  Active signals (non-zero weight):")
        active_names = [n for n in SIGNAL_NAMES if details['weights'][n] > 0]
        for name in active_names:
            bar_len = int(details['signals'][name] * 30)
            bar     = "█" * bar_len + "░" * (30 - bar_len)
            contrib = details['signals'][name] * details['weights'][name]
            print(f"    {name:<22} {details['signals'][name]:.4f}  "
                  f"[{bar}]  w={details['weights'][name]:.3f}  "
                  f"→ {contrib:.4f}")
        zeroed = [n for n in SIGNAL_NAMES if details['weights'][n] == 0]
        if zeroed:
            print(f"  Zeroed (not contributing): {', '.join(zeroed)}")
        print()
        print(f"  Final score : {details['score']:.4f}")
        print(f"  Label       : {details['label']}")
        print(f"  Confidence  : {details['confidence']:.1%}")
        print()
        return

    
    score = predict(args.image, weights_path)

    if args.label:
        _, threshold, _ = _load_weights(weights_path)
        print("SCREEN" if score >= threshold else "REAL")
    else:
        
        print(round(score, 2))


if __name__ == "__main__":
    main()