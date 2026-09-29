#!/usr/bin/env python3
"""Model-agnostic statistics for collapse diagnostics."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


EPS = 1e-8


def unit_normalize(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def invariance_failures(
    clean_near: np.ndarray,
    clean_far: np.ndarray,
    shifted_near: np.ndarray,
    shifted_far: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eligible mask and nuisance-induced near/far order reversals."""
    arrays = [np.asarray(v, dtype=np.float64) for v in (
        clean_near, clean_far, shifted_near, shifted_far
    )]
    finite = np.logical_and.reduce([np.isfinite(v) for v in arrays])
    eligible = finite & (arrays[0] < arrays[1])
    failure = (~finite) | (arrays[2] >= arrays[3])
    return eligible, failure


def identifiability_failures(
    latent_near: np.ndarray, latent_far: np.ndarray
) -> np.ndarray:
    """Far state no farther than near state; ties/non-finite count as failures."""
    near = np.asarray(latent_near, dtype=np.float64)
    far = np.asarray(latent_far, dtype=np.float64)
    return (~np.isfinite(near)) | (~np.isfinite(far)) | (far <= near)


def rate(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=np.float64)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def hierarchical_paired_bootstrap(
    baseline: np.ndarray,
    method: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    """Paired seed/anchor bootstrap for arrays shaped [seed, anchor]."""
    baseline = np.asarray(baseline, dtype=np.float64)
    method = np.asarray(method, dtype=np.float64)
    if baseline.shape != method.shape or baseline.ndim != 2:
        raise ValueError("baseline and method must share [seed, anchor] shape")
    rng = np.random.default_rng(seed)
    n_seed, n_anchor = baseline.shape
    draws = np.empty(resamples, dtype=np.float64)
    for draw in range(resamples):
        seed_ids = rng.integers(0, n_seed, size=n_seed)
        seed_diffs = []
        for seed_id in seed_ids:
            anchor_ids = rng.integers(0, n_anchor, size=n_anchor)
            b = baseline[seed_id, anchor_ids]
            m = method[seed_id, anchor_ids]
            if np.any(np.isfinite(b)) and np.any(np.isfinite(m)):
                seed_diffs.append(float(np.nanmean(m) - np.nanmean(b)))
        draws[draw] = np.mean(seed_diffs) if seed_diffs else np.nan
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return {"difference": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    return {
        "difference": float(np.nanmean(method) - np.nanmean(baseline)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
