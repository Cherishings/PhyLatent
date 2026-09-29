#!/usr/bin/env python3
"""NumPy implementation of counterfactual dynamics ordering diagnostics."""

from __future__ import annotations

import numpy as np


def gather_extremes(
    values: np.ndarray, low_ids: np.ndarray, high_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    low = np.take_along_axis(values, low_ids, axis=-1)
    high = np.take_along_axis(values, high_ids, axis=-1)
    return low, high


def anchor_fraction(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    num = numerator.sum(axis=(-2, -1))
    den = denominator.sum(axis=(-2, -1))
    return np.divide(
        num,
        den,
        out=np.full(num.shape, np.nan, dtype=np.float64),
        where=den > 0,
    )


def model_ordering_arrays(
    observed: np.ndarray,
    predicted: np.ndarray,
    physical: np.ndarray,
    low_ids: np.ndarray,
    high_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    if observed.shape != physical.shape or predicted.shape != physical.shape:
        raise RuntimeError(
            f"Distance shape mismatch: {observed.shape=} {predicted.shape=} "
            f"{physical.shape=}"
        )
    if not (
        np.isfinite(observed).all()
        and np.isfinite(predicted).all()
        and np.isfinite(physical).all()
    ):
        raise RuntimeError("Non-finite counterfactual distance encountered")

    phy_low, phy_high = gather_extremes(physical, low_ids, high_ids)
    obs_low, obs_high = gather_extremes(observed, low_ids, high_ids)
    pred_low, pred_high = gather_extremes(predicted, low_ids, high_ids)
    physical_contrast = phy_high[..., :, None] > phy_low[..., None, :]
    observed_correct = obs_high[..., :, None] > obs_low[..., None, :]
    predicted_correct = pred_high[..., :, None] > pred_low[..., None, :]
    eligible = physical_contrast & observed_correct
    failure = ~predicted_correct
    observed_wrong = physical_contrast & ~observed_correct

    return {
        "physical_contrast": physical_contrast,
        "eligible": eligible,
        "failure": failure,
        "collapse_per_anchor": anchor_fraction(eligible & failure, eligible),
        "eligibility_per_anchor": anchor_fraction(eligible, physical_contrast),
        "end_to_end_failure_per_anchor": anchor_fraction(
            physical_contrast & failure, physical_contrast
        ),
        "correction_per_anchor": anchor_fraction(
            observed_wrong & predicted_correct, observed_wrong
        ),
    }


def hierarchical_pooled_paired_bootstrap(
    common_eligible: np.ndarray,
    baseline_failure: np.ndarray,
    method_failure: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    """Paired seed/anchor bootstrap of pooled conditional-rate difference."""
    eligible = np.asarray(common_eligible, dtype=bool)
    baseline = np.asarray(baseline_failure, dtype=bool)
    method = np.asarray(method_failure, dtype=bool)
    if eligible.shape != baseline.shape or eligible.shape != method.shape:
        raise ValueError("eligibility and failure tensors must share shape")
    if eligible.ndim != 4:
        raise ValueError("expected [seed, anchor, high, low] tensors")
    denominator = eligible.sum(axis=(-2, -1)).astype(np.float64)
    baseline_num = (eligible & baseline).sum(axis=(-2, -1)).astype(np.float64)
    method_num = (eligible & method).sum(axis=(-2, -1)).astype(np.float64)
    total_den = denominator.sum()
    if total_den <= 0:
        return {
            "difference": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    point = method_num.sum() / total_den - baseline_num.sum() / total_den
    rng = np.random.default_rng(seed)
    n_seed, n_anchor = denominator.shape
    draws = np.empty(resamples, dtype=np.float64)
    for draw in range(resamples):
        seed_ids = rng.integers(0, n_seed, size=n_seed)
        b_num = m_num = den = 0.0
        for seed_id in seed_ids:
            anchor_ids = rng.integers(0, n_anchor, size=n_anchor)
            b_num += baseline_num[seed_id, anchor_ids].sum()
            m_num += method_num[seed_id, anchor_ids].sum()
            den += denominator[seed_id, anchor_ids].sum()
        draws[draw] = m_num / den - b_num / den if den > 0 else np.nan
    draws = draws[np.isfinite(draws)]
    return {
        "difference": float(point),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }
