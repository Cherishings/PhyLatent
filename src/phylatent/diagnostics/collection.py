#!/usr/bin/env python3
"""Paired four-task evaluator for Dynamics-Relevant Collapse Protocol v2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import gymnasium as gym
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from einops import rearrange

import stable_worldmodel as swm  # noqa: F401 - environment registration


from phylatent.models.jepa import JEPA
from phylatent.models.loading import load_model
from .metrics import EPS, hierarchical_paired_bootstrap, identifiability_failures, invariance_failures, sha256, unit_normalize
IMAGENET = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}


FRAMESKIP = 5
HISTORY_SIZE = 3
PAIR_IDS = np.asarray(list(combinations(range(7), 2)), dtype=np.int64)

TASKS = {
    "cube": {
        "dataset": "cube_single_expert.h5",
        "env": "swm/OGBCube-v0",
        "env_kwargs": {
            "env_type": "single", "ob_type": "states", "multiview": False,
            "width": 224, "height": 224, "visualize_info": False,
            "terminate_at_goal": False,
        },
        "ep_key": "ep_idx",
    },
    "pusht": {
        "dataset": "pusht_expert_train.h5",
        "env": "swm/PushT-v1",
        "env_kwargs": {"render_mode": "rgb_array", "resolution": 224, "with_target": True},
        "ep_key": "episode_idx",
    },
    "reacher": {
        "dataset": "reacher.h5",
        "env": "swm/ReacherDMControl-v0",
        "env_kwargs": {"task": "hard", "render_mode": "rgb_array"},
        "ep_key": "ep_idx",
    },
    "tworoom": {
        "dataset": "tworoom.h5",
        "env": "swm/TwoRoom-v1",
        "env_kwargs": {"render_mode": "rgb_array"},
        "ep_key": "ep_idx",
    },
}


def imagenet_normalize(images: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(images).to(device=device, dtype=torch.float32) / 255.0
    x = rearrange(x, "b h w c -> b c h w")
    mean = torch.tensor(IMAGENET["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET["std"], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def transform_images(images: np.ndarray, kind: str, severity: int) -> np.ndarray:
    x = images.astype(np.float32) / 255.0
    level = (0.5, 1.0, 1.5)[severity]
    if kind == "brightness":
        scale, offset = ((0.92, 0.035), (0.84, 0.07), (0.76, 0.105))[severity]
        x = x * scale + offset
    elif kind == "tint":
        base = np.array([1.16, 0.88, 1.05], dtype=np.float32)
        tint = 1.0 + level * (base - 1.0)
        x = x * tint[None, None, None, :]
    elif kind == "checker":
        h, w = x.shape[1:3]
        cell = max(1, int(round(min(h, w) / 12.0)))
        yy, xx = np.indices((h, w))
        mask = (((xx // cell) + (yy // cell)) % 2).astype(np.float32)[None, :, :, None]
        overlay = level * np.array([0.08, -0.03, 0.06], dtype=np.float32)
        x = x + mask * overlay[None, None, None, :]
    else:
        raise ValueError(kind)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


@torch.no_grad()
def encode_images(
    model: JEPA,
    images: np.ndarray,
    batch_size: int,
    device: torch.device,
    transform: tuple[str, int] | None = None,
) -> np.ndarray:
    chunks = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        if transform is not None:
            batch = transform_images(batch, *transform)
        output = model.encoder(imagenet_normalize(batch, device), interpolate_pos_encoding=True)
        emb = model.projector(output.last_hidden_state[:, 0]).detach().cpu().numpy()
        chunks.append(emb)
    return unit_normalize(np.concatenate(chunks, axis=0)).astype(np.float32)


def angle_features(angle: np.ndarray) -> np.ndarray:
    angle = np.asarray(angle, dtype=np.float64).reshape(len(angle), -1)
    return np.concatenate([np.sin(angle), np.cos(angle)], axis=1)


def frame_physical_from_h5(f: h5py.File, task: str, indices: np.ndarray) -> np.ndarray:
    n = len(indices)
    if task == "cube":
        return np.concatenate([
            np.asarray(f["proprio_effector_pos"][indices]),
            angle_features(f["proprio_effector_yaw"][indices]),
            np.asarray(f["proprio_gripper_opening"][indices]).reshape(n, -1),
            np.asarray(f["proprio_gripper_contact"][indices]).reshape(n, -1),
            np.asarray(f["privileged_block_0_pos"][indices]),
            angle_features(f["privileged_block_0_yaw"][indices]),
        ], axis=1)
    if task == "pusht":
        state = np.asarray(f["state"][indices], dtype=np.float64)
        return np.concatenate([state[:, :4], angle_features(state[:, 4])], axis=1)
    if task == "reacher":
        qpos = np.asarray(f["qpos"][indices], dtype=np.float64)
        return np.concatenate([
            angle_features(qpos),
            np.asarray(f["finger_pos"][indices]),
            np.asarray(f["target_pos"][indices]),
        ], axis=1)
    if task == "tworoom":
        return np.asarray(f["pos_agent"][indices], dtype=np.float64)
    raise ValueError(task)


def robust_location_scale(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(raw, dtype=np.float64)
    location = np.nanmedian(raw, axis=0)
    q25, q75 = np.nanquantile(raw, [0.25, 0.75], axis=0)
    scale = (q75 - q25) / 1.349
    keep = np.isfinite(location) & np.isfinite(scale) & (scale > 1e-8)
    if not np.any(keep):
        raise RuntimeError("Task adapter produced no non-constant physical coordinates")
    return location[keep], scale[keep], keep


def standardize(raw: np.ndarray, location: np.ndarray, scale: np.ndarray, keep: np.ndarray) -> np.ndarray:
    return ((np.asarray(raw, dtype=np.float64)[:, keep] - location) / scale).astype(np.float32)


def random_pairs(n: int, count: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.integers(0, n, size=count)
    b = rng.integers(0, n - 1, size=count)
    b += b >= a
    return np.stack([a, b], axis=1)


def make_physical_calibration(
    f: h5py.File, task: str, samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = len(f["pixels"])
    ids = np.linspace(0, total - 1, min(samples, total), dtype=np.int64)
    raw = frame_physical_from_h5(f, task, ids)
    finite = np.isfinite(raw).all(axis=1)
    raw = raw[finite]
    location, scale, keep = robust_location_scale(raw)
    phys = standardize(raw, location, scale, keep)
    rng = np.random.default_rng(seed)
    pairs = random_pairs(len(phys), min(200000, len(phys) * 4), rng)
    dist = np.linalg.norm(phys[pairs[:, 0]] - phys[pairs[:, 1]], axis=1)
    edges = np.quantile(dist[np.isfinite(dist)], [0.25, 0.5, 0.75]).astype(np.float64)
    return location, scale, keep, edges


def valid_temporal_indices(f: h5py.File, task_cfg: dict, max_horizon: int) -> np.ndarray:
    ep_idx = np.asarray(f[task_cfg["ep_key"]][:], dtype=np.int64)
    step = np.asarray(f["step_idx"][:], dtype=np.int64)
    ep_len = np.asarray(f["ep_len"][:], dtype=np.int64)
    good = (step >= (HISTORY_SIZE - 1) * FRAMESKIP) & (
        step + max_horizon * FRAMESKIP < ep_len[ep_idx]
    )
    return np.flatnonzero(good)


def sample_bin_references(
    anchor_phys: np.ndarray,
    pool_phys: np.ndarray,
    edges: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.empty((len(anchor_phys), 4, count), dtype=np.int32)
    available = np.empty((len(anchor_phys), 4), dtype=np.int32)
    for i, anchor in enumerate(anchor_phys):
        d = np.linalg.norm(pool_phys - anchor, axis=1)
        masks = [d <= edges[0], (d > edges[0]) & (d <= edges[1]),
                 (d > edges[1]) & (d <= edges[2]), d > edges[2]]
        for q, mask in enumerate(masks):
            candidates = np.flatnonzero(mask)
            available[i, q] = len(candidates)
            if len(candidates) == 0:
                raise RuntimeError(f"anchor {i} physical bin Q{q + 1} has no references")
            result[i, q] = rng.choice(
                candidates, size=count, replace=len(candidates) < count
            )
    return result, available


def distances_to_refs(anchor_z: np.ndarray, pool_z: np.ndarray, refs: np.ndarray) -> np.ndarray:
    return np.linalg.norm(anchor_z[:, None, :] - pool_z[refs], axis=-1)


def frame_metrics_for_model(
    model: JEPA,
    images: np.ndarray,
    anchor_count: int,
    refs: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    clean = encode_images(model, images, batch_size, device)
    az, pz = clean[:anchor_count], clean[anchor_count:]
    d = [distances_to_refs(az, pz, refs[:, q]) for q in range(4)]
    ident_primary = identifiability_failures(d[0], d[3]).mean(axis=1)
    ident_q1_q3 = identifiability_failures(d[0], d[2]).mean(axis=1)
    ident_q2_q4 = identifiability_failures(d[1], d[3]).mean(axis=1)

    clean_near = np.concatenate([d[0], d[0]], axis=1)
    clean_far = np.concatenate([d[2], d[3]], axis=1)
    condition_rates = []
    condition_coverage = []
    nuisance_medians = []
    for kind in ("brightness", "tint", "checker"):
        for severity in range(3):
            shifted = encode_images(model, images, batch_size, device, (kind, severity))
            saz, spz = shifted[:anchor_count], shifted[anchor_count:]
            sd1 = distances_to_refs(saz, spz, refs[:, 0])
            sd3 = distances_to_refs(saz, spz, refs[:, 2])
            sd4 = distances_to_refs(saz, spz, refs[:, 3])
            eligible, failure = invariance_failures(
                clean_near, clean_far,
                np.concatenate([sd1, sd1], axis=1),
                np.concatenate([sd3, sd4], axis=1),
            )
            denom = eligible.sum(axis=1)
            per_anchor = np.divide(
                (failure & eligible).sum(axis=1), denom,
                out=np.full(anchor_count, np.nan), where=denom > 0,
            )
            condition_rates.append(per_anchor)
            condition_coverage.append(eligible.mean(axis=1))
            nuisance_medians.append(np.median(np.linalg.norm(az - saz, axis=1)))
    with np.errstate(invalid="ignore"):
        inv_primary = np.nanmean(np.stack(condition_rates, axis=1), axis=1)
    return {
        "invariance_primary": inv_primary,
        "invariance_condition_rates": np.stack(condition_rates, axis=1),
        "invariance_clean_coverage": np.mean(np.stack(condition_coverage, axis=1), axis=1),
        "same_state_nuisance_medians": np.asarray(nuisance_medians),
        "identifiability_primary": ident_primary,
        "identifiability_q1_q3": ident_q1_q3,
        "identifiability_q2_q4": ident_q2_q4,
    }


def frame_stage(
    f: h5py.File,
    task: str,
    models: dict[str, JEPA],
    seeds: list[int],
    anchors: int,
    pool_size: int,
    refs_per_bin: int,
    calibration: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    location, scale, keep, edges = calibration
    all_results: dict[str, dict[str, list[np.ndarray]]] = {name: {} for name in models}
    plans = {}
    total = len(f["pixels"])
    universe = np.arange(total, dtype=np.int64)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(universe, size=anchors + pool_size, replace=False)
        anchor_ids = np.sort(chosen[:anchors])
        pool_ids = np.sort(chosen[anchors:])
        anchor_raw = frame_physical_from_h5(f, task, anchor_ids)
        pool_raw = frame_physical_from_h5(f, task, pool_ids)
        anchor_phys = standardize(anchor_raw, location, scale, keep)
        pool_phys = standardize(pool_raw, location, scale, keep)
        refs, available = sample_bin_references(
            anchor_phys, pool_phys, edges, refs_per_bin, rng
        )
        images = np.concatenate([np.asarray(f["pixels"][anchor_ids]), np.asarray(f["pixels"][pool_ids])], axis=0)
        plans[str(seed)] = {
            "anchor_indices": anchor_ids.tolist(), "pool_indices": pool_ids.tolist(),
            "reference_positions_sha256": __import__("hashlib").sha256(refs.tobytes()).hexdigest(),
            "available_references_per_anchor_bin_min": available.min(axis=0).tolist(),
            "anchor_bins_requiring_replacement": int(np.sum(available < refs_per_bin)),
        }
        for name, model in models.items():
            print(f"[{task}] frame seed={seed} model={name}", flush=True)
            result = frame_metrics_for_model(model, images, anchors, refs, batch_size, device)
            for key, value in result.items():
                all_results[name].setdefault(key, []).append(value)
    stacked = {
        name: {key: np.stack(values, axis=0) for key, values in metrics.items()}
        for name, metrics in all_results.items()
    }
    return stacked, {"physical_edges": edges.tolist(), "plans": plans}


def action_calibration(f: h5py.File, samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    actions_ds = f["action"]
    ids = np.linspace(0, len(actions_ds) - 1, min(samples, len(actions_ds)), dtype=np.int64)
    actions = np.nan_to_num(np.asarray(actions_ds[ids], dtype=np.float64), nan=0.0)
    center = np.median(actions, axis=0)
    q25, q75 = np.quantile(actions, [0.25, 0.75], axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.std(actions, axis=0)
    scale = np.where(scale > 1e-8, scale, np.maximum(fallback, 1e-6))
    standardized = (actions - center) / scale
    covariance = np.cov(standardized, rowvar=False)
    values, vectors = np.linalg.eigh(np.atleast_2d(covariance))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    if actions.shape[1] >= 3:
        directions = vectors[:, :3].T
    elif actions.shape[1] == 2:
        diagonal = vectors[:, 0] + vectors[:, 1]
        diagonal /= np.linalg.norm(diagonal)
        directions = np.stack([vectors[:, 0], vectors[:, 1], diagonal])
    else:
        raise RuntimeError("Protocol requires action dimension >= 2")
    radius = float(np.quantile(np.linalg.norm(standardized, axis=1), 0.75))
    raw_branches = [center]
    for direction in directions:
        raw_branches.extend([center + scale * radius * direction, center - scale * radius * direction])
    low = np.asarray([-1.0] * actions.shape[1])
    high = np.asarray([1.0] * actions.shape[1])
    branches = np.clip(np.asarray(raw_branches), low, high).astype(np.float32)
    raw_branch_array = np.asarray(raw_branches)
    clip_rate = float(np.mean((raw_branch_array < low) | (raw_branch_array > high)))
    explained = values / max(values.sum(), EPS)
    metadata = {
        "center": center.tolist(), "robust_scale": scale.tolist(), "radius_q75": radius,
        "directions_standardized": directions.tolist(), "branches": branches.tolist(),
        "explained_variance_ratio": explained.tolist(), "coordinate_clipping_rate": clip_rate,
    }
    return center.astype(np.float32), scale.astype(np.float32), branches, metadata


def block_normalizer(f: h5py.File, samples: int = 200000) -> tuple[np.ndarray, np.ndarray]:
    actions = f["action"]
    total = len(actions) - FRAMESKIP
    starts = np.linspace(0, total - 1, min(samples, total), dtype=np.int64)
    blocks = np.stack([
        np.nan_to_num(actions[i : i + FRAMESKIP], nan=0.0).reshape(-1) for i in starts
    ], axis=0)
    return blocks.mean(axis=0).astype(np.float32), np.maximum(blocks.std(axis=0), 1e-6).astype(np.float32)


def current_state_from_h5(f: h5py.File, task: str, idx: int) -> dict:
    if task == "cube":
        return {"state": np.concatenate([f["qpos"][idx], f["qvel"][idx]]).astype(np.float64)}
    if task == "pusht":
        return {"state": np.asarray(f["state"][idx], dtype=np.float64)}
    if task == "reacher":
        return {"state": np.concatenate([f["qpos"][idx], f["qvel"][idx]]).astype(np.float64)}
    if task == "tworoom":
        return {
            "state": np.asarray(f["pos_agent"][idx], dtype=np.float32),
            "target_state": np.asarray(f["pos_target"][idx], dtype=np.float32),
        }
    raise ValueError(task)


def physical_from_info(task: str, info: dict) -> np.ndarray:
    if task == "cube":
        yaw_e = float(np.asarray(info["proprio/effector_yaw"]).reshape(-1)[0])
        yaw_b = float(np.asarray(info["privileged/block_0_yaw"]).reshape(-1)[0])
        return np.concatenate([
            np.asarray(info["proprio/effector_pos"]).reshape(-1),
            [np.sin(yaw_e), np.cos(yaw_e)],
            np.asarray(info["proprio/gripper_opening"]).reshape(-1),
            np.asarray(info["proprio/gripper_contact"]).reshape(-1),
            np.asarray(info["privileged/block_0_pos"]).reshape(-1),
            [np.sin(yaw_b), np.cos(yaw_b)],
        ]).astype(np.float64)
    if task == "pusht":
        pose = np.asarray(info["block_pose"], dtype=np.float64)
        return np.concatenate([
            np.asarray(info["pos_agent"], dtype=np.float64), pose[:2],
            [np.sin(pose[2]), np.cos(pose[2])],
        ])
    if task == "reacher":
        qpos = np.asarray(info["qpos"], dtype=np.float64)
        return np.concatenate([
            np.sin(qpos), np.cos(qpos),
            np.asarray(info["finger_pos"], dtype=np.float64),
            np.asarray(info["target_pos"], dtype=np.float64),
        ])
    if task == "tworoom":
        return np.asarray(info["proprio"], dtype=np.float64)
    raise ValueError(task)


def reset_branch(env: gym.Env, task: str, state: dict, seed: int) -> dict:
    if task == "pusht":
        _, info = env.reset(seed=seed, options={"state": state["state"]})
    elif task == "tworoom":
        _, info = env.reset(seed=seed, options=state)
    else:
        _, info = env.reset(seed=seed, options={"state": state["state"]})
    if task == "reacher":
        restored = np.concatenate([info["qpos"], info["qvel"]])
        if not np.allclose(restored, state["state"], atol=1e-10):
            raise RuntimeError("Reacher reset failed to restore qpos/qvel")
    return info


def simulate_all_branches(
    env: gym.Env,
    task: str,
    state: dict,
    branches: np.ndarray,
    max_horizon: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    all_images, all_phys = [], []
    for action in branches:
        reset_branch(env, task, state, seed)
        branch_images, branch_phys = [], []
        invalid = False
        for _ in range(max_horizon):
            info = None
            for _ in range(FRAMESKIP):
                _, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    invalid = True
                    break
            if invalid:
                break
            branch_images.append(env.render())
            branch_phys.append(physical_from_info(task, info))
        if invalid:
            return None
        all_images.append(branch_images)
        all_phys.append(branch_phys)
    return np.asarray(all_images, dtype=np.uint8), np.asarray(all_phys, dtype=np.float64)


@torch.no_grad()
def predict_branches(
    model: JEPA,
    history_images: np.ndarray,
    history_blocks: np.ndarray,
    branches: np.ndarray,
    block_mean: np.ndarray,
    block_std: np.ndarray,
    max_horizon: int,
    device: torch.device,
) -> np.ndarray:
    branch_count = len(branches)
    pixels = imagenet_normalize(history_images, device).unsqueeze(0).unsqueeze(0)
    pixels = pixels.expand(1, branch_count, HISTORY_SIZE, -1, -1, -1).contiguous()
    hist = ((history_blocks - block_mean) / block_std).astype(np.float32)
    sequences = []
    for action in branches:
        block = np.tile(action, FRAMESKIP).astype(np.float32)
        future = np.stack([block] * max_horizon)
        future = ((future - block_mean) / block_std).astype(np.float32)
        sequences.append(np.concatenate([hist, future], axis=0))
    actions = torch.from_numpy(np.stack(sequences)[None]).to(device=device, dtype=torch.float32)
    out = model.rollout({"pixels": pixels}, actions, history_size=HISTORY_SIZE)
    pred = out["predicted_emb"][0, :, HISTORY_SIZE + 1 : HISTORY_SIZE + max_horizon + 1]
    return unit_normalize(pred.detach().cpu().numpy()).astype(np.float32)


def pair_distances(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values[PAIR_IDS[:, 0]] - values[PAIR_IDS[:, 1]], axis=-1)


def counterfactual_stage(
    f: h5py.File,
    task: str,
    task_cfg: dict,
    models: dict[str, JEPA],
    seeds: list[int],
    anchors: int,
    calibration: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    location, scale, keep, _ = calibration
    _, _, branches, branch_meta = action_calibration(f, 200000)
    block_mean, block_std = block_normalizer(f)
    valid_indices = valid_temporal_indices(f, task_cfg, max_horizon=5)
    env = gym.make(task_cfg["env"], **task_cfg["env_kwargs"])
    physical_by_seed, selected_by_seed = [], []
    model_true = {name: [] for name in models}
    model_pred = {name: [] for name in models}
    for seed in seeds:
        rng = np.random.default_rng(seed + 100000)
        candidates = rng.choice(valid_indices, size=min(len(valid_indices), anchors * 3), replace=False)
        physical_rows = []
        true_rows = {name: [] for name in models}
        pred_rows = {name: [] for name in models}
        selected = []
        for candidate_i, idx in enumerate(candidates):
            if len(selected) >= anchors:
                break
            history_ids = idx - np.arange(HISTORY_SIZE - 1, -1, -1) * FRAMESKIP
            history_images = np.asarray(f["pixels"][history_ids], dtype=np.uint8)
            history_blocks = np.stack([
                np.nan_to_num(f["action"][h : h + FRAMESKIP], nan=0.0).reshape(-1)
                for h in history_ids
            ]).astype(np.float32)
            state = current_state_from_h5(f, task, int(idx))
            simulated = simulate_all_branches(
                env, task, state, branches, 5, seed * 1000000 + candidate_i * 101
            )
            if simulated is None:
                continue
            future_images, future_phys_raw = simulated
            future_phys = ((future_phys_raw[..., keep] - location) / scale).astype(np.float32)
            physical_rows.append(np.stack([
                pair_distances(future_phys[:, h - 1]) for h in (1, 3, 5)
            ], axis=0))
            for name, model in models.items():
                pred = predict_branches(
                    model, history_images, history_blocks, branches,
                    block_mean, block_std, 5, device,
                )
                true = encode_images(
                    model, future_images.reshape(-1, *future_images.shape[2:]),
                    batch_size, device,
                ).reshape(7, 5, -1)
                true_rows[name].append(np.stack([pair_distances(true[:, h - 1]) for h in (1, 3, 5)]))
                pred_rows[name].append(np.stack([pair_distances(pred[:, h - 1]) for h in (1, 3, 5)]))
            selected.append(int(idx))
            if len(selected) % 25 == 0:
                print(f"[{task}] counterfactual seed={seed} anchors={len(selected)}/{anchors}", flush=True)
        if len(selected) != anchors:
            raise RuntimeError(f"{task} seed {seed}: only {len(selected)}/{anchors} valid counterfactual anchors")
        selected_by_seed.append(selected)
        physical_by_seed.append(np.stack(physical_rows))
        for name in models:
            model_true[name].append(np.stack(true_rows[name]))
            model_pred[name].append(np.stack(pred_rows[name]))
    env.close()
    physical = np.stack(physical_by_seed)  # seed, anchor, horizon, pair
    thresholds = np.median(physical, axis=(0, 1, 3))
    results = {}
    for name in models:
        true = np.stack(model_true[name])
        pred = np.stack(model_pred[name])
        results[name] = {
            "counterfactual_physical_distance": physical,
            "counterfactual_true_distance": true,
            "counterfactual_pred_distance": pred,
        }
    metadata = {
        "selected_indices": selected_by_seed,
        "physical_eligibility_thresholds_h1_h3_h5": thresholds.tolist(),
        "physical_pair_distance_shape": list(physical.shape),
        "branch_calibration": branch_meta,
    }
    return results, metadata
