#!/usr/bin/env python3
"""Evaluate one exported PhyLatent model with the LeWM CEM budget."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("cube", "tworoom", "reacher", "pusht"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-label", default="PhyLatent")
    parser.add_argument("--num-eval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--cem-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int)
    parser.add_argument("--cem-topk", type=int, default=30)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("PHYLatent_DATA_ROOT", "data"))
    )
    return parser.parse_args()


def main() -> None:
    global np, torch
    args = parse_args()
    import numpy as np
    import torch
    
    import hdf5plugin  # Register compressed HDF5 filters.
    import hdf5plugin  # Register compressed HDF5 filters.
    import stable_worldmodel as swm
    
    from phylatent.evaluation.tasks import TASKS, fit_processors, image_transform, sample_tasks
    from phylatent.models.loading import load_model
    from phylatent.models import jepa as jepa_module, modules as model_modules
    

    cem_steps = args.cem_steps if args.cem_steps is not None else (
        30 if args.task == "pusht" else 10
    )
    if args.batch_size <= 0 or args.num_eval <= 0 or args.num_eval % args.batch_size:
        raise ValueError("num-eval must be positive and divisible by batch-size")
    if not 0 < args.cem_topk <= args.cem_samples:
        raise ValueError("cem-topk must be in [1, cem-samples]")

    config_path = args.model_dir / "config.json"
    weights_path = args.model_dir / "weights.pt"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"Incomplete exported model: {args.model_dir} "
            "(config.json and weights.pt are required)"
        )

    task = TASKS[args.task]
    dataset_path = args.data_root / task["dataset"]
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Missing dataset: {dataset_path}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = load_model(args.model_dir, device="cpu")
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    dataset = swm.data.HDF5Dataset(path=dataset_path, keys_to_cache=task["keys"])
    episodes, starts = sample_tasks(
        dataset, args.num_eval, args.goal_offset, args.seed
    )
    process = fit_processors(dataset, task["keys"])
    transform = {"pixels": image_transform(), "goal": image_transform()}

    world = swm.World(
        **dict(task["world"]),
        num_envs=args.batch_size,
        image_shape=(224, 224),
        max_episode_steps=2 * args.eval_budget,
    )
    solver = swm.solver.CEMSolver(
        model=model,
        batch_size=1,
        num_samples=args.cem_samples,
        n_steps=cem_steps,
        topk=args.cem_topk,
        var_scale=1.0,
        device="cuda",
        seed=args.seed,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5),
        process=process,
        transform=transform,
    )

    successes: list[bool] = []
    started = time.time()
    for first in range(0, args.num_eval, args.batch_size):
        last = first + args.batch_size
        world.set_policy(policy)
        metrics = world.evaluate(
            dataset=dataset,
            episodes_idx=episodes[first:last],
            start_steps=starts[first:last],
            goal_offset=args.goal_offset,
            eval_budget=args.eval_budget,
            callables=task["callables"],
            video=None,
        )
        successes.extend(
            np.asarray(metrics["episode_successes"], dtype=bool).tolist()
        )
        print(
            f"PROGRESS task={args.task} seed={args.seed} "
            f"{last}/{args.num_eval} successes={sum(successes)}",
            flush=True,
        )
        torch.cuda.empty_cache()

    result = {
        "model": args.model_label,
        "task": args.task,
        "seed": args.seed,
        "num_eval": len(successes),
        "success_rate": 100.0 * float(np.mean(successes)),
        "episode_successes": successes,
        "protocol": {
            "name": "LeWM official CEM budget",
            "goal_offset": args.goal_offset,
            "eval_budget": args.eval_budget,
            "evaluation_batch_size": args.batch_size,
            "cem_samples": args.cem_samples,
            "cem_steps": cem_steps,
            "cem_topk": args.cem_topk,
            "cem_var_scale": 1.0,
            "planning_horizon": 5,
            "receding_horizon": 5,
            "action_block": 5,
        },
        "model_dir": str(args.model_dir.resolve()),
        "weights_sha256": sha256(weights_path),
        "dataset": str(dataset_path.resolve()),
        "sampled_episodes": np.asarray(episodes).tolist(),
        "sampled_starts": np.asarray(starts).tolist(),
        "provenance": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "stable_worldmodel": package_version("stable-worldmodel"),
            "stable_pretraining": package_version("stable-pretraining"),
            "gymnasium": package_version("gymnasium"),
            "mujoco": package_version("mujoco"),
            "transformers": package_version("transformers"),
            "evaluator_sha256": sha256(Path(__file__)),
            "jepa_sha256": sha256(Path(jepa_module.__file__)),
            "module_sha256": sha256(Path(model_modules.__file__)),
        },
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("model", "task", "seed", "num_eval", "success_rate")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
