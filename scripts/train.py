#!/usr/bin/env python3
"""Train Full or an ablation from initialization with a task configuration."""
import os
from pathlib import Path
import hydra
from omegaconf import OmegaConf
from phylatent.training import train, apply_ablation

@hydra.main(version_base=None, config_path="../configs/train", config_name="cube")
def main(cfg):
    if int(cfg.clip_length) != int(cfg.num_preds) + int(cfg.history_size):
        raise ValueError("clip_length must equal history_size + num_preds")
    data_root = Path(str(cfg.data_root)).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Create the data directory and place task HDF5 files in it: {data_root}")
    os.environ["STABLEWM_HOME"] = str(data_root)
    os.environ["LOCAL_DATASET_DIR"] = str(data_root)
    os.environ.setdefault("SPT_CACHE_DIR", str(Path("outputs/cache").resolve()))
    apply_ablation(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
