"""Export portable inference weights at each training epoch."""
import json
from pathlib import Path
import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf


class ExportModel(Callback):
    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.config = config

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        config = OmegaConf.to_container(self.config, resolve=True)
        (self.directory / "config.json").write_text(json.dumps(config, indent=2)+"\n")
        state = {k: v.detach().cpu() for k, v in pl_module.model.state_dict().items()}
        epoch = trainer.current_epoch + 1
        torch.save(state, self.directory / f"weights_epoch_{epoch}.pt")
        torch.save(state, self.directory / "weights.pt")
