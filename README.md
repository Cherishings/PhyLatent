# PhyLatent

[中文说明](README.zh-CN.md)

PhyLatent learns a latent world model with physical state grounding, future
relation alignment, static visual invariance, counterfactual action separation,
and latent denoising. This repository provides the **base method, from-scratch
training and ablations, four inference checkpoints, planning evaluation, and
final collapse diagnostics** for Cube, TwoRooms, Reacher and PushT.

## Installation

Use Linux and Python 3.10+; CUDA is needed for the provided full training and
planning commands. Install a matching PyTorch/torchvision build first, then
run from this repository:

```bash
python -m pip install -e '.[runtime]'
python scripts/verify_checkpoints.py --load
```

Core-only installation for checkpoint inspection and CPU tests:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

The runtime pins stable-pretraining 0.1.7 and stable-worldmodel 0.1.1.
Simulator assets/system libraries must also satisfy their upstream installation
instructions. Full runtime installation and GPU evaluation have not been
validated during extraction; see [validation notes](docs/validation.md).

## Data

Obtain and extract the official datasets as described in [data preparation](docs/data.md).
Place all four HDF5 files in `data/`, or pass another absolute directory.
Datasets are external and are not bundled with this code.

## Train from initialization

```bash
python scripts/train.py --config-name cube data_root=/path/to/data
python scripts/train.py --config-name tworoom data_root=/path/to/data
python scripts/train.py --config-name reacher data_root=/path/to/data
python scripts/train.py --config-name pusht data_root=/path/to/data
```

`tworoom` is the CLI identifier for TwoRooms. Exports are written to
`outputs/<task>/<variant>/seed_<seed>/model/` with `config.json`, `weights.pt`
(latest epoch), and `weights_epoch_<N>.pt`. Use a distinct `output_dir` for
new runs; the entry point never resumes a nonempty output directory.

Cube uses effective batch 128 for ten full epochs. The other base recipes use
batch 32 and at most 10,000 training batches per epoch for ten epochs.
These are base recipes, not a claim to regenerate every bundled paper checkpoint.

## Ablations

Use the same fresh-training entry point and switch the configuration:

```bash
python scripts/train.py --config-name cube ablation=wo_psg data_root=/path/to/data
python scripts/train.py --config-name pusht ablation=wo_counterfactual_group data_root=/path/to/data
```

Available choices: `full`, `wo_psg`, `wo_fra`, `wo_svip`, `wo_casp`, `wo_ld`,
`wo_physical_group`, `wo_invariance_group`, `wo_counterfactual_group`.
Group/component meanings and evaluation denominators are described in
[reproducibility notes](docs/reproducibility.md).

## Pretrained checkpoints and planning

Four checkpoints are included in `checkpoints/<task>/`. Their SHA256 identities
are recorded in `checkpoints/manifest.json`; each has a portable loading config.

```bash
python scripts/evaluate.py cube --seed 600 --model-dir checkpoints/cube \
  --data-root /path/to/data --output outputs/evaluation/cube_seed600.json
```

Default budget: 100 episodes, CEM 300 samples/top-30, ten iterations (30 for
PushT), horizon 5, action block 5, goal offset 25, evaluation budget 50.
Repeat with seeds 600–605 to collect per-seed planning results.

## Collapse diagnostics

```bash
python scripts/diagnose.py --task cube --model-dir checkpoints/cube \
  --data-root /path/to/data --output outputs/diagnostics/cube
```

This reports invariance, distinguishability and **CF ordering-v3**; no legacy
ratio-threshold CF score is used. Add `--smoke` for a small sampling check.
An optional `--reference-dir` runs a paired common-eligibility comparison using
an externally obtained compatible JEPA checkpoint. No baseline source or
baseline weights are bundled. Standalone CF scores have their own denominators.

## Repository layout

```text
configs/train/       Four base task recipes and loss/component ablations
src/phylatent/       Models, auxiliary losses, preprocessing and training
  evaluation/        Closed-loop CEM planning
  diagnostics/       Frame/branch sampling and final diagnostic statistics
scripts/             Training, evaluation, diagnostics and checkpoint checks
checkpoints/         Four inference weights, loading configs and hash manifest
docs/                Data, reproducibility, source mapping and validation
licenses/            Original third-party license notices
tests/               CPU checks of losses, configurations and metrics
```

## License and citation

The code is released under [MIT](LICENSE). LeWorldModel attribution is retained
in [third-party notices](THIRD_PARTY_NOTICES.md). Data and external dependencies
retain their own licenses. Cite the PhyLatent paper when using this method;
`CITATION.cff` records the software without inventing paper metadata.

The bundled architecture configs have been strictly checked against the weights;
they are loading configs rather than original training histories. For scope and
known reproducibility limits, read [reproducibility notes](docs/reproducibility.md).
