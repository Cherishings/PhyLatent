# Third-party notices

The JEPA backbone, predictor, action embedding, SIGReg, and base data/training
patterns derive from LeWorldModel by Lucas Maes (2026), released under MIT.
The original license is retained in `licenses/LeWorldModel-MIT.txt`.
Source: https://github.com/lucas-maes/le-wm

Changes include package namespaces, portable checkpoint loading/export,
fresh-training-only entry points, task/ablation configurations, and diagnostic
entry points. PhyLatent adds the physical state, future alignment, visual
invariance, counterfactual separation, and latent denoising objectives.

PyTorch, Transformers, stable-pretraining, stable-worldmodel, simulators,
and their assets are external dependencies with their own licenses.
They are not bundled here. Obtain datasets separately from their publishers.
The four bundled model files are PhyLatent inference weights; their source
identity is recorded in `checkpoints/manifest.json`.
