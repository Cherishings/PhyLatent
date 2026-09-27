# Validation record / 验证记录

Date: 2026-09-27. CPU only; no optimizer updates or GPU jobs.

Passed / 已通过：

- Python syntax checks for all package and entry-point sources.
- All 36 combinations of four task configs and nine ablation choices compose.
- Seven unittest cases: task/config composition, normalized loss scale, clean
  prediction stop-gradient, tie/non-finite metric behavior, physical CF ties,
  paired bootstrap, and common-eligibility CF denominators.
- All four checkpoint SHA256 values match the frozen final-model manifest.
- All four models strict-load with no missing/unexpected state-dictionary keys.
- All four models encode/predict and autoregressively roll out on CPU with the
  expected shapes and finite outputs.
- Extracted full base loss and every parameter gradient exactly match the source
  paper-aligned forward under identical CPU random seeds and a small test model.
- Evaluation/diagnostic CLI help is available without simulator dependencies.
- Relative documentation links resolve; selected private-path/key patterns are
  absent from package text. This is not an exhaustive secret certification.

Local core environment: Python 3.10, torch 2.7.0+cu128, torchvision 0.22.0+cu128,
transformers 5.8.0, hydra-core 1.3.2. All validation tensors were on CPU.

Not executed / 未运行：

- Clean-environment dependency installation.
- Full training or checkpoint regeneration.
- Dataset-based simulator/GPU planning and collapse collection.
- Reproduction of all published table values from frozen raw archives.

The current local environment lacks lightning, stable-pretraining and
stable-worldmodel. They are declared in the runtime installation extra;
full runtime operation still needs verification with the official datasets.
