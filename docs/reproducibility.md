# Training and evaluation notes / 训练与评估说明

## Training

The task recipes are in `configs/train/`. All tasks use a 192-dimensional
ViT-Tiny/14 JEPA, three history observations, four-observation clips with
frameskip 5, and AdamW with learning rate 5e-5 and weight decay 1e-3.
Training uses constant learning rate, bf16, gradient clipping 1.0 and seed 3072.

Cube uses batch 32 with four gradient-accumulation steps, giving an effective
batch size of 128, for ten full epochs. The other tasks use batch 32 for ten
epochs, capped at 10,000 training batches per epoch.

Clips are randomly split approximately 90/10. This is not an episode-level
split: overlapping clips may occur in both subsets. Non-image normalization
statistics are computed before the split.

The supplied recipes support training from initialization. They are not
complete training histories for the bundled checkpoints, so exact checkpoint
regeneration is not guaranteed. Checkpoint configs describe model loading.
Evaluation seeds select evaluation runs, not independently trained models.

配方支持从头训练；随附权重的配置用于模型加载，并非完整训练记录。
数据按片段随机划分，相邻片段可能跨训练集与验证集。评估种子不代表独立训练次数。

## Losses and ablations

Task-specific loss weights are in `configs/train/<task>.yaml`.
The configuration keys correspond to these objectives:

- PSG: `state` — physical state supervision.
- FRA: `align` — future relation alignment.
- SVIP: `invariance` — consistency under appearance changes.
- CASP: `action_separation` — separation of action-conditioned futures.
- LD: `diffusion` — conditional latent denoising.

Each ablation starts a new training run. `wo_physical_group` disables PSG and
FRA; `wo_counterfactual_group` disables CASP and LD; `wo_invariance_group`
disables SVIP and is equivalent to `wo_svip`. Use a separate output directory
for each run.

## Diagnostic metrics

Invariance measures near/far order reversals after appearance changes, using
comparisons ordered correctly before perturbation. Results are averaged over
appearance conditions, observations and seeds.

Distinguishability compares states from the nearest and farthest physical-distance
quartiles. Tied or non-finite latent distances count as failures.

Counterfactual diagnostics compare seven constant-action branches. Physically
similar and dissimilar branch pairs are selected by their actual outcomes.
A comparison is scored when encoded observed futures preserve the physical
ordering; a reversal or tie in predicted ordering counts as failure.
The primary prediction horizon is five, with one and three also reported.

For model comparisons, use the same sampled observations and action branches,
and the intersection of comparisons that qualify for every model being compared.
Single-model counterfactual rates can use different comparison sets and should
not be compared as if their denominators matched. Confidence intervals resample
seeds and then observations. The command saves raw arrays and sampling settings
for further analysis.

模型对比应使用相同观测、动作分支和共同可比较样本。单模型统计的样本范围可能不同，
不能直接当作同一组配对结果比较。

## Verification

See [verification commands](validation.md) for checkpoint and CPU checks.
Full training and dataset-based GPU evaluation have not been independently
verified for this release package.
