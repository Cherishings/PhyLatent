# Reproducibility / 可复现范围

This release contains the base PhyLatent method and fresh-training ablations.
It does not contain fine-tuning, recovery, candidate selection, or server
orchestration. The four inference weights match the frozen final-model
manifest by SHA256. Their architecture configurations were reconstructed from
the common architecture template and checked with strict state-dictionary
loading; they are not original per-run training records.

本发布只提供基础方法、从头训练和消融；四个推理权重哈希匹配最终模型清单。
配套模型配置是根据结构模板和权重核验生成的加载配置，不是原始训练履历。
基础配方不承诺直接重新生成每个现成论文权重。

## Base recipes

All tasks use a 192-dimensional ViT-Tiny/14 JEPA, history size 3, four-observation
clips with frameskip 5, AdamW (LR 5e-5, weight decay 1e-3), constant LR, bf16,
gradient clipping 1.0, and seed 3072. This release explicitly seeds initialization
and workers; that does not establish the initialization seeds of historical runs.

Cube uses microbatch 32 × accumulation 4 (effective batch 128) for ten full
epochs. TwoRooms, Reacher and PushT use batch 32, ten epochs and at most 10,000
training batches per epoch. A capped epoch is not a complete dataset pass.

数据按 clip 随机约 90/10 划分，并非 episode-disjoint；相邻重叠 clips 可能跨子集。
非图像字段的标准化统计来自划分前的数据列。evaluation seeds 不是独立训练次数。

Auxiliary weights are task-specific in `configs/train/{task}.yaml`. Loss mapping:
PSG→state; FRA→align (projected plus action-attention alignment);
SVIP→invariance; CASP→action_separation; LD→diffusion.
Legacy internal names SVIC/CASC correspond to SVIP/CASP.

Each ablation is initialized afresh using the same base training entry point.
`wo_physical_group` disables PSG+FRA; `wo_counterfactual_group` disables CASP+LD.
`wo_invariance_group` disables SVIP and is an alias of `wo_svip` in this method.
Use distinct output directories for distinct runs; existing nonempty training
output directories are rejected instead of resumed.

## Final diagnostics

Inv averages clean-eligible ordering reversals across the nine appearance
conditions at each anchor, then across anchors/seeds, and reports clean coverage.
Dist uses Q1-vs-Q4 physical near/far ordering; ties and non-finite values fail.
CF uses seven constant action branches, 21 branch pairs, fixed bottom/top-five
physical strata, and encoder eligibility. Predicted ordering ties fail.
CF pools eligible failures/counts and uses H5 as primary, H1/H3 as sensitivity.

Optional paired comparisons use the same anchors/actions; CF uses the
reference–PhyLatent common eligibility set, while ablation families require
an intersection over every included model. A single pair's intersection must
not be labeled a family-global intersection. Bootstrap resamples seed then
anchor; default 10,000 percentile replicates.

Standalone model CF rates have model-specific denominators and are not the
paper's matched-comparison rates. Raw arrays and sampling manifests are saved
so comparison families can be reconstructed without changing model weights.
单模型 CF 分母与论文配对共同资格分母不同，不可直接混用数字。

## Validation boundary

Local CPU checks cover syntax, configuration composition, metric edge cases,
base-loss equivalence, strict loading of all four checkpoints and inference
shape/finite checks. Full training and simulator/GPU evaluations require the
runtime dependencies and official datasets and are not claimed to have been
run as part of this extraction. See `validation.md` for the recorded checks.

No frozen paper-result archive is bundled. In particular the prior Reacher
success-table override was screenshot-confirmed and is not promoted here to a
verified raw-per-seed reconstruction. Published table values are not changed.
未把原工作区的截图覆盖汇总当作已验证的逐 seed 原始结果，未修改论文数字。
