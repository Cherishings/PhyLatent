# Visual guide

[中文说明](visualizations.zh-CN.md) · [Back to README](../README.md)

## Architecture

The homepage reproduces the author-confirmed Figure 3 from the paper. Its
source artwork was named `figure2_v30` before the final paper numbering.
Repeated encoders and predictors share parameters; `sg` marks stop-gradient.
Future observations provide training supervision. Planning uses the learned
latent dynamics and does not decode predicted latent states into video frames.

## Reading the empirical collapse plot

The homepage scatter plot shows the **LeWM reference model on Cube**. The
original export selected `cube`, `lewm`, INV/Dist v2 and CF ordering-v3; it is
an illustration of diagnostic failures, not a measurement of PhyLatent gains.
The three panels use different eligible comparison sets. The counterfactual
panel uses horizon 5 and observed-order eligibility.

There are 15,000 eligible invariance comparisons, 15,000 distinguishability
comparisons and 3,485 counterfactual comparisons before plot subsampling.
Each panel displays a uniform sample of 2,500 points. The distinguishability
panel uses a label-blind 1%–99.9% viewport with padding; 44 sampled points lie
outside the viewport. Other panels use their full range. See the copied
[plot metadata](../assets/figures/collapse_diagnostics.json).

## What the failures look like in observations

![Selected Cube examples of three diagnostic failures](../assets/figures/collapse_examples.png)

These selected **LeWM / Cube** examples explain the three diagnostics:

1. A checker perturbation leaves physical distances unchanged but reverses a previously correct latent ordering.
2. The physically farther reference is closer in latent distance.
3. The predicted ordering of action-branch differences reverses the ordering of encoded observed futures.

The future images in the third row are actual simulator observations, not
images decoded from latent predictions. The rows come from different initial
states. Reference images in the second row are not consecutive video frames.
These examples establish diagnostic behavior; they do not by themselves show
closed-loop failure, causal attribution, aggregate success rates, or a paired
PhyLatent improvement.

## Asset provenance

All images are copied from existing paper assets without changing their
contents. The [asset manifest](../assets/figures/manifest.json) records source
filenames and SHA256 hashes. No baseline implementation or weights are added.
