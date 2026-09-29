# Visual guide

[中文说明](visualizations.zh-CN.md) · [Back to README](../README.md)

## Architecture

Figure 3 shows how PhyLatent learns and uses a latent world model.
The encoder represents observations, and the predictor estimates future states under candidate actions.
The auxiliary objectives guide training; planning uses the encoder and predictor.
Repeated encoders and predictors share parameters. `sg` denotes stop-gradient.

## Collapse diagnostics

We introduce three diagnostics for the physical relationships needed by latent-space planning.
The paper provides their formal definitions and equations. The scatter plot applies these diagnostics
to the **LeWM reference model on Cube**.
Red points mark errors, other colors mark preserved order, and the planes mark the boundaries between them.

- **Invariance:** changing appearance reverses the model's near/far ordering.
- **Distinguishability:** a physically farther state appears closer in latent space.
- **Counterfactual dynamics:** predictions reverse the ordering found in observed futures.

The figure illustrates these errors; it does not compare PhyLatent against LeWM or report task success rates.

## Observation examples

![Three forms of collapse in Cube observations](../assets/figures/collapse_examples.png)

From top to bottom:

1. A checker pattern changes the appearance while leaving the physical state unchanged.
2. Physically near and far states receive the wrong latent ordering.
3. Different actions lead to real futures whose ordering the predictor fails to preserve.

The future images are simulator observations. The model predicts latent states, not rendered images.
