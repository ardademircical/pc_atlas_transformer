# Single-cell Prostate Cancer Atlas Transformer (Work in Progress)

Prostate cancer-specific single-cell RNA-seq language models. 

Configs include: 

- PCA-paired fully-supervised cell annotation model. ✅
- Ranked gene identity prediction-based self-supervised model ⏳
- Gene expression prediction-based self-supervised model (SSL task: regression) ⏳

# Updates:

* March 8, 2024: Ranked gene identity model almost finished. Fixes required for `GeneExpressionDataset`, training loop incomplete.

* March 22, 2024: Ranked gene identity model with RMSNorm + Swish activation. Training loop, collator, tokenizer updates.
* March 23, 2024: Mixed precision training and dataset caching for ranked gene identity model.


Estimated timeline: all building/training should finish by end of March
