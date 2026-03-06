# AMPeLM Pipeline

This repository contains a peptide classification pipeline built around an ESM2 feature extractor and four downstream models.

## Script Map

- `ESM2_AMP.py`
  - Trains/fine-tunes ESM2 binary classifier with 5-fold CV.
  - Produces ESM2 weights used as feature backbone.
- `ESM2_AMP_RNN.py`
  - Trains the RNN downstream classifier.
- `ESM2_AMP_GNN.py`
  - Trains the GNN downstream classifier.
- `ESM2_AMP_GraphSAGE.py`
  - Trains the GraphSAGE downstream classifier.
- `COMDEL.py`
  - Trains the CNN (sequence-only) downstream classifier.
- `integration.py`
  - Runs 4-model inference and strict fusion (`positive` only if all 4 models predict positive).
- `integration2.py`
  - Builds stacking features (4 model positive probabilities) and trains/predicts with XGBoost.

## Expected Data Columns

Input CSV must contain:
- `seq` (or `sequence`)
- `label` (0/1)

Optional:
- `id` (auto-generated if missing)

## Main Outputs

- `single_inference_result_4models/*_4models_inference_result.csv`
- `xgboost_ensemble_result/*.csv`
- `xgboost_ensemble_result/xgboost_ensemble_model.json`

Model checkpoints are searched in both styles:
- `model/results_ultra_light_*/...`
- `results_ultra_light_*/...`

## Run

```bash
python integration.py
python integration2.py
```

## Common Failure Checks

1. Missing checkpoint file
   - Verify required model weights exist under either `model/...` or project-root output dirs.
2. Empty dataset after filtering
   - Ensure `seq`/`sequence` is non-empty and `label` is valid 0/1.
3. XGBoost training fails with one class
   - Ensure training CSV contains both positive and negative samples.

## Environment

Typical dependencies:
- `torch`
- `torch-geometric`
- `transformers`
- `pandas`, `numpy`, `scikit-learn`, `tqdm`
- `xgboost` (for `integration2.py`)

Compatibility note:
- Keep binary package versions aligned.
- If you see errors like `numpy.core.multiarray failed to import`, use a consistent stack (for example, pin `numpy<2` in an environment where `pandas/scikit-learn/pyarrow` were built against NumPy 1.x, or upgrade all those packages together).
