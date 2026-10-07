# ColMod reproduction and efficient compression experiments

This branch keeps the original ColMod path under `llmesr_colmod` and adds the
new intent-aware compression path under `llmesr_intent_colmod`.

## Original ColMod reproduction

```bash
bash experiments/reproduce_colmod.bash fashion
```

The script uses `llmesr_colmod`, so it does not exercise the new intent-aware
modules. It is intended as the control run after code changes.

## New intent-aware compression model

```bash
bash experiments/efficient_compress.bash fashion
```

The script uses `llmesr_intent_colmod`, which adds:

- semantic-filtered collaborative graph: `relu(co_adj - lambda * semantic_adj)`
- per-user intent strength `exp(-gap_u)` between collaborative and semantic views
- trainable graph propagation without the legacy `no_grad` block
- per-user gap-weighted alignment for the 768-to-64 compression path
- dual recommendation branches: a trainable PCA-initialized ID embedding and a
  frozen LLM embedding with a trainable projection adapter both enter SASRec
- two fixed-length query tokens compress semantic-neighbor histories into
  content-dynamic collaborative tokens and prefix only the LLM branch
- semantic-neighbor self-distillation is retained on the LLM branch; the
  collaborative-neighbor distillation path is not used
- full attention is retained, and pointwise BCE is evaluated only at the last
  valid position of each anchor sequence

## Dataset prerequisites

Both scripts expect the handled dataset directory to contain:

- `inter.txt`
- `itm_emb_np.pkl`
- `pca64_itm_emb_np.pkl`
- `sim_user_100.pkl`
- `frequency.txt`

The original `llmesr_colmod` control additionally expects
`sim_user_collab_100.pkl` when its ID-view neighbor distillation is enabled.

Set environment variables when needed:

```bash
GPU_ID=1 SEEDS="42 43 44" bash experiments/efficient_compress.bash musical
```
