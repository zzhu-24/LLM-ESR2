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
- user-level intent gate between collaborative and semantic views
- trainable graph propagation without the legacy `no_grad` block
- gap-weighted pairwise alignment for the 768-to-64 compression path

## Dataset prerequisites

Both scripts expect the handled dataset directory to contain:

- `inter.txt`
- `itm_emb_np.pkl`
- `pca64_itm_emb_np.pkl`
- `sim_user_100.pkl`
- `sim_user_collab_100.pkl`
- `frequency.txt`

Set environment variables when needed:

```bash
GPU_ID=1 SEEDS="42 43 44" bash experiments/efficient_compress.bash musical
```
