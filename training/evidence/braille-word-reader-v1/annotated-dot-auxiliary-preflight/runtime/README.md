# Prepared auxiliary-dot backend preflight

Status: prepared and syntax-checked only. `runtime-preflight.py` has not been invoked; no Torch, TensorFlow or Keras backend was imported during preparation. The original private prototype v1 and active training/selector sources remain unchanged.

The frozen fixture uses combined TRAIN manifest indices **253, 21577, 22437, 23134, 23435, 6326**: the first q=1 member of the already frozen TRAIN activation cohort for one Angelina and four DSBI families, followed by the already frozen negative `train/006326.png` from the earlier geometry probe. Identities and prior image hashes were read from existing metadata before any new pixel read or inference. The batch cannot replace samples after observing outcomes. The fixture protocol hash is pinned inside the runtime script.

Twenty-five baseline dependency files are copied under `source/`, including `benchmark_angelina.py`, which the data module imports. The script also binds the four immutable prototype candidate files and their original index. It loads the exact plain v15 parent with SHA-256 `cce2432c0ddb8952c885dd58697589c285209588dafa81aacdd891b1660068b7`.

When scheduled, one CPU Torch process will:

1. Verify input/protocol/source hashes and rederive the existing DSBI point metadata from raw TRAIN annotations. Produce actual old/prototype Patches batches with auxiliary OFF and ON, keeping the same real Pillow augmentation, source sampler and dataset phase 2. Compare RGB, main target arrays, source order and augmentation RNG states exactly. Independently recompute query/bit correspondence from the actual transformed point state and explicit masks.
2. Check the shared reader graph, 280 arrays, 829307 parameters, full tail, frozen 52 BN layers and .05/.001 normalization before/after wrapper construction. Compare initial original/wrapper main outputs exactly.
3. Verify a nonzero auxiliary-only gradient into the shared fine tensor, its convolution and the new head, without updating weights. Verify exact zero auxiliary loss and gradients for an all-ignored finite-logit batch. Check actual compiled total loss equals main + .25 auxiliary, with a separate denominator.
4. Exercise actual validation mode on the two fixed TRAIN Angelina/negative rows with point states omitted. Check main arrays and main-only compiled/test-on-batch loss. No 135 held validation image or other held fixture is read.
5. Execute exactly one genuine optimizer step at the predeclared diagnostic LR 2e-5, check frozen BN arrays, then save/reload the original 11-channel reader and the full training wrapper/optimizer. Reader loading is tested with the auxiliary custom classes temporarily removed from the Keras registry. Full-wrapper state readability is checked; no CLI continuation or interruption-exact resume is claimed.

The runtime creates a fresh `results/` folder and refuses to overwrite it. Its one-step reader and wrapper are diagnostic artifacts only and must never initialize a training run or become an active model. This preflight does not run TF/Lite conversion, decoder calibration, word selection, external regression or a training cycle. Any eventually trained reader still needs the normal 405 parity and original accuracy/rejection gates.

After run30 and its selection finish, root can schedule the single command:

```powershell
& 'C:/bm-envs/word-scan-torch-only/Scripts/python.exe' '.braille-training-work/word-scan-v1-20260905/dot-auxiliary-runtime-preparation-v1/runtime-preflight.py' --execute
```

Preparation does not authorize running it alongside active training or adopting a new run. A runtime failure must be reported with the failing operation; this source-only readiness report does not establish runtime correctness.
