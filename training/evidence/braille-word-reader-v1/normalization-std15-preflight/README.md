# Run29 embedded normalization preflight

The actual initializer changes only CameraNormalization.target_std from0.05 to0.15, retaining all280 model arrays and the full-tail graph. Source24, actual configuration/protocol,21 source-bound prelaunch tests, independent guard review, and six fixed-input Torch/TF/Lite runtime evidence are retained. Nonconstant outputs change by design; runtime parity compares that same std15 initializer across runtimes.

The postlaunch-epoch01 material is explicitly later evidence:208 BatchNorm arrays remain exact while convolution/head arrays changed. Epoch1 is not sampling eligible and does not establish recognition improvement. No active model contract or release gate was changed by this retainer.

preflight.json maps each role to exact bytes. Large JSON inventories/configs use lossless gzip with raw hashes. Model binaries and raw tensors are omitted with original paths, measured sizes and verified hashes. Existing DSBI data/point archives are linked rather than copied. All retained references resolve within the repository; original reports preserve historical paths and wording. archive-index.json hashes every new payload. Full model replay additionally requires the explicitly omitted artifacts.
