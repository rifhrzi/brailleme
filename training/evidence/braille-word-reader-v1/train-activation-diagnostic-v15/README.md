# Descriptive TRAIN activation audit

The unchanged v15 model processed 36 cached RGB patches, three from each of 12 positive source families (31 unique original photos), on CPU with training disabled. Sampling was fixed from manifest order before inspecting activations. No optimizer, gradients, model save, validation or external inputs were used.

All model arrays and the checkpoint file remained unchanged. The independent review checked input selection and hashes, the serialized normalization/backbone graph, 52 saved BatchNorm moment arrays, and the derived channel statistics and ReLU6 counts. It did not rerun deep activations; raw activation tensors were not retained.

| Layer/statistic | Observed value |
| --- | ---: |
| First BN, median observed standard deviation / stored standard deviation | 0.1441 |
| Block 13 depthwise BN, median absolute mean difference / stored standard deviation | 3.1754 |
| Block 13 depthwise BN, median observed / stored standard deviation | 3.1258 |
| Block 13 depthwise ReLU6 values at the upper limit | 19.23% |

Here “stored standard deviation” includes epsilon: `sqrt(moving_variance + epsilon)`. Statistics pool spatial positions, including background and verso dots. The small, deterministic sample includes overlapping crops and excludes training augmentation and negatives. Differences and saturation are descriptive; they do not establish a recognition cause or an accuracy improvement.

These results motivate evaluating learnable BN scale/offset with fixed moving moments as a separate experiment. Merely setting a nested backbone call to `training=False` is insufficient proof of that behavior: a training-mode forward/backward test must show that moving statistics remain fixed. This diagnostic did not change the vertical-stretch experiment.

Files contain the sampling protocol, original audit script/report, and independent review script/report. No model or APK is included. The v15 recognition gates remain failed.
