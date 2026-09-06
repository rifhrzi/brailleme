# Run25 epoch 1: actual weight update

This separate postlaunch HDF5 inspection preserves all 104 moving arrays and 104 frozen BN affine arrays. All 52 inherited convolution arrays and 20 inherited head arrays changed. The new depthwise kernel changed 107/144 values; the initially zero lateral now has 805/1024 nonzero kernel values and 62/64 nonzero bias values.

This is evidence that the branch receives training updates, not recognition accuracy or a passing checkpoint. Epoch 1 is not eligible for selection. No model inference, native execution, APK, or physical-camera test was performed by this inspection.

The original report, audit source and exact validation-history snapshot are retained in this folder. No model binary is included.
