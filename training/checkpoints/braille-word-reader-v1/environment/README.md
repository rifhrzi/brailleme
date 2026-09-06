# Braille word training environment (Torch only)

Created on 2026-09-05. This environment deliberately cannot import TensorFlow.

Python executable: `C:\bm-envs\word-scan-torch-only\Scripts\python.exe` (3.12.14).

`Lib\site-packages\braille-torch-reuse.pth` is one plain path line:
`C:\bm-envs\word-scan\Lib\site-packages`.
Python appends that directory without processing its nested `.pth` files; the
old environment's reference to `angelina-benchmark` does not appear in `sys.path`.
Torch 2.7.1+cu128 is reused from this directory, so retain the old Torch package.
Keras, NumPy, Pillow, h5py, and all required supporting packages have their own
copies in this environment. `requirements-installed.txt` records exact versions.
`python -m pip check` passed with no broken requirements.

Set `KERAS_BACKEND=torch` for training before importing Keras. No default backend
or CUDA visibility settings are persisted by this environment.

Known working smoke from PowerShell:

```powershell
& 'C:\bm-envs\word-scan-torch-only\Scripts\python.exe' 'C:\bm-envs\word-scan-torch-only\smoke-word-env.py'
```

This script builds the normalized architecture without downloading weights,
loads the existing `e79046508c2a-std05.keras` wrapper, runs an explicit CPU
inference, saves a copy only inside this environment, reloads it, and verifies
an identical output (maximum absolute difference 0). It checks the original
checkpoint SHA256 remains unchanged. The final smoke completed in 12.51 seconds,
peaked at 1,030,156,288 bytes working set, and ended at 711,798,784 bytes RSS.

The smoke explicitly sets `CUDA_VISIBLE_DEVICES=-1` inside its own process;
Keras can otherwise initialize CUDA even around CPU-scoped build/load operations.
It therefore does not change training GPU visibility. A separate Torch-only
availability probe recorded CUDA 12.8 available without initializing CUDA.
No TensorFlow module was available or imported in either successful probe.

Evidence: `smoke-report.json`, `cuda-availability.json`, `smoke-word-env.py`,
`requirements-installed.txt`. These are environment diagnostics, not model
accuracy or physical-device validation evidence.
