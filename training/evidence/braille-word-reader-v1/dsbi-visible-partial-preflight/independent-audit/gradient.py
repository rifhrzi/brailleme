"""CPU autodiff of the unchanged real loss on an actual DSBI fragment target."""
import os
for name, value in {'KERAS_BACKEND':'torch', 'KERAS_TORCH_DEVICE':'cpu', 'CUDA_VISIBLE_DEVICES':'-1',
                    'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}.items(): os.environ[name] = value
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'training/scripts'))
from word_model import word_relative_loss
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
target = np.load(OUT/'gradient-target.npy')
prediction = torch.full((1,48,120,11), .5, requires_grad=True, device='cpu')
loss = word_relative_loss(torch.from_numpy(target[None]), prediction)
assert torch.isfinite(loss)
loss.backward()
gradient = prediction.grad[0].detach().numpy()
partials = (target[..., 0] == 1)&np.all(target[..., 5:] == -1, axis=-1)
complete = (target[..., 0] == 1)&np.all(target[..., 5:] >= 0, axis=-1)
assert partials.any() and complete.any()
assert not np.any(gradient[partials,5:])
assert np.all(np.abs(gradient[partials,0]) > 0)
assert np.all(np.abs(gradient[partials,1:5]).sum(axis=-1) > 0)
assert np.abs(gradient[complete,5:]).sum() > 0
report = dict(status='passed', scope='One actual saved DSBI target; unchanged word_relative_loss CPU autodiff. No model/checkpoint/inference/optimizer/GPU.',
              script_sha256=sha(__file__), loss_source_sha256=sha(ROOT/'training/scripts/word_model.py'),
              target_sha256=sha(OUT/'gradient-target.npy'), prediction_device=str(prediction.device),
              partial_centers=int(partials.sum()), complete_centers=int(complete.sum()),
              partial_dot_gradient_max_abs=float(np.abs(gradient[partials,5:]).max()),
              partial_objectness_gradient_min_abs=float(np.abs(gradient[partials,0]).min()),
              partial_geometry_gradient_min_l1=float(np.abs(gradient[partials,1:5]).sum(axis=-1).min()),
              complete_dot_gradient_l1=float(np.abs(gradient[complete,5:]).sum()), loss=float(loss.detach()))
(OUT/'gradient-report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report, indent=2))
