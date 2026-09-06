"""One disposable GPU backward batch; no optimizer, save, selection or activation."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
sys.path.insert(0,str(Path('training/scripts').resolve()))
import keras,torch
from word_train import Patches
from word_sources import audit_supplemental_sources,expand_negative_records
from word_point_metadata import load_point_states
from word_model import word_relative_loss
import word_geometry_head

torch.set_num_threads(4)
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.allow_tf32=True
torch.backends.cuda.matmul.allow_tf32=False
keras.utils.set_random_seed(20260905)
root=Path('.braille-training-work/word-scan-v1-20260905')
data=root/'data'
parent=root/'geometry-head-parent-v1/geometry-parent.keras'
before=hashlib.sha256(parent.read_bytes()).hexdigest()
records=json.loads((root/'combined-training-v5/combined-training-manifest.json').read_text())
audit=audit_supplemental_sources(records,data,'../complete-source-training-v3/combined-training-manifest.json',['../dsbi-training-contexts-v2/source-index.json'])
states,point_audit=load_point_states('../dsbi-point-metadata-v1/points.json',data,['../dsbi-training-contexts-v2/source-index.json'],records)
records,negative=expand_negative_records(records,27,audit['supplemental_sources'],2)
patches=Patches(data,records,24,True,20260905,illumination_augmentation=True,balance_source_families=True,point_states=states)
patches.on_epoch_end() # actual first-fit dataset epoch2, phase1
images,labels=patches[0]
model=keras.models.load_model(parent,compile=False)
assert model.weights[0].value.device.type=='cuda'
torch.cuda.reset_peak_memory_stats()
start=time.perf_counter()
prediction=model(images,training=True)
assert bool(torch.isfinite(prediction).all())
loss=word_relative_loss(torch.as_tensor(labels,device='cuda'),prediction)
assert bool(torch.isfinite(loss))
loss.backward()
torch.cuda.synchronize()
elapsed=time.perf_counter()-start
gradients=[v.value.grad for v in model.trainable_weights if v.value.grad is not None]
assert gradients and all(bool(torch.isfinite(v).all()) for v in gradients)
assert hashlib.sha256(parent.read_bytes()).hexdigest()==before
report={'status':'passed','scope':'One actual phase1 batch24 CUDA backward, no optimizer or checkpoint save; all tensors finite. This is a memory/execution probe, not training or accuracy evidence.',
        'parent_sha256':before,'batch_size':24,'dataset_epoch':patches.epoch,'loss':float(loss.detach().cpu()),
        'elapsed_seconds':elapsed,'max_allocated_bytes':torch.cuda.max_memory_allocated(),'max_reserved_bytes':torch.cuda.max_memory_reserved(),
        'device':torch.cuda.get_device_name(0),'gradient_tensors':len(gradients),'negative_expansion':negative,
        'point_metadata_sha256':point_audit['sha256'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256':{name:hashlib.sha256((Path('training/scripts')/name).read_bytes()).hexdigest() for name in ('word_train.py','word_geometry_head.py','word_model.py','word_data.py')}}
(root/'geometry-head-parent-v1/gpu-batch-preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
