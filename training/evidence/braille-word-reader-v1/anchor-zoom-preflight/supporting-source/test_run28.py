"""One source-bound focused integration suite; CPU only, no model construction."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

work=Path(__file__).resolve().parent;run=work/'run28';run.mkdir(exist_ok=True)
out=run/'preflight-tests.json';assert not out.exists()
os.environ.update(KERAS_BACKEND='torch',KERAS_TORCH_DEVICE='cpu',CUDA_VISIBLE_DEVICES='-1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
names=['test_word_anchor_zoom','test_word_pipeline','test_word_point_geometry','test_word_visible_partial','test_word_training_config','test_word_horizontal_aspect','test_word_vertical_stretch']
test_root=Path('training/tests');source=Path('training/scripts')
tree=ast.parse((source/'word_train.py').read_text())
sources=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TRAINING_SOURCES' for t in n.targets))
source_hashes={name:sha(source/name) for name in sources};test_hashes={name+'.py':sha(test_root/(name+'.py')) for name in names}
(run/'test-source').mkdir(exist_ok=True)
for name in test_hashes:(run/'test-source'/name).write_bytes((test_root/name).read_bytes())
sys.path.insert(0,str(test_root.resolve()))
start=datetime.now(timezone.utc).isoformat()
with (run/'preflight-tests.log').open('w',encoding='utf-8') as log:
    suite=unittest.TestLoader().loadTestsFromNames(names)
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
for name,digest in source_hashes.items():assert sha(source/name)==digest
for name,digest in test_hashes.items():assert sha(test_root/name)==digest
report={'status':'passed' if result.wasSuccessful() else 'failed','started_at_utc':start,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
 'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'scope':'Focused anchor cap integration, existing word/partial/point/config/horizontal/vertical regressions; CPU only. No model construction, inference or optimizer step.',
 'test_modules':names,'tests_sha256':test_hashes,'source_sha256':source_hashes,'test_source_before_after_exact':True,'log_sha256':sha(run/'preflight-tests.log'),'runner_sha256':sha(__file__)}
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
if not result.wasSuccessful():print((run/'preflight-tests.log').read_text());raise SystemExit(1)
