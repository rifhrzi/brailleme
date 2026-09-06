"""Focused guard/default tests and the actual loaded std.15 initializer guard."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

WORK=Path(__file__).resolve().parent;run=WORK/'run29';run.mkdir(exist_ok=True)
assert not (run/'preflight-tests.json').exists()
os.environ.update(KERAS_BACKEND='torch',KERAS_TORCH_DEVICE='cpu',CUDA_VISIBLE_DEVICES='-1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=Path('training/scripts');tests=Path('training/tests')
tree=ast.parse((source/'word_train.py').read_text());names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TRAINING_SOURCES' for t in n.targets))
source_hashes={n:sha(source/n) for n in names}
modules=['test_word_normalization_transition','test_word_training_config','test_word_anchor_zoom']
test_hashes={n+'.py':sha(tests/(n+'.py')) for n in modules};(run/'test-source').mkdir(exist_ok=True)
for n in test_hashes:(run/'test-source'/n).write_bytes((tests/n).read_bytes())
sys.path.insert(0,str(tests.resolve()));sys.path.insert(0,str(source.resolve()))
started=datetime.now(timezone.utc).isoformat()
with (run/'preflight-tests.log').open('w',encoding='utf-8') as log:
    suite=unittest.TestLoader().loadTestsFromNames(modules);result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
if not result.wasSuccessful():print((run/'preflight-tests.log').read_text());raise SystemExit(1)
import keras
import word_model
from word_normalization_transition import validated_normalization_transition
init=WORK/'normalization-std15-parent-v1/initialization-report.json';proof=WORK/'normalization-std15-runtime-v1/parity-report.json';parent=WORK/'normalization-std15-parent-v1/std15-parent.keras'
model=keras.models.load_model(parent,compile=False)
guard=validated_normalization_transition(model,parent,init,proof,word_model.image_normalization_spec(model))
assert guard['enabled'] and guard['target_std_after']==.15
for n,h in source_hashes.items():assert sha(source/n)==h
for n,h in test_hashes.items():assert sha(tests/n)==h
old=ast.parse((WORK/'run28/source/word_train.py').read_text());new=ast.parse((source/'word_train.py').read_text())
find=lambda t:next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Patches')
assert ast.dump(find(old))==ast.dump(find(new))
report={'status':'passed','started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'tests_run':result.testsRun,'failures':0,'errors':0,
 'source_sha256':source_hashes,'tests_sha256':test_hashes,'log_sha256':sha(run/'preflight-tests.log'),'runner_sha256':sha(__file__),
 'actual_loaded_initializer_guard':guard,'patches_ast_exact_run28':True,'model_loss_sources_unchanged':source_hashes['word_model.py']=='c446732a2905a612cd6f1dd64d2d3d01303b1c71933c34552d26f3b5052fd551',
 'scope':'CPU-only focused configuration/normalization/cap guards and actual loaded918d std.15 initializer. No inference or optimizer step in this test run; actual six-input inference belongs to the separate frozen runtime proof.'}
(run/'preflight-tests.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','tests_run':result.testsRun,'report_sha256':sha(run/'preflight-tests.json'),'actual_guard':guard}))
