"""One private serialized normalization change; no Keras/backend/model execution."""
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import h5py
import numpy as np

OUT=Path(__file__).resolve().parent;WORK=OUT.parent;ROOT=WORK.parent.parent
PARENT=WORK/'selected-v15/best-word.keras'
PARENT_SHA='cce2432c0ddb8952c885dd58697589c285209588dafa81aacdd891b1660068b7'
TARGET=OUT/'std15-parent.keras'
START=datetime.now(timezone.utc).isoformat()

def sha(data):return hashlib.sha256(data).hexdigest()

def main():
    assert not TARGET.exists() and not (OUT/'initialization-report.json').exists()
    original_bytes=PARENT.read_bytes();assert sha(original_bytes)==PARENT_SHA
    with zipfile.ZipFile(io.BytesIO(original_bytes)) as archive:
        original_members={entry.filename:archive.read(entry.filename) for entry in archive.infolist()}
        original_infos=archive.infolist()
    assert set(original_members)=={'metadata.json','config.json','model.weights.h5'}
    original_config=json.loads(original_members['config.json'])
    config=json.loads(original_members['config.json'])
    matches=[(i,layer) for i,layer in enumerate(config['config']['layers']) if layer['class_name']=='CameraNormalization']
    assert len(matches)==1
    normalization_index,layer=matches[0]
    assert layer['registered_name']=='BrailleMe>CameraNormalization' and layer['config']['name']=='camera_normalization'
    assert layer['config']['target_std']==.05 and layer['config']['std_floor']==.001
    layer['config']['target_std']=.15
    assert config['compile_config']['optimizer']['class_name']=='Adam'
    config['compile_config']={}
    # A new HDF5 container omits the optimizer entirely (not only its links/free blocks).
    weights_stream=io.BytesIO()
    with h5py.File(io.BytesIO(original_members['model.weights.h5']),'r') as old:
        assert set(old.keys())=={'layers','optimizer','vars'}
        with h5py.File(weights_stream,'w') as new:
            for key,value in old.attrs.items():new.attrs[key]=value
            old.copy('layers',new)
            old.copy('vars',new)
    members=dict(original_members)
    members['config.json']=json.dumps(config,separators=(',',':'),allow_nan=False).encode()
    members['model.weights.h5']=weights_stream.getvalue()
    with zipfile.ZipFile(TARGET,'x') as archive:
        for entry in original_infos:archive.writestr(entry,members[entry.filename])
    artifact_sha=sha(TARGET.read_bytes())
    # Reuse the already-reviewed HDF5 reader functions, not its main/model-run code.
    inventory_source=WORK/'run27-parent-independent-audit/audit.py'
    inventory_bytes=inventory_source.read_bytes()
    assert sha(inventory_bytes)=='ceaa8908add7b37a0287521cc9df181bb0f4f14a1ec95ced0eb55665faf86fc8'
    tree=ast.parse(inventory_bytes)
    funcs=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('inventory','canonical')]
    assert len(funcs)==2
    namespace=dict(sha=sha,io=io,json=json,zipfile=zipfile,h5py=h5py,np=np,hashlib=hashlib)
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(inventory_source),'exec'),namespace)
    before=namespace['inventory'](PARENT,PARENT_SHA)
    after=namespace['inventory'](TARGET,artifact_sha)
    assert before['metadata']==after['metadata'] and before['arrays'].keys()==after['arrays'].keys()
    rows=[]
    for name in sorted(before['arrays']):
        a,b=before['arrays'][name],after['arrays'][name]
        assert a.tobytes()==b.tobytes()
        rows.append(dict(path=name,**before['metadata'][name],parent_sha256=sha(a.tobytes()),initialized_sha256=sha(b.tobytes()),identical_bytes=True))
    assert len(rows)==280 and sum(value.size for value in after['arrays'].values())==829307
    assert len([r for r in rows if r['category'].startswith('bn_')])==208
    assert before['moving_sha256']==after['moving_sha256']=='1f190a8fe5f4b3f7162147f7cf254a9fc437dfd20e0b34f3603003fffdc00ae7'
    assert not after['auxiliary'] and before['auxiliary'] and all(r['path'].startswith('optimizer/') for r in before['auxiliary'])
    expected=json.loads(original_members['config.json'])
    expected['config']['layers'][normalization_index]['config']['target_std']=.15
    expected['compile_config']={}
    assert after['config']==expected==config
    restored=json.loads(json.dumps(after['config']))
    restored['config']['layers'][normalization_index]['config']['target_std']=.05
    restored['compile_config']=original_config['compile_config']
    assert restored==original_config
    backbone=next(l for l in config['config']['layers'] if l['config']['name']=='mobilenetv2_half')
    assert backbone['config']['output_layers']==[['block_3_expand_relu',0,0],['block_6_expand_relu',0,0],['block_13_expand_relu',0,0],['out_relu',0,0]]
    assert all(n.get('kwargs',{}).get('training') is False for n in backbone['inbound_nodes'])
    assert any(l['config']['name']=='fpn_32_zero' for l in config['config']['layers'])
    assert not any(l['config']['name'].startswith(('fpn_2_','geometry_')) for l in config['config']['layers'])
    assert sha(PARENT.read_bytes())==PARENT_SHA
    with zipfile.ZipFile(TARGET) as archive:
        assert archive.read('metadata.json')==original_members['metadata.json']
        assert json.loads(archive.read('config.json'))==expected
        with h5py.File(io.BytesIO(archive.read('model.weights.h5')),'r') as handle:assert set(handle.keys())=={'layers','vars'}
    assert not any(name in sys.modules for name in ('keras','torch','tensorflow'))
    source=OUT/'source';source.mkdir()
    (source/'hdf5_inventory_audit.py').write_bytes(inventory_bytes)
    model_source=ROOT/'training/scripts/word_model.py'
    assert sha(model_source.read_bytes())=='c446732a2905a612cd6f1dd64d2d3d01303b1c71933c34552d26f3b5052fd551'
    (source/'word_model.py').write_bytes(model_source.read_bytes())
    (OUT/'parent-config.json').write_bytes(original_members['config.json'])
    (OUT/'initialized-config.json').write_bytes(members['config.json'])
    report=dict(schema_version=1,status='serialized_parent_and_hdf5_identity_passed_runtime_load_pending',
        started_at_utc=START,completed_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Private single-value initialization for a possible later ablation. No backend/model execution, training, selection, promotion, or active-source edits.',
        parent_checkpoint=str(PARENT),parent_checkpoint_sha256=PARENT_SHA,
        initialized_checkpoint=str(TARGET),initialized_checkpoint_sha256=artifact_sha,initialized_checkpoint_bytes=TARGET.stat().st_size,
        only_model_configuration_change={'path':f'config.layers[{normalization_index}].config.target_std','from':.05,'to':.15},
        optimizer_reset={'compile_config':{},'optimizer_group_absent':True,'removed_optimizer_arrays':len(before['auxiliary']),
            'removed_arrays':before['auxiliary'],'fresh_hdf5_container_written':True,
            'future_training':'Load with compile=False, construct fresh Adam and existing scheduler; this preparation does not execute those steps.'},
        original_config_sha256=sha(original_members['config.json']),initialized_config_sha256=sha(members['config.json']),
        parent_hdf5_sha256=sha(original_members['model.weights.h5']),initialized_hdf5_sha256=sha(members['model.weights.h5']),
        hdf5_byte_difference_expected='Optimizer datasets were omitted from a new container; all280 model arrays, names, shapes and dtypes remain exact.',
        model_arrays=280,model_parameters=829307,all_model_arrays_byte_identical=True,frozen_bn_arrays_byte_identical=208,
        moving_moments_sha256=after['moving_sha256'],full_tail_preserved=True,backbone_output_layers=backbone['config']['output_layers'],
        full_graph_equal_except_target_std_and_compile_reset=True,backbone_training_false_preserved=True,
        graph_contains_no_fine_geometry_or_bn_affine_branch=True,arrays=rows,
        source_sha256={'prepare.py':sha(Path(__file__).read_bytes()),'hdf5_inventory_audit.py':sha(inventory_bytes),'word_model.py':sha(model_source.read_bytes())},
        parent_metadata_bytes_retained=True,parent_metadata_note='Original Keras metadata date_saved is preserved; preparation timestamps are recorded in this report.',
        gradient_updates=0,backend_modules_imported=[],runtime_model_load_executed=False,output_equality_with_old_parent_claimed=False,
        output_parity_executed=False,trained_checkpoint_sampling_eligible=False,
        future_profile='Single embedded targetStd.15 intervention fromplainv15; anchor zoom cap and othernew ablations OFF. Any launch requires separate reviewed protocol.',
        limitations=['HDF5/config checks cannot establish successful Keras deserialization, save/load runtime equality or Torch/TF/TFLite parity.',
                     'The initializer intentionally changes normalization outputs; equality against oldv15 predictions is not a gate.',
                     'No TRAIN/held/physical pixels or outputs were read. No model accuracy or APK availability is claimed.'],
        versions={'python':sys.version.split()[0],'h5py':h5py.__version__,'numpy':np.__version__})
    (OUT/'initialization-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    files=[dict(file=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in sorted(OUT.rglob('*')) if p.is_file()]
    (OUT/'index.json').write_text(json.dumps(dict(files=files),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(artifact_sha256=artifact_sha,artifact_bytes=TARGET.stat().st_size,
        report_sha256=sha((OUT/'initialization-report.json').read_bytes()),index_sha256=sha((OUT/'index.json').read_bytes()),
        model_arrays=len(rows),optimizer_arrays_removed=len(before['auxiliary']))))

if __name__=='__main__':main()
