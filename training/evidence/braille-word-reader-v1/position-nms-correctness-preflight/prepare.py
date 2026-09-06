"""Freeze a private positional-NMS-only decoder prototype; no active edits."""
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
SOURCES=['training/scripts/word_decode.py','training/scripts/word_select.py',
         'training/scripts/word_data.py','training/scripts/word_model.py',
         'training/scripts/word_anchor_zoom.py','training/scripts/benchmark_angelina.py',
         'src/ml/word-decoder.ts','src/ml/word-model-contract.ts','src/types/word-scan.ts',
         'src/types/live-scan.ts']
sha=lambda b:hashlib.sha256(b).hexdigest()
assert not (OUT/'prototype-spec.json').exists()
hashes={}
for name in SOURCES:
    data=(ROOT/name).read_bytes();hashes[name]=sha(data)
    for side in ['source-original','source-private']:
        p=OUT/side/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
py_path=OUT/'source-private/training/scripts/word_decode.py'
py=py_path.read_text()
start=py.index('        neighborhood = output[')
end=py.index('        cx, cy = ',start)
py=py[:start]+py[end:]
start=py.index('    """Cache exact local maxima once;')
end=py.index('\n\n\ndef calibrate(',start)
py=py[:start]+'''    """Cache only the minimum score filter; preserve every positional proposal.

    All calibration trials must use proposalScore >= minimum_proposal. Geometry,
    dots and malformed rows remain exact. Never apply local-grid suppression.
    """
    prepared=np.asarray(predictions).copy()
    score=prepared[...,0]
    valid=np.isfinite(prepared).all(axis=(1,2,3)) & (prepared.min(axis=(1,2,3))>=0) & (prepared.max(axis=(1,2,3))<=1)
    prepared[...,0]=np.where(valid[:,None,None],np.where(score>=minimum_proposal,score,0),score)
    return prepared'''+py[end:]
py_path.write_text(py,encoding='utf-8')
ts_path=OUT/'source-private/src/ml/word-decoder.ts'
ts=ts_path.read_text();start=ts.index('      let isMaximum = true;');end=ts.index('      const width = ',start)
ts=ts[:start]+ts[end:];ts_path.write_text(ts,encoding='utf-8')
patch=[]
for name in ['training/scripts/word_decode.py','src/ml/word-decoder.ts']:
    before=(OUT/'source-original'/name).read_text().splitlines(keepends=True)
    after=(OUT/'source-private'/name).read_text().splitlines(keepends=True)
    patch.extend(difflib.unified_diff(before,after,fromfile='original/'+name,tofile='private/'+name))
(OUT/'private-edits.patch').write_text(''.join(patch),encoding='utf-8')
spec={'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Private correctness prototype only; no active source, model, fixture or gate changes.',
    'policy':'Every score-qualified proposal intersecting the guide enters the existing score-ordered positional IoU NMS. Remove 3x3 local-grid competition only. Equal scores keep original grid-index ordering.',
    'calibration':'Cache may discard scores below the lowest trial proposalScore only; it must retain every qualifying nonmaximum. Previously peak-filtered caches cannot be reused.',
    'thresholds':'Unchanged existing thresholds; synthetic weak-cell case uses a declared acceptScore.4 to test rejection between proposal and acceptance gates.',
    'source_original_sha256':hashes,'source_private_sha256':{n:sha((OUT/'source-private'/n).read_bytes()) for n in SOURCES},
    'constraints':['No405 replay before explicit root authorization; no607 outputs, inference, training or threshold tuning.',
        'CenterNet-style focal treats noncenter Gaussian sites as downweighted negatives, not soft-positive centers; their box/dot channels have no direct center supervision.',
        'Additional proposals may expose poorly predicted geometry/bits and increase rejections or runtime. Synthetic contract correctness is not accuracy improvement.',
        'The live decoder identifier/asset contract remains unchanged because these private sources are not integrated. Promotion would need a distinct audited decoder revision.']}
(OUT/'prototype-spec.json').write_text(json.dumps(spec,indent=2)+'\n')
print(json.dumps({'status':'prepared','spec_sha256':sha((OUT/'prototype-spec.json').read_bytes()),'private_python_sha256':spec['source_private_sha256']['training/scripts/word_decode.py'],'private_ts_sha256':spec['source_private_sha256']['src/ml/word-decoder.ts']},indent=2))
