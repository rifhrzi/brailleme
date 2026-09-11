"""Prepare actual TFLite image/output decisions for the TypeScript parity replay."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from word_decode import decode
from word_symbols import DECODER_VERSION,SYMBOL_PROTOCOL
from word_benchmark import quality
from word_data import save_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_oracle_binding(archive,model_hash,checkpoint_hash,fixture_hash):
    expected=dict(oracle_kind='tflite',model_sha256=model_hash,checkpoint_sha256=checkpoint_hash,fixture_sha256=fixture_hash)
    for name,value in expected.items():
        if name not in archive.files or str(archive[name])!=value:
            raise ValueError('Cached TFLite oracle identity differs or is missing: '+name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=Path('.'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--oracle',type=Path)
    parser.add_argument('--plain-latin-lowercase',action='store_true')
    args=parser.parse_args()
    if (args.output/'python-reference.json').exists():
        raise SystemExit('Use a new parity output folder; frozen proof is not overwritten')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.fixtures.read_text());records=manifest['records']
    selection=json.loads((args.run/'selected-validation-fullprecision.json').read_text())
    thresholds=selection['thresholds']
    model_path=args.run/'word_reader.tflite'
    if args.oracle:
        with np.load(args.oracle) as archive:
            validate_oracle_binding(archive,sha(model_path),selection['checkpoint_sha256'],sha(args.fixtures))
            outputs=archive['outputs']
        if outputs.shape!=(len(records),48,120,11) or outputs.dtype!=np.float32:
            raise ValueError('Cached oracle shape differs from fixture rows')
    else:
        import tensorflow as tf
        model=tf.lite.Interpreter(model_path=str(model_path),num_threads=4);model.allocate_tensors()
        inp=model.get_input_details()[0]['index'];out=model.get_output_details()[0]['index']
        outputs=[]
    rows=[]
    with (args.output/'inputs.rgb').open('wb') as stream:
        for index,record in enumerate(records):
            with Image.open(args.data/record['image']) as source:
                image=source.convert('RGB').resize((480,192),Image.Resampling.BILINEAR)
                stream.write(image.tobytes());pixels=np.asarray(image,np.float32)/255.
            if args.oracle:
                output=outputs[index]
            else:
                model.set_tensor(inp,pixels[None]);model.invoke();output=model.get_tensor(out)[0].copy();outputs.append(output)
            if not np.isfinite(output).all():
                raise ValueError('Non-finite model output in parity case')
            result=decode(output,thresholds,record.get('selection'));precheck=quality(pixels)
            text,reason=('',precheck) if precheck else (result['text'],result['reason'])
            expected=record['expected'].lower() if args.plain_latin_lowercase else record['expected']
            row=dict(index=index,image=record['image'],source=record.get('source',record['image']),expected=expected,expected_reject=record.get('expected_reject',not bool(expected)),selection=record.get('selection',[0.,0.,1.,1.]),python_text=text,python_reason=reason,python_quality_reason=precheck,python_decoder_text=result['text'],python_decoder_reason=result['reason'],saved_output_is_zero_placeholder=False)
            if result['text']:
                row['python_cells']=[dict(mask=cell['mask'],role=cell['role'],output=cell['output'],box=cell['box']) for cell in result['cells']]
            rows.append(row)
    outputs=np.asarray(outputs,np.float32)
    outputs.astype('<f4').tofile(args.output/'outputs.f32le')
    np.savez_compressed(args.output/'outputs.npz',outputs=outputs,oracle_kind='tflite',model_sha256=sha(model_path),checkpoint_sha256=selection['checkpoint_sha256'],fixture_sha256=sha(args.fixtures))
    root=Path(__file__).resolve().parents[2]
    sources=['training/scripts/'+name for name in ['word_symbol_replay.py','word_decode.py','word_symbols.py','word_data.py','word_benchmark.py']]
    report=dict(schema_version=1,scope='Actual TFLite model outputs and image-quality Python decisions; no physical camera proof',decoderVersion=DECODER_VERSION,symbolProtocol=SYMBOL_PROTOCOL,model_sha256=sha(model_path),thresholds=thresholds,fixture_manifest_sha256=sha(args.fixtures),raw_outputs_sha256=sha(args.output/'outputs.f32le'),raw_inputs_sha256=sha(args.output/'inputs.rgb'),cached_npz_sha256=sha(args.oracle) if args.oracle else sha(args.output/'outputs.npz'),source_sha256={name:sha(root/name) for name in sources},rows=rows)
    save_json(args.output/'python-reference.json',report)
    print(json.dumps(dict(cases=len(rows),accepted=sum(bool(row['python_text']) for row in rows),correct=sum(bool(row['expected']) and row['expected']==row['python_text'] for row in rows))),flush=True)


if __name__=='__main__':
    main()
