"""Create an explicit normalization wrapper without optimizing any model weight."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ['KERAS_BACKEND']='torch'
import keras
import torch
from word_model import build_model, image_normalization_spec
from word_data import save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    records=[]
    for path in args.checkpoints:
        original_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        target=args.output/(original_hash[:12]+'-std05.keras')
        if target.exists():
            raise SystemExit('Do not overwrite an existing normalization wrapper')
        original=keras.models.load_model(path,compile=False)
        if image_normalization_spec(original)['kind']!='none':
            raise SystemExit('Expected a raw parent model; do not normalize it twice')
        weights=original.get_weights()
        model=build_model(weights=None,image_normalization='channel_gray_std05')
        model.set_weights(weights)
        import numpy as np
        if not all(np.array_equal(a,b) for a,b in zip(weights,model.get_weights())):
            raise AssertionError('Normalization wrapper changed a learned parameter')
        model.save(target)
        record={'file':target.name,'checkpoint_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                'parent_checkpoint':str(path),'parent_checkpoint_sha256':original_hash,
                'gradient_updates':0,'all_learned_parameters_identical':True,
                'imageNormalization':image_normalization_spec(model)}
        records.append(record)
        print(json.dumps(record),flush=True)
        del model,original,weights
        keras.backend.clear_session()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    sources={}
    for name in ('word_wrap_normalization.py','word_model.py'):
        data=(Path(__file__).parent/name).read_bytes()
        (args.output/name).write_bytes(data)
        sources[name]=hashlib.sha256(data).hexdigest()
    save_json(args.output/'normalization-wrappers.json',{'scope':'Untrained normalization ablation; held validation determines selection before any external regression.','source_sha256':sources,'records':records})


if __name__=='__main__':
    main()
