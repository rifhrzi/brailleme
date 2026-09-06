"""Select retained checkpoints only on the final, reviewed validation fixture set."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
from PIL import Image
import torch
import keras
from word_model import word_loss, image_normalization_spec
from word_decode import calibrate, summarize, DEFAULT_THRESHOLDS, prepare_calibration_outputs, DECODER_VERSION
from word_data import save_json
import word_geometry_head  # register optional serialized sampling layers
import word_bn_affine  # register fixed-moment BN for trained affine candidates


THRESHOLD_GRIDS = {
    'default-v1': {'proposal': (.15, .25, .35), 'accept': (.4, .55, .65, .75), 'dot': (.7, .8, .9, .95)},
    'weak-peaks-v2': {'proposal': (.15, .25, .35), 'accept': (.2, .25, .3, .35, .4, .55, .65, .75), 'dot': (.7, .8, .9, .95)},
}


def grouped_calibration(predictions, records, threshold_grid='default-v1'):
    predictions=prepare_calibration_outputs(predictions)
    grid=THRESHOLD_GRIDS[threshold_grid]
    indices={group:[i for i,row in enumerate(records) if row['validation_group']==group] for group in ('natural','stress')}
    if any(not values for values in indices.values()):
        raise ValueError('Grouped selection requires both natural and stress validation')
    best=None
    for proposal in grid['proposal']:
        for accept in grid['accept']:
            if accept < proposal:
                continue
            for dot in grid['dot']:
                thresholds=dict(DEFAULT_THRESHOLDS,proposalScore=proposal,acceptScore=accept,dotConfidence=dot)
                groups={group:summarize(predictions[chosen],[records[i] for i in chosen],thresholds) for group,chosen in indices.items()}
                minimum_precision=min(value['accepted_accuracy'] or 0. for value in groups.values())
                mean_exact=float(np.mean([value['exact_word_accuracy'] for value in groups.values()]))
                eligible=minimum_precision>=.95
                rank=(eligible,mean_exact if eligible else minimum_precision,mean_exact)
                if best is None or rank>best[0]:
                    best=(rank,thresholds,groups)
    rank,thresholds,groups=best
    return thresholds,summarize(predictions,records,thresholds),groups,rank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-manifest", default="validation-words.json")
    parser.add_argument("--grouped-selection", action="store_true")
    parser.add_argument('--threshold-grid', choices=tuple(THRESHOLD_GRIDS), default='default-v1', help='Frozen validation-only threshold search. weak-peaks-v2 additionally tests accepted objectness .2/.25/.3/.35; both group precision gates remain >=95 percent.')
    args = parser.parse_args()
    if not args.grouped_selection and args.threshold_grid != 'default-v1':
        parser.error('The extended threshold grid requires grouped validation selection')
    if (args.output/"best-word.keras").exists():
        raise SystemExit("Use a new output directory; a frozen checkpoint must not be overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    source_folder=args.output/'source'
    source_folder.mkdir(exist_ok=True)
    source_hashes={}
    for filename in ('word_select.py','word_model.py','word_decode.py','word_data.py','word_geometry_head.py','word_bn_affine.py','word_anchor_zoom.py'):
        contents=(Path(__file__).parent/filename).read_bytes()
        (source_folder/filename).write_bytes(contents)
        source_hashes[filename]=hashlib.sha256(contents).hexdigest()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    fixture_path = args.data/args.validation_manifest
    fixture_bytes = fixture_path.read_bytes()
    fixture_hash = hashlib.sha256(fixture_bytes).hexdigest()
    records = json.loads(fixture_bytes)["records"]
    images = np.stack([np.asarray(Image.open(args.data/row["image"]).convert("RGB"), np.float32)/255. for row in records])
    luminance=images@np.asarray([.2126,.7152,.0722],np.float32)
    means,stds=luminance.mean(axis=(1,2)),luminance.std(axis=(1,2))
    quality_rejected=(means<.06)|(means>.985)|(stds<.01)
    del luminance
    candidates, hashes, best = [], set(), None
    for path in args.candidates:
        checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if checkpoint_hash in hashes:
            continue
        hashes.add(checkpoint_hash)
        model = keras.models.load_model(path, compile=False)
        predictions = model.predict(images, batch_size=args.batch_size, verbose=0)
        if not np.isfinite(predictions).all():
            raise SystemExit(f"Non-finite candidate outputs: {path}")
        scoring_outputs=predictions.copy()
        scoring_outputs[quality_rejected]=0
        groups=None
        if args.grouped_selection:
            thresholds,metrics,groups,rank=grouped_calibration(scoring_outputs,records,args.threshold_grid)
        else:
            thresholds, metrics = calibrate(scoring_outputs, records)
            precision = metrics["accepted_accuracy"] or 0.
            rank = (precision>=.95, metrics["exact_word_accuracy"] if precision>=.95 else precision, metrics["correct"])
        candidate = {"source_checkpoint": str(path), "checkpoint_sha256": checkpoint_hash, "thresholds": thresholds, "words": metrics, "decoder_version": DECODER_VERSION}
        candidate['imageNormalization']=image_normalization_spec(model)
        if groups is not None:
            candidate.update(validation_groups=groups,equal_group_mean_exact=float(np.mean([value['exact_word_accuracy'] for value in groups.values()])),both_group_precision_gates_pass=rank[0])
        candidates.append(candidate)
        brief={key:candidate[key] for key in ('source_checkpoint','checkpoint_sha256','thresholds','imageNormalization')}
        brief['words']={key:metrics[key] for key in ('total','accepted','correct','exact_word_accuracy','accepted_accuracy')}
        if groups is not None:
            brief['validation_groups']={name:{key:value[key] for key in ('total','accepted','correct','exact_word_accuracy','accepted_accuracy')} for name,value in groups.items()}
            brief['equal_group_mean_exact']=candidate['equal_group_mean_exact']
        print(json.dumps(brief), flush=True)
        if best is None or rank>best[0]:
            best = (rank, candidate, predictions.copy(), path)
        del model, predictions, scoring_outputs
        keras.backend.clear_session()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if fixture_path.read_bytes() != fixture_bytes:
        raise SystemExit("Validation fixture manifest changed during checkpoint comparison")
    _, chosen, outputs, path = best
    shutil.copyfile(path, args.output/"best-word.keras")
    oracle = {"backend": "torch", "tf32_enabled": False, "fixture_sha256": fixture_hash, "fixture_file":args.validation_manifest,"checkpoint_sha256": chosen["checkpoint_sha256"],"source_sha256":source_hashes,"decoder_version":DECODER_VERSION}
    selected = dict(chosen, oracle=oracle)
    selected['quality_rejected_before_decode']=int(quality_rejected.sum())
    save_json(args.output/"selected-validation.json", selected)
    save_json(args.output/"selected-validation-fullprecision.json", selected)
    np.savez_compressed(args.output/"backend-fullprecision.npz", outputs=outputs, quality_rejected=quality_rejected,fixture_sha256=fixture_hash, checkpoint_sha256=chosen["checkpoint_sha256"])
    rule="Require accepted accuracy >=95% separately on natural and stress groups, then maximize equal-group mean whole-word exactness including quality/decoder rejections." if args.grouped_selection else "Prefer accepted accuracy >=95%, then maximum exact whole-word accuracy (rejections count as errors)."
    save_json(args.output/"checkpoint-selection.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(), "decoder_version":DECODER_VERSION,"fixture_sha256": fixture_hash, "validation_words": len(records), "selection_rule": rule+" All candidates use identical frozen validation inputs and a full-float32 oracle.", "threshold_grid":args.threshold_grid,"threshold_grid_values":THRESHOLD_GRIDS[args.threshold_grid],"candidates": candidates, "selected_checkpoint": chosen["source_checkpoint"], "selected_checkpoint_sha256": chosen["checkpoint_sha256"], "external_outputs_used": False})
    print("FROZEN_SELECTED "+json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
