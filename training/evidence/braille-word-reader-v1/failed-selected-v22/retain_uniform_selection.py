"""Retain the failed single-average comparison without exporting any model."""
import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
WORK = ROOT / '.braille-training-work/word-scan-v1-20260905'
OUT = ROOT / 'training/evidence/braille-word-reader-v1/failed-selected-v22'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError('Existing evidence must remain immutable')
    selected = WORK / 'selected-v22'
    folder = WORK / 'run20-uniform-average-preparation'
    result = json.loads((selected / 'selected-validation-fullprecision.json').read_bytes())
    control = json.loads((WORK / 'selected-v15/selected-validation-fullprecision.json').read_bytes())
    assert result['checkpoint_sha256'] == control['checkpoint_sha256'] == sha(selected / 'best-word.keras')
    assert result['words'] == control['words'] and result['thresholds'] == control['thresholds']
    selection = json.loads((selected / 'checkpoint-selection.json').read_bytes())
    assert selection['external_outputs_used'] is False
    assert sha(selected / 'checkpoint-selection.json') == 'd25e1aaada9d20c8408701bead7a8d32ae4584a72174958f649139788e8a0e4c'
    audit = json.loads((folder / 'independent-audit.json').read_bytes())
    assert audit['status'] == 'passed' and audit['model_execution'] is False
    assert sha(folder / 'independent-audit.json') == 'f71e485c232d314df1d363de994ac77768be51d42211f17f8c8872e7f85e8cd3'
    for path, expected in audit['source_bindings'].items():
        assert sha(Path(path)) == expected
    paths = [*selected.glob('*.json'), *(selected / 'source').glob('*.py')]
    paths += [p for p in folder.rglob('*') if p.is_file() and p.suffix not in {'.keras', '.pdf', '.pyc'} and '__pycache__' not in p.parts]
    paths += [WORK / name for name in ('audit_uniform_average.py', 'select_uniform_after_run26.py')]
    paths += [WORK / 'run20' / name for name in ('training-config.json', 'sampling-lineage-final.json', 'sampling-history.json')]
    paths.append(Path(__file__).resolve())
    OUT.mkdir(parents=True)
    files = []
    for source in sorted(set(paths)):
        relative = source.relative_to(WORK)
        target = OUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        assert sha(target) == sha(source)
        files.append({'file': relative.as_posix(), 'sha256': sha(target), 'sizeBytes': target.stat().st_size})
    report = {
        'scope': 'The single predeclared uniform mean of eligible run20 epochs8/9/10 lost frozen whole-word validation to the unchanged v15 control. No external evaluation, export, activation, APK or physical evidence.',
        'candidateConstructionOnly': {'model_arrays': 280, 'preserved_bn_arrays': 208, 'uniform_mean_arrays': 72,
                                      'optimizer_datasets_removed': 146, 'inference_only': True},
        'candidateValidation': {'correct': 355, 'total': 405, 'accepted': 360, 'naturalCorrect': 129, 'stressCorrect': 226},
        'controlValidation': {'correct': 357, 'total': 405, 'accepted': 362, 'naturalCorrect': 130, 'stressCorrect': 227},
        'internalAcceptancePassed': False, 'physicalCameraValidated': False,
        'externalResultReference': '../failed-selected-v15/public-benchmark.json',
        'omissions': 'Candidate/parent Keras binaries and the cited paper PDF remain in the private work area. Their hashes and source URL are retained in the original plan. The original preparation index predates independent audit/evaluation files; this index binds the complete retained set.',
        'files': files,
    }
    target = OUT / 'archive.json'
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'files': len(files), 'bytes': sum(r['sizeBytes'] for r in files), 'archive_sha256': sha(target)}))


if __name__ == '__main__':
    main()
