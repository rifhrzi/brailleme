"""Append-only run28 source/target-proof retention; stdlib only, no model execution."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

WORK = Path(__file__).resolve().parent
ROOT = WORK.parent.parent
EVIDENCE = ROOT / 'training/evidence/braille-word-reader-v1'
OUT = EVIDENCE / 'anchor-zoom-preflight'
RUN = WORK / 'run28'
V2 = WORK / 'anchor-preserving-zoom-prototype-v2'
V1 = WORK / 'anchor-preserving-zoom-prototype-v1'
REVIEW = WORK / 'anchor-zoom-independent-review'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checked(path, expected=None):
    data = Path(path).read_bytes()
    if expected is not None:
        assert digest(data) == expected, path
    return data


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def main():
    assert not OUT.exists(), 'Archive is immutable; choose a new version instead.'
    config = read(RUN / 'training-config.json')
    protocol = read(RUN / 'anchor-zoom-protocol.json')
    tests = read(RUN / 'preflight-tests.json')
    spec = read(V2 / 'prototype-spec.json')
    proof = read(V2 / 'proof-report.json')
    review = read(REVIEW / 'report.json')
    launch = read(RUN / 'launch-command.json')
    integration = read(V2 / 'integration-record.json')
    assert config['source_sha256'] == protocol['source_sha256'] == tests['source_sha256']
    assert len(config['source_sha256']) == 23
    assert config['data_sha256'] == protocol['data_sha256']
    assert config['resume_sha256'] == protocol['parent_checkpoint_sha256'] == spec['baseline_checkpoint_sha256']
    assert config['resume_sha256'] == 'cce2432c0ddb8952c885dd58697589c285209588dafa81aacdd891b1660068b7'
    assert config['anchor_zoom_cap'] and config['illumination_augmentation']
    assert config['dsbi_visible_partial_supervision']['enabled']
    for field in ('horizontal_aspect_augmentation', 'vertical_stretch_augmentation',
                  'horizontal_mirror_augmentation', 'camera_augmentation',
                  'half_turn_augmentation', 'polarity_augmentation'):
        assert not config[field] and not protocol[field]
    for field in ('partial_geometry_supervision', 'geometry_conditioned_residual',
                  'batchnorm_affine', 'fine_detail_branch'):
        assert not config[field]['enabled'] and not protocol[field]
    assert tests['status'] == proof['status'] == review['status'] == 'passed'
    assert tests['tests_run'] == 71 and tests['failures'] == tests['errors'] == 0
    assert tests['test_source_before_after_exact']
    assert datetime.fromisoformat(tests['completed_at_utc']) < datetime.fromisoformat(config['started_at_utc'])
    assert datetime.fromisoformat(protocol['created_at_utc']) < datetime.fromisoformat(config['started_at_utc'])
    assert len(protocol['preflight_audits']) == 2
    for audit in protocol['preflight_audits']:
        checked(audit['path'], audit['sha256'])
    checked(REVIEW / 'report.json', 'b6942349e922dc962e08b26ef0c67b817964e4b1c6b35576a3c62783fc1a9f3c')
    checked(REVIEW / 'audit.py', review['script_sha256'])
    checked(V2 / 'prototype-spec.json', protocol['private_prototype']['spec_sha256'])
    checked(V2 / 'proof-report.json', protocol['private_prototype']['proof_sha256'])
    checked(V2 / 'integration-record.json', protocol['private_prototype']['integration_sha256'])
    checked(V2 / 'proof.py', proof['script_sha256'])
    checked(RUN / 'anchor-zoom-protocol.json', launch['protocol_sha256'])
    checked(WORK / 'launch_run28.py', launch['launcher_sha256'])
    checked(WORK / 'test_run28.py', tests['runner_sha256'])
    checked(RUN / 'preflight-tests.log', tests['log_sha256'])
    assert integration['changed_sources'] == {n: config['source_sha256'][n] for n in integration['changed_sources']}
    assert integration['review_sha256'] == digest(checked(REVIEW / 'report.json'))
    assert spec['source_original_sha256'] == review['source_original_sha256']
    assert spec['source_private_sha256'] == review['source_private_sha256']
    assert proof['proofs']['default_off_paired_actual_patches_records'] == 80
    assert proof['proofs']['paired_main_rng_records'] == 1280
    for folder, hashes in [(RUN / 'source', config['source_sha256']),
                           (RUN / 'test-source', tests['tests_sha256']),
                           (V2 / 'source-original', spec['source_original_sha256']),
                           (V2 / 'source-private', spec['source_private_sha256'])]:
        for name, sha in hashes.items():
            checked(folder / name, sha)

    # Validate the existing lossless archives, without another data/point copy.
    data_index_path = EVIDENCE / 'data-manifests/index.json'
    data_index = read(data_index_path)
    expected_data = dict(config['data_sha256'])
    for entry in data_index['files']:
        if entry['originalWorkRelativePath'] in (
            'dsbi-training-contexts-v2/train.json',
            'dsbi-training-contexts-v2/preparation-protocol.json',
            'dsbi-training-contexts-v2/independent-context-audit.json',
            'dsbi-training-contexts-v2/source-cell-coverage.json'):
            expected_data[entry['originalWorkRelativePath']] = entry['sha256']
    linked_data = []
    for name, expected in expected_data.items():
        entries = [x for x in data_index['files'] if x['sha256'] == expected]
        assert len(entries) == 1, name
        entry = entries[0]
        archive = EVIDENCE / 'data-manifests' / entry['file']
        checked(archive, entry['gzipSha256'])
        h = hashlib.sha256()
        size = 0
        with gzip.open(archive, 'rb') as stream:
            while chunk := stream.read(1024 * 1024):
                h.update(chunk)
                size += len(chunk)
        assert h.hexdigest() == expected and size == entry['sizeBytes'], name
        linked_data.append({'configurationKey': name, 'file': rel(archive),
                            'sha256': entry['gzipSha256'], 'encoding': 'gzip',
                            'rawSha256': expected, 'rawSizeBytes': size})

    OUT.mkdir(parents=True)
    copied = {}
    payloads = []

    def retain(path, destination, expected=None, compress=False):
        data = checked(path, expected)
        sha = digest(data)
        if (sha, compress) in copied:
            return dict(copied[(sha, compress)])
        saved = gzip.compress(data, compresslevel=9, mtime=0) if compress else data
        dest = OUT / destination
        dest.parent.mkdir(parents=True, exist_ok=True)
        assert not dest.exists(), dest
        dest.write_bytes(saved)
        assert dest.read_bytes() == saved
        reference = {'file': rel(dest), 'sha256': digest(saved)}
        if compress:
            assert gzip.decompress(saved) == data
            reference.update(encoding='gzip', rawSha256=sha, rawSizeBytes=len(data))
        copied[(sha, compress)] = reference
        payloads.append(dict(reference, sizeBytes=len(saved), originalFile=rel(path)))
        return dict(reference)

    def link(path, expected=None):
        return {'file': rel(path), 'sha256': digest(checked(path, expected))}

    def save(name, obj):
        path = OUT / name
        assert not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')
        reference = link(path)
        payloads.append(dict(reference, sizeBytes=path.stat().st_size))
        return reference

    sources = {name: retain(RUN / 'source' / name, 'run28/source/' + name, sha)
               for name, sha in config['source_sha256'].items()}
    original = {name: retain(V2 / 'source-original' / name, 'prototype-v2/source-original/' + name, sha)
                for name, sha in spec['source_original_sha256'].items()}
    private = {name: retain(V2 / 'source-private' / name, 'prototype-v2/source-private/' + name, sha)
               for name, sha in spec['source_private_sha256'].items()}
    test_sources = {name: retain(RUN / 'test-source' / name, 'run28/test-source/' + name, sha)
                    for name, sha in tests['tests_sha256'].items()}
    run_files = {name: retain(RUN / name, 'run28/' + name) for name in (
        'training-config.json', 'anchor-zoom-protocol.json', 'launch-command.json',
        'dsbi-addition-protocol.json', 'dsbi-visible-partial-protocol.json',
        'preflight-tests.json', 'preflight-tests.log')}
    helpers = {name: retain(WORK / name, 'supporting-source/' + name) for name in (
        'prepare_run28.py', 'launch_run28.py', 'test_run28.py')}
    helpers['retainer'] = retain(__file__, 'supporting-source/retain_anchor_zoom_preflight.py')
    prototype = {name: retain(V2 / name, 'prototype-v2/' + name) for name in (
        'prototype-spec.json', 'proof-report.json', 'proof.py', 'word_anchor_zoom.py',
        'integrate.py', 'integration-record.json', 'test_word_anchor_zoom.py')}
    prototype['sourcesOriginal'] = original
    prototype['sourcesPrivate'] = private
    prototype['pairedResults'] = retain(V2 / 'paired-record-results.json',
        'prototype-v2/paired-record-results.json.gz', proof['paired_results_sha256'], True)
    prototype['cohort'] = retain(WORK / 'train-target-survivorship-audit/cohort.json',
        'prototype-v2/cohort.json', proof['cohort_sha256'])
    audits = {name: retain(REVIEW / name, 'independent-review/' + name)
              for name in ('report.json', 'audit.py')}
    historical = {name: retain(V1 / name, 'historical-v1/' + name) for name in (
        'word_anchor_zoom.py', 'prototype-spec.json', 'proof-report.json', 'proof.py',
        'initial-test-correction.json', 'proof-before-boundary-expectation-fix.py')}
    assert digest(checked(V1 / 'paired-record-results.json')) == proof['paired_results_sha256']
    historical['pairedResults'] = prototype['pairedResults']
    historical['boundaryFailureNote'] = save('historical-v1/rounding-failure.json', {
        'scope': 'Historical prelaunch defect; v1 was not used for run28 training.',
        'v1Helper': historical['word_anchor_zoom.py'],
        'observedCase': review['v1_boundary_defect'],
        'fix': 'V2 verifies actual affine arithmetic even at bound>=sampled zoom; one ULP fixes this case within the original eight-ULP budget.',
        'fixedInSourceSha256': config['source_sha256']['word_anchor_zoom.py'],
        'independentProof': audits['report.json'],
        'separateEarlierTestCorrection': 'The initial test expected a full left-edge cell with -4% translation to be infeasible. Its valid cap is .92. The original test and correction are retained separately; that mistake was not the later float32 rounding defect.'})
    existing_proofs = {name: link(EVIDENCE / name) for name in (
        'data-manifests/index.json', 'dsbi-training-preparation/archive.json',
        'dsbi-training-preparation/source-index.json',
        'dsbi-training-preparation/preparation-protocol.json',
        'dsbi-training-preparation/independent-context-audit.json',
        'dsbi-visible-partial-preflight/archive.json',
        'dsbi-visible-partial-preflight/real-dsbi-point-run20-verifier-check.json',
        'dsbi-visible-partial-preflight/real-dsbi-point-verifier-reconstruction.json')}
    parent = {}
    for name, expected in [('run20/training-config.json', protocol['parent_training_config_sha256']),
                           ('selected-v15/selected-validation.json', protocol['parent_selected_validation_sha256'])]:
        parent[name] = retain(WORK / name, 'parent/' + name.replace('/', '__'), expected)
    report = {
        'schemaVersion': 1, 'createdAtUtc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Immutable run28 source and prelaunch target-proof retention. Not model accuracy, observed training completion, checkpoint eligibility, native-camera evidence or release approval.',
        'run': 'run28', 'configuration': run_files['training-config.json'],
        'protocol': run_files['anchor-zoom-protocol.json'], 'runFiles': run_files,
        'sourceSnapshots': sources, 'testSourceSnapshots': test_sources,
        'supportingSources': helpers, 'prototype': prototype, 'independentReview': audits,
        'historicalV1': historical, 'parentProofs': parent,
        'parentCheckpointSha256': config['resume_sha256'],
        'linkedDataArchives': linked_data, 'linkedExistingProofs': existing_proofs,
        'verification': {'sourceSnapshots': 23, 'testSnapshots': 7, 'prelaunchTestsPassed': 71,
            'pairedDiagnosticRecords': 1280, 'defaultOffPairs': 80,
            'independentRationalContainmentCases': 162, 'gzipRawBytesReverified': True,
            'allFrozenSourceReportHashesMatched': True, 'newModelExecution': False,
            'newFullTests': False, 'newDataOrPointArchiveCopy': False},
        'limitations': proof['limitations'] + [
            '23 source files include dormant branch utilities; actual config keeps geometry/BN-affine/fine branches and other geometric augmentations disabled.',
            'Prototype word_train.py differs from the final trainer by reviewed integration plumbing; both byte versions are retained and referenced distinctly.',
            'Absolute original paths are retained verbatim in original reports. These reference maps locate exact archived bytes; equal-hash source copies are deduplicated.',
            'Replaying the original source-dependent point loader also needs hash-matched original DSBI TRAIN annotation files. Existing DSBI archives document those bytes; original photographs remain external.',
            'The original proof scripts refuse to overwrite their outputs. Restore into a fresh scratch layout for a replay; never rerun them over this archive.']}
    report_ref = save('preflight.json', report)
    readme = OUT / 'README.md'
    readme.write_text('# Run28 anchor zoom preflight\n\n'
        'This archive binds the actual run28 configuration, protocol, 23 source files and 71 source-bound prelaunch tests. '
        'The prototype and independent proof exercise geometry and training targets with synthetic RGB and 40 fixed TRAIN metadata records over 32 diagnostic phases. They do not measure recognition accuracy or actual training-epoch coverage.\n\n'
        'The nearest originally complete known cell can cap the existing affine zoom. No new RNG or raster is added; visibility stays irreversible and later rotation can still cut a cell. '
        'The v1 float32 boundary defect and separate earlier mistaken test expectation are retained under historical-v1. Only the corrected v2 helper was trained.\n\n'
        'preflight.json maps every source/report role to exact repository-relative bytes. Identical source bytes are stored once. '
        'paired-record-results.json.gz expands losslessly to the recorded raw SHA-256. Large immutable training manifests and DSBI point metadata are linked to existing data-manifests archives, not duplicated. '
        'Original reports retain their original paths and claims. No model binaries, source photographs, validation predictions or training-completion claim are included.\n\n'
        'archive-index.json hashes every new payload including preflight.json. The stdlib retainer verifies original bindings, gzip contents and copied bytes; it refuses to overwrite this archive.\n', encoding='utf-8')
    payloads.append(dict(link(readme), sizeBytes=readme.stat().st_size))
    for entry in payloads:
        checked(ROOT / entry['file'], entry['sha256'])
    index = {'scope': 'New immutable payloads only; existing data/proof links are bound in preflight.json.',
             'files': sorted(payloads, key=lambda x: x['file']),
             'payloadCount': len(payloads), 'payloadBytes': sum(x['sizeBytes'] for x in payloads)}
    path = OUT / 'archive-index.json'
    path.write_text(json.dumps(index, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'passed', 'preflight': report_ref, 'archiveIndex': link(path),
                      'payloadCount': index['payloadCount'], 'payloadBytes': index['payloadBytes']}, indent=2))


if __name__ == '__main__':
    main()
