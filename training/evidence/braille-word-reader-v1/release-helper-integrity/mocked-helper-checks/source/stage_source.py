"""Snapshot checked source and copy only that exact source to the build stage."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess

CHECKS = ['model:verify', 'release:config:verify', 'lint', 'typecheck', 'test:coverage', 'export:android']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_path(directory, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Invalid source path: {relative}')
    path = directory / relative
    if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f'Source/stage path escapes its directory: {relative}')
    return path


def snapshot(root, verification_id):
    listed = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                            cwd=root, check=True, capture_output=True).stdout
    paths = sorted(set(name.decode('utf-8') for name in listed.split(b'\0') if name))
    records, deleted = [], []
    for relative in paths:
        source = safe_path(root, relative)
        if not source.is_file():
            deleted.append(relative)
            continue
        before = source.stat()
        digest = sha(source)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'Source changed while hashing: {relative}')
        records.append({'file': relative, 'sha256': digest, 'sizeBytes': after.st_size})
    content = {'files': records, 'deletedPaths': deleted}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    return {'schemaVersion': 1, 'verificationId': verification_id, 'root': str(root.resolve()),
            'createdAtUtc': datetime.now(timezone.utc).isoformat(), 'sourceSetSha256': digest,
            'snapshotHelperSha256': sha(__file__), **content}


def require_same_source(before, after):
    for key in ('schemaVersion', 'verificationId', 'root', 'snapshotHelperSha256', 'sourceSetSha256', 'files', 'deletedPaths'):
        if before.get(key) != after.get(key):
            raise ValueError(f'Checked source changed: {key}; rerun source checks')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verified_source(root, evidence):
    before_path, after_path = evidence/'source-snapshot-before.json', evidence/'source-snapshot-after.json'
    before, after = read_json(before_path), read_json(after_path)
    require_same_source(before, after)
    if before['schemaVersion'] != 1 or before['root'] != str(root.resolve()) or before['snapshotHelperSha256'] != sha(__file__):
        raise ValueError('Snapshot belongs to a different root or snapshot helper')
    checks_path = evidence/'source-checks.json'
    checks = read_json(checks_path)
    if not isinstance(checks, list) or [row['check'] for row in checks] != CHECKS or any(row['exitCode'] != 0 for row in checks):
        raise ValueError('All six source checks must pass')
    if not before.get('verificationId') or any(row.get('verificationId') != before['verificationId'] or
            row.get('sourceSnapshotSha256') != sha(before_path) for row in checks):
        raise ValueError('Source checks are not bound to this verification snapshot')
    if after.get('sourceChecksSha256') != sha(checks_path):
        raise ValueError('Source check results changed after the final snapshot')
    require_same_source(after, snapshot(root, before['verificationId']))
    return after, checks_path


def copy_verified_source(root, stage, checked):
    for relative in checked['deletedPaths']:
        if safe_path(stage, relative).exists():
            raise ValueError(f'Stale stage file needs explicit review: {relative}')
    previous = stage/'evidence/source-copy.json'
    if previous.is_file():
        present = {row['file'] for row in checked['files']}
        for row in read_json(previous)['files']:
            if row['file'] not in present and safe_path(stage, row['file']).exists():
                raise ValueError(f'Stale prior staged source needs explicit review: {row["file"]}')
    for row in checked['files']:
        source, target = safe_path(root, row['file']), safe_path(stage, row['file'])
        if sha(source) != row['sha256']:
            raise ValueError(f'Source changed before copy: {row["file"]}')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(source) != row['sha256'] or sha(target) != row['sha256']:
            raise ValueError(f'Source changed while copying: {row["file"]}')
    require_same_source(checked, snapshot(root, checked['verificationId']))
    for row in checked['files']:
        if sha(safe_path(stage, row['file'])) != row['sha256']:
            raise ValueError(f'Staged file changed during copy: {row["file"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--verification-id')
    parser.add_argument('--checks', type=Path)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if args.snapshot:
        if not args.verification_id:
            parser.error('--snapshot requires --verification-id')
        result = snapshot(root, args.verification_id)
        if args.compare:
            require_same_source(read_json(args.compare), result)
        if args.checks:
            result['sourceChecksSha256'] = sha(args.checks)
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({'snapshot': str(args.snapshot), 'sha256': sha(args.snapshot), 'sourceSetSha256': result['sourceSetSha256']}))
        return
    stage = Path('C:/bm-word-apk-20260905').resolve()
    assert root.name == 'brailleme' and (root/'.git').exists()
    assert stage != root and stage.parent == Path('C:/').resolve()
    checked, checks_path = verified_source(root, root/'output/word-scan-20260905')
    checks_hash = sha(checks_path)
    after_path = checks_path.parent/'source-snapshot-after.json'
    snapshot_hash = sha(after_path)
    lock = root/'package-lock.json'
    assert sha(lock) == sha(stage/'package-lock.json'), 'Reinstall stage dependencies if the lockfile changed'
    manifest = read_json(root/'assets/models/braille-word-reader-v1/manifest.json')
    assert manifest['internalAcceptance']['passed'] is True
    assert sha(root/'assets/models/braille-word-reader-v1'/manifest['artifact']['file']) == manifest['artifact']['sha256']
    copy_verified_source(root, stage, checked)
    if sha(checks_path) != checks_hash or sha(after_path) != snapshot_hash:
        raise ValueError('Verification evidence changed during staging')
    report = {'scope': 'Exact checked source copy; not native build or runtime evidence',
              'createdAtUtc': datetime.now(timezone.utc).isoformat(), 'root': str(root), 'stage': str(stage),
              'verificationId': checked['verificationId'], 'sourceSetSha256': checked['sourceSetSha256'],
              'sourceChecksSha256': checks_hash, 'sourceSnapshotSha256': snapshot_hash,
              'packageLockSha256': sha(lock), 'modelSha256': manifest['artifact']['sha256'], 'files': checked['files']}
    output = stage/'evidence/source-copy.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'copiedFiles': len(checked['files']), 'sourceManifest': str(output), 'sha256': sha(output)}))


if __name__ == '__main__':
    main()
