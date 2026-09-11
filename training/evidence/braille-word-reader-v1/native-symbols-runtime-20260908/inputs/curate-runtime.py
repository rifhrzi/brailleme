"""Preserve compact emulator evidence with hashes; never assess runtime acceptance.

Inputs must be absolute. Output must be new and inside this repository. A failed
copy leaves a partial directory without index.json; select a new destination.
Only tests may exercise this helper before final evidence inputs are selected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import stat

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MAX_FILE_BYTES = 10_000_000
PACKAGE_SUFFIXES = {'.apk', '.apks', '.aab'}


class EvidenceError(ValueError):
    """An input or copy cannot be safely preserved as declared."""


def _no_links(path: Path) -> None:
    # Windows junctions/reparse points also redirect traversal; is_symlink alone
    # does not cover them. Check ancestors before resolving the supplied path.
    for part in reversed((path, *path.parents)):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise EvidenceError(f'Symlink or reparse point is not allowed: {part}')


def _existing(path: Path, boundary: Path | None = None) -> tuple[Path, object]:
    _no_links(path)
    resolved = path.resolve(strict=True)
    if boundary is not None and not resolved.is_relative_to(boundary):
        raise EvidenceError(f'Path escapes its input directory: {path}')
    info = path.stat()
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        raise EvidenceError(f'Only regular files and directories are allowed: {path}')
    return resolved, info


def _identity(info) -> tuple:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _scan(root: Path) -> list[tuple[Path, Path, object]]:
    records = []

    def visit(path: Path, relative: Path):
        actual, info = _existing(path, root)
        records.append((actual, relative, info))
        if stat.S_ISDIR(info.st_mode):
            for child in sorted(path.iterdir(), key=lambda value: value.name.casefold()):
                visit(child, relative / child.name)

    visit(root, Path())
    return records


def curate(inputs: list[Path], output: Path, *, repository_root: Path = REPOSITORY_ROOT) -> dict:
    if not inputs:
        raise EvidenceError('At least one input is required')
    repository_root, _ = _existing(repository_root)
    output = output if output.is_absolute() else repository_root / output
    if '..' in output.parts:
        raise EvidenceError('Output must not contain parent traversal')
    if output.exists() or output.is_symlink():
        raise EvidenceError(f'Destination already exists: {output}')
    existing_parent = next(parent for parent in output.parents if parent.exists() or parent.is_symlink())
    _existing(existing_parent)
    output = output.resolve(strict=False)
    if output == repository_root or not output.is_relative_to(repository_root):
        raise EvidenceError('Output must be a new directory inside the repository')

    planned = []
    names = set()
    for supplied in inputs:
        if not supplied.is_absolute() or '..' in supplied.parts:
            raise EvidenceError(f'Input must be absolute without parent traversal: {supplied}')
        source, info = _existing(supplied)
        name = supplied.name
        if not name or name.casefold() in names:
            raise EvidenceError(f'Conflicting or empty input basename: {name!r}')
        names.add(name.casefold())
        if stat.S_ISDIR(info.st_mode) and output.is_relative_to(source):
            raise EvidenceError('Destination must not be nested inside an input directory')
        records = _scan(source) if stat.S_ISDIR(info.st_mode) else [(source, Path(), info)]
        planned.append((name, source, records))

    # All recursive containment/link checks complete before creating output.
    output.parent.mkdir(parents=True, exist_ok=True)
    _existing(output.parent, repository_root)
    output.mkdir(exist_ok=False)
    (output/'inputs').mkdir()
    copied, skipped, sources = [], [], []
    for name, source, records in planned:
        destination_root = output/'inputs'/name
        sources.append({'basename': name, 'sourceAbsolutePath': str(source),
                        'destination': destination_root.relative_to(output).as_posix()})
        for path, relative, before in records:
            destination = destination_root/relative
            _existing(path, source)
            if stat.S_ISDIR(before.st_mode):
                destination.mkdir(exist_ok=False)
                continue
            if _identity(path.stat()) != _identity(before):
                raise EvidenceError(f'Source changed after preflight: {path}')
            original_hash = _hash(path)
            reason = ('file_exceeds_10_mb' if before.st_size > MAX_FILE_BYTES else
                      'excluded_application_package' if path.suffix.casefold() in PACKAGE_SUFFIXES else '')
            record = {'sourceAbsolutePath': str(path),
                      'destination': destination.relative_to(output).as_posix(),
                      'sizeBytes': before.st_size, 'sourceSha256': original_hash}
            if reason:
                _existing(path, source)
                if _identity(path.stat()) != _identity(before) or _hash(path) != original_hash:
                    raise EvidenceError(f'Skipped source changed while hashing: {path}')
                skipped.append({**record, 'reason': reason, 'copied': False})
                continue
            _existing(destination.parent, output)
            if destination.exists() or destination.is_symlink():
                raise EvidenceError(f'Conflicting destination: {destination}')
            shutil.copyfile(path, destination, follow_symlinks=False)
            _existing(path, source)
            _existing(destination, output)
            copied_hash = _hash(destination)
            if (_identity(path.stat()) != _identity(before) or _hash(path) != original_hash
                    or destination.stat().st_size != before.st_size or copied_hash != original_hash):
                raise EvidenceError(f'Original/copy SHA-256 mismatch or source changed: {path}')
            copied.append({**record, 'copiedSha256': copied_hash, 'copied': True})

    for _, source, records in planned:
        current = _scan(source) if stat.S_ISDIR(records[0][2].st_mode) else [(source, Path(), _existing(source)[1])]
        fingerprint = lambda values: [(str(relative), _identity(info), stat.S_ISDIR(info.st_mode))
                                      for _, relative, info in values]
        if fingerprint(current) != fingerprint(records):
            raise EvidenceError(f'Input tree changed while preserving evidence: {source}')

    report = {'schemaVersion': 1, 'createdAtUtc': datetime.now(timezone.utc).isoformat(),
              'scope': 'emulator_evidence_preservation_only',
              'outputAbsolutePath': str(output), 'inputs': sources,
              'limits': {'maxIndividualFileBytes': MAX_FILE_BYTES,
                         'excludedPackageSuffixes': sorted(PACKAGE_SUFFIXES),
                         'symlinksAndReparsePointsAllowed': False,
                         'acceptanceInferred': False, 'physicalDeviceValidationAsserted': False},
              'limitations': ['Only artifact preservation and hashes are verified; no correctness, performance, or acceptance result is inferred.',
                              'Inputs are emulator evidence and do not establish physical Braille camera accuracy.',
                              'Oversize files and application packages remain at their original paths and are indexed rather than copied.'],
              'copiedFiles': copied, 'skippedFiles': skipped,
              'summary': {'copiedFiles': len(copied), 'skippedFiles': len(skipped),
                          'copiedBytes': sum(row['sizeBytes'] for row in copied),
                          'skippedBytes': sum(row['sizeBytes'] for row in skipped)}}
    _existing(output, repository_root)
    with (output/'index.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, action='append', required=True, dest='inputs',
                        help='Absolute regular file or directory; repeat for separate inputs')
    parser.add_argument('--output', type=Path, required=True,
                        help='New repository directory, absolute or relative to repository root')
    args = parser.parse_args()
    try:
        report = curate(args.inputs, args.output)
    except (EvidenceError, OSError) as error:
        parser.exit(1, f'Cannot curate evidence: {error}\n')
    print(json.dumps({'output': report['outputAbsolutePath'], **report['summary'],
                      'acceptanceInferred': False}), flush=True)


if __name__ == '__main__':
    main()
