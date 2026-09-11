"""2026-09-11 overall exact-text protocol and fresh source reservation.

This version is independent of all previously frozen scripts and evidence. It
reserves publisher pages before annotation inspection or inference. Primary
accuracy, accepted precision, and invalid rejection are distinct quantities.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import random

VERSION = 'overall-exact-text-v5-20260911'
DSBI_REVISION = '6937eb8973efe5ff337e5c018485cb626303f5c2'
GROUPS = ('lowercase', 'capital_single', 'capital_word', 'numbers', 'punctuation')
PRIMARY_POSITIVES = 500
PRIMARY_PER_GROUP = 100
SEED = 20260911


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def verify_hashes(bindings):
    for path, digest in bindings.items():
        if not Path(path).is_file() or sha(path) != digest:
            raise ValueError('Frozen protocol input changed: '+str(path))


def split_members(path):
    members = []
    for raw in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not raw.strip():
            continue
        member = PurePosixPath(raw.strip().replace('\\', '/'))
        if member.is_absolute() or '..' in member.parts or member.suffix != '.jpg':
            raise ValueError('Unsafe DSBI publisher member')
        members.append(member.as_posix())
    if len(set(members)) != len(members):
        raise ValueError('Duplicate DSBI publisher member')
    return members


def reserve_pages(publisher_train, publisher_test, observed_test, count=24, seed=SEED):
    """Labels and predictions are deliberately absent from this interface."""
    if (len(set(publisher_train)) != len(publisher_train) or len(set(publisher_test)) != len(publisher_test)
            or len(set(observed_test)) != len(observed_test)):
        raise ValueError('Duplicate source identity in split inputs')
    train, test, observed = set(publisher_train), set(publisher_test), set(observed_test)
    if train & test or not observed <= test:
        raise ValueError('Publisher split overlap or unknown observed TEST page')
    eligible = sorted(test-train-observed)
    if not 1 <= count <= len(eligible):
        raise ValueError('Not enough unobserved publisher TEST pages')
    families = defaultdict(list)
    for member in eligible:
        families[str(PurePosixPath(member).parent)].append(member)
    rng = random.Random(seed)
    for values in families.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < count:
        for family in sorted(families):
            if families[family] and len(selected) < count:
                selected.append(families[family].pop())
    return selected, sorted(set(eligible)-set(selected))


def metrics(decisions, *, require_primary=False):
    """Rejected positives remain wrong; invalid rows never inflate accuracy.

    Entries require source/group/expected/predicted strings. The >=80% outcome
    is independent of precision and invalid-rejection flags, so neither a
    worst-group criterion nor a precision-first rank can redefine the target.
    """
    if not decisions or len({row['source'] for row in decisions}) != len(decisions):
        raise ValueError('Metric denominator requires unique nonempty examples')
    for row in decisions:
        if (not isinstance(row.get('expected'), str) or not isinstance(row.get('predicted'), str)
                or row.get('group') not in (*GROUPS, 'rejection')
                or bool(row['expected']) == (row['group'] == 'rejection')):
            raise ValueError('Malformed text decision or semantic group')
    positives = [row for row in decisions if row['expected']]
    negatives = [row for row in decisions if not row['expected']]
    if not positives:
        raise ValueError('Overall exact accuracy requires positive text examples')
    groups = Counter(row['group'] for row in positives)
    multichar = sum(len(row['expected']) >= 2 for row in positives)
    if require_primary and (groups != Counter({group: PRIMARY_PER_GROUP for group in GROUPS})
            or multichar != 480 or len(negatives) != 40):
        raise ValueError('Primary evaluation must preserve 500 balanced positives, 96% multicharacter, and 40 invalid rows')
    correct = sum(row['predicted'] == row['expected'] for row in positives)
    accepted = sum(bool(row['predicted']) for row in positives)
    false_accepts = sum(bool(row['predicted']) for row in negatives)
    accuracy = correct/len(positives)
    precision = correct/accepted if accepted else None
    return dict(positive_total=len(positives), exact_correct=correct, overall_exact_accuracy=accuracy,
        target80_met=accuracy >= .8, accepted_positive_total=accepted, accepted_precision=precision,
        precision95_met=precision is not None and precision >= .95,
        invalid_total=len(negatives), invalid_false_accepts=false_accepts,
        invalid_rejection_met=bool(negatives) and false_accepts == 0,
        multi_character_total=multichar,
        groups={group: dict(total=groups[group], correct=sum(row['expected'] == row['predicted'] for row in positives if row['group'] == group))
                for group in GROUPS})


def validation_rank(decisions):
    """Primary optimization follows the user's overall exact-text objective."""
    value = metrics(decisions)
    return (value['overall_exact_accuracy'], value['accepted_precision'] or 0., -value['invalid_false_accepts'])


def reserve(repo, experiment, previous, restored, *, count=24, seed=SEED):
    repo, experiment, previous, restored = map(lambda path: Path(path).resolve(), (repo, experiment, previous, restored))
    if experiment == previous or experiment.name != 'overall80-20260911':
        raise ValueError('Use the new overall80-20260911 evidence directory')
    for name in ('protocol.json', 'test-source-reservation.json', 'test-evaluation-reservation.json'):
        if (experiment/name).exists():
            raise ValueError('New protocol or reservation already exists; never overwrite')
    old_selection = previous/'test-source-selection.json'
    old_downloads = previous/'test-download-manifest.json'
    training_downloads = repo/'training/evidence/braille-word-reader-v1/dsbi-training-preparation/download-manifest.json'
    train_list, test_list = restored/'train.txt', restored/'test.txt'
    pinned = read(training_downloads)
    known = {row['file']: row['sha256'] for row in pinned['sourceFiles']}
    verify_hashes({train_list: known['train.txt'], test_list: known['test.txt']})
    observed = read(old_selection)
    if pinned['revision'] != DSBI_REVISION or observed['revision'] != DSBI_REVISION:
        raise ValueError('Pinned DSBI revisions differ')
    train, test = split_members(train_list), split_members(test_list)
    if len(train) != 26 or len(test) != 88 or len(observed['pages']) != 18:
        raise ValueError('Prior publisher/source split counts changed')
    selected, unallocated = reserve_pages(train, test, observed['pages'], count, seed)
    old = read(old_downloads)
    expected_files = {f'data/{PurePosixPath(page).with_suffix("").as_posix()}+recto.jpg' for page in observed['pages']}
    if {row['file'] for row in old['files'] if row['file'].endswith('.jpg')} != expected_files:
        raise ValueError('Observed page list and old download evidence disagree')
    # Existing evaluated public/legacy images are Angelina sources, not an
    # additional unnamed DSBI subset. Bind and check that assumption explicitly.
    external = previous.parent/'deployment-baseline/external-words.json'
    legacy = previous.parent/'legacy-selection-fixtures.json'
    development_sources = {}
    for path in (external, legacy):
        rows = read(path)['records']
        sources = {row['source'] for row in rows}
        if any('dsbi' in source.lower() or '+recto' in source.lower() for source in sources):
            raise ValueError('Additional evaluated DSBI sources require exclusion before reservation')
        development_sources[str(path)] = dict(records=len(rows), sources=sorted(sources), sha256=sha(path))
    development_files = {name: str(previous/name) for name in ('train-symbols.json', 'train-image-hashes.json',
                         'validation-symbols.json', 'validation-image-hashes.json')}
    inputs = (old_selection, old_downloads, training_downloads, train_list, test_list, external, legacy,
              *map(Path, development_files.values()))
    bindings = {str(path): sha(path) for path in inputs}
    scripts = Path(__file__).parent
    source_bindings = {str(scripts/name): sha(scripts/name) for name in
        ('word_overall_v5_protocol.py', 'word_symbol_v4_prepare.py', 'word_symbols_id_v4.py', 'word_decode_id_v4.py', 'word_dsbi.py')}
    experiment.mkdir(parents=True, exist_ok=True)
    protocol = dict(version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
        user_target='At least 80 percent overall exact text, not 80 percent in every group',
        primary=dict(positive_total=500, per_group=100, groups=GROUPS, multicharacter_total=480,
            invalid_total=40, metric='all characters and symbols exactly equal; rejected positives count wrong',
            target80='exact_correct / 500 >= 0.80', composition='original v4 stress composer and text-length schedule unchanged',
            selection_rank='overall exact accuracy, then accepted precision, then fewer invalid false accepts'),
        separate_quality_flags=dict(accepted_precision_target=.95, invalid_false_accepts_target=0,
            old_public_regression='607 previous public lowercase cases, 405 legacy cases and 44 invalid cases reported separately'),
        supplemental_native_context=dict(role='separate diagnostic; never mixed into primary 500 denominator',
            planned_maximum=200, source='contiguous annotated cells from the same reserved source pages',
            image_policy='identity transform, single native image rectangle, uniform resize and letterbox; no individual cell pasting',
            label_policy='original six-bit sequence; v4 literal token transcription only where grammar-valid, not translation of Chinese source text',
            coverage_policy='report available native groups and exclusions; never synthesize missing capital/number groups'),
        data_scope='source-disjoint recomposed public flatbed scans; not physical-phone-camera validation',
        known_limitations=['Primary recompositions have artificial borders and page/shadow discontinuities',
            'Random token strings differ from real vocabulary frequencies', '18 old TEST pages are development-observed and cannot be final holdout again',
            'Punctuation grammar remains bounded; decimals, signed numbers, spaces and quotes are outside scope'],
        final_holdout_policy='Reserve source pages before labels/inference; freeze method/checkpoint using development validation only; one final evaluation',
        development_files=development_files,
        development_policy='Reuse existing TRAIN and validation by absolute path and SHA256, without copying images/manifests; old observed TEST remains historical only and is not added to gradients',
        input_sha256=bindings, source_sha256=source_bindings, native_runtime_validated=False, physical_camera_validated=False)
    write_new(experiment/'protocol.json', protocol)
    reservation = dict(version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(), revision=DSBI_REVISION,
        seed=seed, policy='seeded family round-robin without annotation or prediction inspection', pages=selected,
        family_counts=dict(Counter(str(PurePosixPath(page).parent) for page in selected)),
        excluded_publisher_train=sorted(train), excluded_development_observed_test=sorted(observed['pages']),
        remaining_unallocated=unallocated, publisher_train_count=26, publisher_test_count=88,
        eligible_before_reservation=70, reserved_count=len(selected), source_overlap=[],
        old_observed_image_sha256=[row['sha256'] for row in old['files'] if row['file'].endswith('.jpg')],
        trained_image_sha256=[row['sha256'] for row in pinned['sourceFiles'] if row['file'].endswith('.jpg')],
        other_evaluated_source_audit=development_sources, input_sha256=bindings,
        protocol_sha256=sha(experiment/'protocol.json'), download_performed=False,
        annotation_inspection_performed=False, model_inference_performed=False)
    write_new(experiment/'test-source-reservation.json', reservation)
    print(json.dumps(dict(reserved=len(selected), unallocated=len(unallocated), families=reservation['family_counts'],
        protocol=str(experiment/'protocol.json'), reservation=str(experiment/'test-source-reservation.json'))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--experiment', type=Path, default=Path('C:/bm-symbols-20260908/overall80-20260911'))
    parser.add_argument('--previous', type=Path, default=Path('C:/bm-symbols-20260908/accuracy80-20260910'))
    parser.add_argument('--restored', type=Path, default=Path('C:/bm-datasets/DSBI-train-6937eb8'))
    args = parser.parse_args()
    reserve(args.repo, args.experiment, args.previous, args.restored)


if __name__ == '__main__':
    main()
