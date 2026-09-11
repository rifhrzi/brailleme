"""Offline summary of recorded emulator evidence; never decides release readiness."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

COUNTERS = ('received', 'admitted', 'busyDrops', 'throttled', 'disabledDrops',
            'runnerDrops', 'completed', 'errors')
BOUNDS = [50, 100, 200, 400, 800, 1600, None]
TRACKING = 'BRAILLE_WORD_TRACKING_METRICS'
PHOTO = 'BRAILLE_WORD_PHOTO_METRICS'


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def integer(value):
    return number(value) and int(value) == value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_json(path, issues):
    try:
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(value, dict):
            raise ValueError('expected object')
        return value
    except (OSError, ValueError) as error:
        issues.append(f'{path}: {error}')
        return {}


def load_lines(path, issues):
    rows = []
    try:
        with path.open(encoding='utf-8-sig') as stream:
            for index, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('expected object')
                    rows.append(row)
                except ValueError as error:
                    issues.append(f'{path.name}:{index}: {error}')
    except OSError as error:
        issues.append(f'{path}: {error}')
    return rows


def inventory(root, issues):
    files = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            issues.append(f'Input link/outside path skipped: {path}')
        elif path.is_file():
            files.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    return files


def nearest_rank(values, percentile=.95):
    return sorted(values)[math.ceil(percentile * len(values)) - 1] if values else None


def histogram_percentile(buckets):
    total = sum(buckets)
    if not total:
        return None
    rank, cumulative = math.ceil(.95 * total), 0
    for index, count in enumerate(buckets):
        cumulative += count
        if cumulative >= rank:
            return {'rank': rank, 'sample_count': total,
                    'lower_ms': BOUNDS[index - 1] if index else 0,
                    'lower_inclusive': index == 0, 'upper_ms': BOUNDS[index],
                    'upper_inclusive': BOUNDS[index] is not None}
    raise AssertionError('Unreachable histogram rank')


def metric_rows(rows, issues):
    tracking, photos, seen, duplicates = [], [], set(), 0
    for index, row in enumerate(rows, 1):
        tag = row.get('tag')
        if tag not in (TRACKING, PHOTO):
            if 'memory' not in row and 'surface' not in row:
                issues.append(f'samples row {index}: unknown record kind')
            continue
        raw = row.get('metrics')
        key = json.dumps([tag, row.get('deviceEpochSeconds'), raw], sort_keys=True)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        valid = isinstance(raw, dict) and integer(raw.get('schemaVersion')) and raw['schemaVersion'] == 1 and number(row.get('deviceEpochSeconds'))
        valid = valid and number(raw.get('elapsedMs'))
        if valid and tag == TRACKING:
            buckets = raw.get('latencyBuckets')
            valid = all(integer(raw.get(field)) for field in COUNTERS)
            valid = valid and isinstance(buckets, list) and len(buckets) == 7 and all(integer(item) for item in buckets)
            valid = valid and raw.get('latencyUpperBoundsMs') == BOUNDS and sum(buckets) == raw['completed']
            valid = valid and raw['errors'] <= raw['completed'] and number(raw.get('meanMs')) and number(raw.get('maxMs'))
        elif valid:
            valid = integer(raw.get('captureId')) and raw.get('status') in ('success', 'rejected', 'error')
        if not valid:
            issues.append(f'samples row {index}: invalid {tag} envelope/schema/counts')
            continue
        (tracking if tag == TRACKING else photos).append(row)
    return tracking, photos, duplicates


def tracking_summary(rows, issues):
    segments, current, reporting_clock_restarts = [], [], 0
    for row in rows:
        if current:
            previous, now = current[-1]['metrics'], row['metrics']
            counter_reset = any(now[key] < previous[key] for key in COUNTERS) or any(
                after < before for before, after in zip(previous['latencyBuckets'], now['latencyBuckets']))
            if counter_reset:
                segments.append(current)
                current = []
            elif now['elapsedMs'] < previous['elapsedMs']:
                # The reporting effect restarts on active false/true without resetting hook counters.
                reporting_clock_restarts += 1
        current.append(row)
    if current:
        segments.append(current)
    result, total, histogram = [], dict.fromkeys(COUNTERS, 0), [0] * 7
    for index, segment in enumerate(segments):
        first, last = segment[0], segment[-1]
        delta = {key: last['metrics'][key] - first['metrics'][key] for key in COUNTERS}
        buckets = [end - start for start, end in zip(first['metrics']['latencyBuckets'], last['metrics']['latencyBuckets'])]
        span = last['deviceEpochSeconds'] - first['deviceEpochSeconds']
        increasing = all(after['deviceEpochSeconds'] > before['deviceEpochSeconds'] for before, after in zip(segment, segment[1:]))
        for key in COUNTERS:
            total[key] += delta[key]
        histogram = [before + after for before, after in zip(histogram, buckets)]
        result.append({'index': index, 'reports': len(segment), 'first_device_epoch_seconds': first['deviceEpochSeconds'],
                       'last_device_epoch_seconds': last['deviceEpochSeconds'], 'device_span_seconds': span,
                       'first_cumulative': {key: first['metrics'][key] for key in COUNTERS},
                       'last_cumulative': {key: last['metrics'][key] for key in COUNTERS},
                       'first_cumulative_buckets': first['metrics']['latencyBuckets'],
                       'last_cumulative_buckets': last['metrics']['latencyBuckets'],
                       'observed_delta_counts_lower_bound': delta, 'observed_delta_buckets_lower_bound': buckets,
                       'callback_rate_lower_bound_hz': delta['received'] / span if span > 0 and increasing else None})
    span = rows[-1]['deviceEpochSeconds'] - rows[0]['deviceEpochSeconds'] if len(rows) > 1 else 0
    increasing = all(after['deviceEpochSeconds'] > before['deviceEpochSeconds'] for before, after in zip(rows, rows[1:]))
    if rows and not increasing:
        issues.append('Tracking device epoch timestamps are not strictly increasing; aggregate rate unavailable.')
    return {'reports': len(rows), 'inferred_counter_resets': max(0, len(segments) - 1),
            'reporting_elapsed_clock_restarts_without_counter_reset': reporting_clock_restarts,
            'segments': result, 'observed_delta_counts_lower_bound': total,
            'observed_delta_buckets_lower_bound': histogram, 'latency_upper_bounds_ms': BOUNDS,
            'observed_delta_histogram_p95_bounds': histogram_percentile(histogram),
            'device_envelope_span_seconds': span,
            'callback_rate_lower_bound_hz': total['received'] / span if span > 0 and increasing else None,
            'method': 'Subtract first snapshot from last within each inferred counter segment. Sum deltas only; never sum cumulative snapshots.',
            'limitations': ['First prefixes and unreported tails are omitted. Unobservable resets may undercount activity.',
                            'Resets are inferred from counter/bucket decreases; elapsedMs alone is not a counter reset.',
                            'Rate uses device epoch envelopes, not the restartable reporting clock or camera sensor frames.',
                            'Histogram percentile bounds describe retained delta events only, not exact or whole-session latency.']}


def summarize(rows, metadata, summary, issues):
    tracking, photos, duplicates = metric_rows(rows, issues)
    elapsed = summary.get('elapsedSeconds')
    valid_elapsed = number(elapsed) and elapsed > 0
    photo_times = [row['metrics']['elapsedMs'] for row in photos]
    memory, surface = [], []
    for index, row in enumerate(rows, 1):
        if 'memory' in row:
            raw = row['memory']
            if isinstance(raw, dict) and number(raw.get('elapsedSeconds')) and integer(raw.get('pssKiB')):
                memory.append(raw)
            else:
                issues.append(f'samples row {index}: memory PSS unavailable/invalid')
        elif 'surface' in row:
            raw = row['surface']
            if isinstance(raw, dict) and number(row.get('elapsedSeconds')) and all(integer(raw.get(key)) for key in ('newPresentedFrames', 'presentedFramesObserved', 'refreshPeriodNs')):
                surface.append(row)
            else:
                issues.append(f'samples row {index}: invalid surface sample')
    memory.sort(key=lambda point: point['elapsedSeconds'])
    frames = sum(row['surface']['newPresentedFrames'] for row in surface)
    for key, count in [('trackingReports', len(tracking)), ('photoReports', len(photos)), ('memorySamples', len(memory)), ('surfaceSamples', len(surface))]:
        if summary.get(key) != count:
            issues.append(f'Collector summary {key}={summary.get(key)!r} differs from valid retained records {count}.')
        if count == 0:
            issues.append(f'No valid {key}; corresponding outcome/metric is unavailable, not an inferred app failure.')
    if photo_times != summary.get('photoLatenciesMs'):
        issues.append('Collector photoLatenciesMs differs from all retained photo statuses.')
    if surface and (frames != surface[-1]['surface']['presentedFramesObserved'] or frames != summary.get('presentedFramesObserved')):
        issues.append('Surface cumulative count differs from retained per-sample increments.')
    if summary.get('collectionStatus') != 'duration_completed':
        issues.append(f"Collector incomplete: {summary.get('collectionStatus')!r}")
    requested = metadata.get('requestedSeconds')
    if not valid_elapsed or not number(requested) or elapsed + 1 < requested:
        issues.append('Requested collection duration is missing/invalid/not reached.')
    if summary.get('errors'):
        issues.append('Collector reported errors: ' + json.dumps(summary['errors']))
    return {'collection': {'status': summary.get('collectionStatus'), 'requested_seconds': requested,
                           'elapsed_seconds': elapsed, 'pid': metadata.get('pid'), 'duplicate_metric_records_ignored': duplicates},
            'photos': {'observed_terminal_records': len(photos), 'nearest_rank_p95_all_statuses_ms': nearest_rank(photo_times),
                       'p95_rank': math.ceil(.95 * len(photos)) if photos else None,
                       'max_all_statuses_ms': max(photo_times) if photos else None,
                       'statuses': {status: {'count': len(times := [row['metrics']['elapsedMs'] for row in photos if row['metrics']['status'] == status]),
                                             'nearest_rank_p95_ms': nearest_rank(times), 'max_ms': max(times) if times else None}
                                    for status in ('success', 'rejected', 'error')},
                       'records': [{'device_epoch_seconds': row['deviceEpochSeconds'], **row['metrics']} for row in photos],
                       'scope': 'Exact nearest-rank over every valid observed terminal record, including rejected/error. No latency for missing terminal logs is invented.'},
            'tracking': tracking_summary(tracking, issues),
            'memory': {'valid_pss_samples': len(memory), 'first_pss_kib': memory[0]['pssKiB'] if memory else None,
                       'last_pss_kib': memory[-1]['pssKiB'] if memory else None,
                       'max_pss_kib': max(point['pssKiB'] for point in memory) if memory else None,
                       'growth_pss_kib': memory[-1]['pssKiB'] - memory[0]['pssKiB'] if memory else None,
                       'scope': 'Sampled process PSS; first/last change alone cannot establish or exclude a memory leak.'},
            'surface': {'metric_kind': 'surface_composition_proxy', 'samples': len(surface),
                        'presented_frames_observed': frames if surface else None,
                        'observed_composition_fps_proxy': frames / elapsed if surface and valid_elapsed else None,
                        'surface_history_gaps': summary.get('surfaceHistoryGaps'), 'surface_layer': metadata.get('surfaceLayer'),
                        'scope': 'Observed surface composition frames per collector elapsed second; finite history may omit frames. Not sensor FPS or camera throughput.'}}


def action_summary(root, issues):
    metadata = load_json(root / 'metadata.json', issues)
    summary = load_json(root / 'summary.json', issues)
    logs = load_lines(root / 'actions.jsonl', issues)
    if summary.get('status') != 'completed_actions_not_assessed':
        issues.append(f'Actions incomplete in {root}: {summary.get("status")!r}; {summary.get("error")!r}')
    observations = summary.get('observations', [])
    if not isinstance(observations, list):
        issues.append(f'Actions observations invalid in {root}')
        observations = []
    missing = []
    for row in observations:
        for key in ('xml', 'screenshot', 'logcat'):
            raw = row.get(key) if isinstance(row, dict) else None
            path = Path(raw).resolve() if isinstance(raw, str) else None
            if path is None or not path.is_relative_to(root) or not path.is_file():
                missing.append({'label': row.get('label') if isinstance(row, dict) else None, 'kind': key, 'path': raw})
    if missing:
        issues.append(f'{root}: {len(missing)} missing/outside observation artifacts (may include expected no-PID cold-start snapshots).')
    return {'directory': str(root), 'command': summary.get('command'), 'status': summary.get('status'),
            'attempted_action_counts': summary.get('action_counts'), 'observation_count': len(observations),
            'log_records': len(logs), 'missing_observation_artifacts': missing,
            'declared_feed': metadata.get('declared_feed'), 'input_files': inventory(root, issues),
            'scope': 'Recorded actions/observations only. A Pindai tap does not prove capture, decoding success or a matching photo metric.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--actions', type=Path, action='append', required=True)
    parser.add_argument('--apk', type=Path, required=True)
    parser.add_argument('--apk-sha256', required=True)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    runtime, actions, output, repo, apk = args.runtime.resolve(), [path.resolve() for path in args.actions], args.output.resolve(), args.repo.resolve(), args.apk.resolve()
    if output.exists() or any(output.is_relative_to(path) for path in [runtime, *actions]):
        parser.error('Use a new report outside input evidence directories; existing evidence is never overwritten.')
    if not runtime.is_dir() or not all(path.is_dir() for path in actions) or not apk.is_file():
        parser.error('Explicit runtime/actions directories and actual APK must exist.')
    if not re.fullmatch(r'[0-9a-fA-F]{64}', args.apk_sha256) or sha(apk) != args.apk_sha256.lower():
        parser.error('Actual APK SHA-256 differs from supplied verified identity.')
    issues = []
    metadata, summary = load_json(runtime / 'metadata.json', issues), load_json(runtime / 'summary.json', issues)
    rows = load_lines(runtime / 'samples.jsonl', issues)
    report = summarize(rows, metadata, summary, issues)
    sources = [repo / 'scripts/word-android-smoke/collect_runtime.py', repo / 'scripts/word-android-smoke/smoke.py',
               repo / 'src/features/live/use-word-camera-tracking.ts', repo / 'src/features/live/word-photo-camera.tsx',
               Path(__file__).resolve().with_name('runtime-actions.py'), Path(__file__).resolve()]
    source_hashes = {}
    for source in sources:
        if source.is_file():
            source_hashes[str(source)] = sha(source)
        else:
            issues.append(f'Missing analysis source: {source}')
    for name in ('media-camera-start.txt', 'media-camera-end.txt'):
        if not (runtime / name).is_file():
            issues.append(f'Missing collector camera diagnostic: {name}')
    report.update({'schema_version': 1, 'generated_utc': datetime.now(timezone.utc).isoformat(),
                   'apk': {'path': str(apk), 'bytes': apk.stat().st_size, 'sha256': args.apk_sha256.lower(),
                           'installed_apk_verified_by_this_analyzer': False},
                   'source_sha256_at_analysis': source_hashes, 'runtime_directory': str(runtime),
                   'runtime_input_files': inventory(runtime, issues), 'actions': [action_summary(path, issues) for path in actions],
                   'acceptanceResult': 'not_assessed', 'physical_camera_validated': False,
                   'release_readiness': 'not_assessed', 'limitations': [
                       'Offline input/APK identity binding does not prove that this APK or current source was installed/running; pair with build/install evidence.',
                       'Collection completeness and observed emulator results are separate from model accuracy, physical-camera readiness and release acceptance.',
                       'Counters contain no session identity; segment resets are inferred. Missing logs and action-to-photo correspondence remain unproven.']})
    report['issues'] = issues
    report['analysis_status'] = 'partial_input_analysis' if issues else 'complete_input_analysis'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(output), 'sha256': sha(output), 'analysis_status': report['analysis_status'],
                      'issues': len(issues), 'acceptanceResult': 'not_assessed'}))
    return 1 if issues else 0


if __name__ == '__main__':
    raise SystemExit(main())
