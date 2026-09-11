from pathlib import Path
import datetime
import hashlib
import json
import re
import statistics

STAGE = Path('C:/bm-symbols-apk-20260908/evidence')
OUT = Path(__file__).resolve().parent
sha = lambda data: hashlib.sha256(data).hexdigest()


def source(path):
    raw = path.read_bytes()
    return {'path': str(path.resolve()), 'sha256': sha(raw), 'bytes': len(raw)}, raw


sources = {}
streams = {}
for label, path in [
    ('start', STAGE / 'runtime-600-v6-complete/media-camera-start.txt'),
    ('end', STAGE / 'timing-600-v6/media-camera-initial-preview-end.txt'),
]:
    sources[label], raw = source(path)
    text = raw.decode('utf-8-sig')
    matches = re.finditer(r'(?ms)^\s*Stream\[0\]: Output\r?\n(.*?)(?=^\s*Stream\[\d+\]:|\Z)', text)
    matching = []
    for match in matches:
        block = match.group(0)
        consumer = re.search(r'Consumer name: (.+)', block)[1].strip()
        if consumer != 'SurfaceTexture-0-9064-1':
            continue
        dims = re.search(r'Dims: (\d+) x (\d+), format (\S+),', block)
        counts = re.search(r'Frames produced: (\d+), last timestamp: (\d+) ns', block)
        matching.append({'streamIndex': 0, 'consumer': consumer, 'pid': 9064,
                         'width': int(dims[1]), 'height': int(dims[2]), 'pixelFormat': dims[3],
                         'framesProduced': int(counts[1]), 'lastTimestampNs': int(counts[2])})
    assert len(matching) == 1, (label, matching)
    streams[label] = matching[0]

for label, name in [('metadata', 'runtime-600-v6-complete/metadata.json'), ('timing', 'timing-600-v6/timing.json')]:
    sources[label], raw = source(STAGE / name)
    if label == 'metadata':
        metadata = json.loads(raw.decode('utf-8-sig'))
    else:
        timing = json.loads(raw.decode('utf-8-sig'))
assert metadata['pid'] == 9064
for field in ['streamIndex', 'consumer', 'pid', 'width', 'height', 'pixelFormat']:
    assert streams['start'][field] == streams['end'][field], field
assert (streams['start']['width'], streams['start']['height']) == (1280, 960)
frames = streams['end']['framesProduced'] - streams['start']['framesProduced']
ns = streams['end']['lastTimestampNs'] - streams['start']['lastTimestampNs']
assert frames > 0 and ns > 0

# Snapshot only a bounded prefix of complete lines, even while collector appends.
samples_path = STAGE / 'runtime-600-v6-complete/samples.jsonl'
raw = samples_path.read_bytes()
bounded = []
parsed = []
for line in raw.splitlines(keepends=True):
    if not line.endswith(b'\n'):
        break
    row = json.loads(line.decode('utf-8-sig'))
    elapsed = row.get('elapsedSeconds', row.get('memory', {}).get('elapsedSeconds'))
    assert elapsed is not None, row.keys()
    if elapsed > 110:
        break
    bounded.append(line)
    parsed.append(row)
snapshot_raw = b''.join(bounded)
snapshot = OUT / 'initial-preview-samples-v6.jsonl'
snapshot.write_bytes(snapshot_raw)
sources['samplesSnapshot'] = {
    'path': str(snapshot), 'sha256': sha(snapshot_raw), 'bytes': len(snapshot_raw),
    'sourcePath': str(samples_path),
    'selection': 'Complete original JSONL lines from start through elapsedSeconds <=110; immutable bounded prefix of a live append-only collection.',
    'sourceReadBytes': len(raw), 'sourcePrefixAtReadMatchesSnapshot': raw.startswith(snapshot_raw),
    'sampleRecords': len(parsed),
}
surface = [r for r in parsed if 'surface' in r and 5 <= r['elapsedSeconds'] <= 110]
assert len(surface) > 1
first, last = surface[0], surface[-1]
assert all(b['elapsedSeconds'] > a['elapsedSeconds'] and b['surface']['presentedFramesObserved'] >= a['surface']['presentedFramesObserved'] for a, b in zip(surface, surface[1:]))
count = last['surface']['presentedFramesObserved'] - first['surface']['presentedFramesObserved']
increments = sum(r['surface']['newPresentedFrames'] for r in surface[1:])
assert count == increments
elapsed = last['elapsedSeconds'] - first['elapsedSeconds']
gaps = [b['elapsedSeconds'] - a['elapsedSeconds'] for a, b in zip(surface, surface[1:])]
parseiso = lambda s: datetime.datetime.fromisoformat(s.replace('Z', '+00:00'))
sample_seconds = (parseiso(timing['cameraSampleStartedUtc']) - parseiso(timing['collectorStartedUtc'])).total_seconds()
sources['collector'], _ = source(Path('scripts/word-android-smoke/collect_runtime.py'))
sources['analysisScript'], _ = source(Path(__file__))
report = {
    'schemaVersion': 1,
    'createdUtc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'analysisKind': 'initial_preview_native_buffer_and_surface_composition_rates',
    'deviceKind': 'Android emulator; not a physical camera test',
    'package': metadata['package'], 'pid': metadata['pid'], 'sources': sources,
    'phase': {
        'collectorStartedUtc': timing['collectorStartedUtc'],
        'cameraEndSampleStartedUtc': timing['cameraSampleStartedUtc'],
        'cameraEndSampleSecondsFromCollectorMetadata': sample_seconds,
        'actionStartedUtc': timing['actionStartedUtc'],
        'actionStartSecondsFromCollector': timing['actionStartSecondsFromCollector'],
        'cameraSnapshotPrecedesActions': sample_seconds < timing['actionStartSecondsFromCollector'],
        'note': 'Collector elapsedSeconds uses its own monotonic loop clock initialized after startup dumps; UTC metadata is recorded earlier. Envelopes below use their respective actual clocks.',
    },
    'nativeCameraBufferProduction': {
        'metricKind': 'camera_service_output_stream_buffers_produced',
        'start': streams['start'], 'end': streams['end'],
        'framesProducedDelta': frames, 'lastTimestampDeltaNs': ns,
        'timestampEnvelopeSeconds': ns / 1e9, 'buffersPerSecond': frames / (ns / 1e9),
        'formula': '(end.framesProduced - start.framesProduced) / ((end.lastTimestampNs - start.lastTimestampNs) / 1e9)',
        'sameStreamIdentity': True,
        'interpretation': 'Rate of output buffers reported by the native camera service for the same SurfaceTexture stream, over its device-timestamp envelope. This is not a count of uniquely displayed preview images or a measurement of a physical image sensor.',
    },
    'surfaceCompositionProxy': {
        'metricKind': metadata['surfaceMetricKind'], 'surfaceLayer': metadata['surfaceLayer'],
        'requestedElapsedRangeSeconds': [5, 110], 'surfaceSamples': len(surface),
        'first': first, 'last': last, 'observedEnvelopeSeconds': elapsed,
        'deduplicatedPresentedFramesDelta': count, 'sumNewPresentedFramesAfterFirstSample': increments,
        'framesPerSecondObserved': count / elapsed,
        'maximumSamplingGapSeconds': max(gaps), 'medianSamplingGapSeconds': statistics.median(gaps),
        'refreshPeriodNsValues': sorted({r['surface']['refreshPeriodNs'] for r in surface}),
        'monotonicCountersVerified': True,
        'formula': '(last.presentedFramesObserved - first.presentedFramesObserved) / (last.elapsedSeconds - first.elapsedSeconds)',
        'boundaryRule': 'Exclude first selected sample increment because it covers time before the first selected observation; include each later increment once.',
        'historyCompleteness': 'SurfaceFlinger exposes bounded recent timestamp histories; collector deduplicates actual-present timestamps across samples. Raw histories and per-sample history-gap flags are not stored in samples.jsonl, so completeness cannot be proved from this bounded snapshot alone. Missing history would undercount.',
        'interpretation': 'App activity surface composition proxy; UI/overlay redraws can be presented between new camera buffers. A higher surface proxy rate than native buffer production does not indicate a physical camera above 30 FPS.',
    },
    'acceptanceResult': 'not_assessed',
    'limitations': [
        'Initial stationary preview interval only; does not cover the full 600-second scenario sequence, capture latency, memory, or thermal behavior.',
        'Native camera buffers and app surface presentation use different clocks and envelopes; they are reported separately and must not be combined into one FPS claim.',
        'No physical sensor throughput or physical-device performance claim.',
        'This analysis performs local file reads only; no ADB or application changes.',
    ],
}
path = OUT.parent / 'initial-preview-analysis-v6.json'
path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'report': str(path), 'sha256': sha(path.read_bytes()),
                  'nativeBuffersPerSecond': report['nativeCameraBufferProduction']['buffersPerSecond'],
                  'nativeFramesDelta': frames, 'nativeSeconds': ns / 1e9,
                  'surfaceFramesPerSecond': count / elapsed, 'surfaceFramesDelta': count,
                  'surfaceEnvelope': [first['elapsedSeconds'], last['elapsedSeconds']],
                  'surfaceSamples': len(surface), 'cameraSampleSeconds': sample_seconds}, indent=2))
