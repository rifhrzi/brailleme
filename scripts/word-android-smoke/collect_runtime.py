"""Read-only, explicitly targeted emulator runtime collector. Never decides release readiness."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import time

from smoke import default_adb, SmokeError


TRACKING_TAG = 'BRAILLE_WORD_TRACKING_METRICS'
PHOTO_TAG = 'BRAILLE_WORD_PHOTO_METRICS'
TRACKING_FIELDS = ('schemaVersion', 'elapsedMs', 'received', 'admitted', 'busyDrops',
                   'throttled', 'disabledDrops', 'runnerDrops', 'completed', 'errors',
                   'meanMs', 'maxMs')
PHOTO_FIELDS = ('schemaVersion', 'captureId', 'elapsedMs')


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def parse_metrics(line):
    """Allowlist aggregates. Recognized text, coordinates, unrelated logs never reach artifacts."""
    # RN's native console formats multiple string arguments as 'TAG', '{...}'.
    # Also accept the unquoted form used by single-string/native log emitters.
    match = re.search(r'''(BRAILLE_WORD_(?:TRACKING|PHOTO)_METRICS)["']?\s*[, :]?\s*["']?(\{.*\})''', line)
    if not match:
        return None
    try:
        raw = json.loads(match[2])
    except (ValueError, TypeError):
        return None
    fields = TRACKING_FIELDS if match[1] == TRACKING_TAG else PHOTO_FIELDS
    if not isinstance(raw, dict) or any(not number(raw.get(field)) for field in fields):
        return None
    value = {field: raw[field] for field in fields}
    if match[1] == TRACKING_TAG:
        buckets = raw.get('latencyBuckets')
        if not isinstance(buckets, list) or len(buckets) != 7 or not all(number(item) for item in buckets):
            return None
        value['latencyBuckets'] = buckets
        value['latencyUpperBoundsMs'] = [50, 100, 200, 400, 800, 1600, None]
    else:
        if raw.get('status') not in ('success', 'rejected', 'error'):
            return None
        value['status'] = raw['status']
    epoch = re.match(r'\s*(\d+\.\d+)', line)
    return {'tag': match[1], 'deviceEpochSeconds': float(epoch[1]) if epoch else None, 'metrics': value}


def parse_memory(output):
    pss = re.search(r'TOTAL PSS:\s*(\d+)', output)
    rss = re.search(r'TOTAL RSS:\s*(\d+)', output)
    if not pss:
        pss = re.search(r'^\s*TOTAL\s+(\d+)\s+', output, re.MULTILINE)
    return {'pssKiB': int(pss[1]) if pss else None, 'rssKiB': int(rss[1]) if rss else None}


def parse_surface_latency(output):
    lines = output.strip().splitlines()
    if not lines or not re.fullmatch(r'\d+', lines[0].strip()):
        return None
    period = int(lines[0])
    timestamps = set()
    for line in lines[1:]:
        values = line.split()
        if len(values) != 3 or not all(value.isdecimal() for value in values):
            continue
        # SurfaceFlinger columns: desired presentation, actual presentation, frame ready.
        actual = int(values[1])
        if 0 < actual < 9_223_372_036_854_775_807:
            timestamps.add(actual)
    return {'refreshPeriodNs': period, 'presentedNs': sorted(timestamps)}


def normalize_surface_layer(value):
    """API36 wraps layer names; --latency still expects the unwrapped name."""
    value = value.strip()
    wrapper = re.fullmatch(r'RequestedLayerState\{(.+?#\d+)(?:\s+(?:parentId|z)=[^{}]*)?\}', value)
    return wrapper[1] if wrapper else value


def select_surface_layer(layers, package):
    """Choose one actual app buffer; containers and ambiguous names stay unset."""
    candidates = [layer for layer in layers if package in layer]
    activity_pattern = re.escape(package) + r'/[A-Za-z_.$][A-Za-z0-9_.$]*(?:#\d+)?'
    activities = [layer for layer in candidates if re.fullmatch(activity_pattern, layer)]
    if activities:
        return activities[0] if len(activities) == 1 else None
    surface_views = [layer for layer in candidates if layer.startswith('SurfaceView')]
    return surface_views[0] if len(surface_views) == 1 else None


class RuntimeCollector:
    def __init__(self, adb, serial, package, output):
        if not re.fullmatch(r'emulator-\d{4,5}', serial):
            raise SmokeError('Serial eksplisit emulator-NNNN wajib; perangkat fisik ditolak.')
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+', package):
            raise SmokeError('Package Android tidak valid.')
        self.adb, self.serial, self.package = str(adb), serial, package
        self.output = Path(output).resolve()
        self.deadline = None

    def run(self, *arguments):
        timeout = min(5, self.deadline - time.monotonic()) if self.deadline else 5
        if timeout <= 0:
            raise SmokeError('Batas waktu koleksi tercapai.')
        try:
            result = subprocess.run([self.adb, '-s', self.serial, *arguments], capture_output=True,
                                    text=True, encoding='utf-8', errors='replace', timeout=timeout,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except (OSError, subprocess.TimeoutExpired) as error:
            raise SmokeError(str(error)) from error
        if result.returncode:
            raise SmokeError(result.stderr.strip() or f'ADB exit {result.returncode}')
        return result.stdout

    def guard(self):
        if self.run('get-state').strip() != 'device' or self.run('shell', 'getprop', 'ro.kernel.qemu').strip() != '1':
            raise SmokeError('Target harus emulator QEMU tersambung; tidak ada tindakan dijalankan.')
        pid = self.run('shell', 'pidof', self.package).strip()
        if not re.fullmatch(r'\d+', pid):
            raise SmokeError('Buka satu proses aplikasi dan scanner sebelum menjalankan collector.')
        return pid

    def write(self, name, data):
        (self.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def sample(self, data):
        with (self.output / 'samples.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(data, ensure_ascii=False) + '\n')

    def collect(self, duration=600, surface_layer=None):
        if not 1 <= duration <= 600:
            raise SmokeError('Durasi harus 1 sampai 600 detik.')
        pid = self.guard()
        if self.output.exists() and any(self.output.iterdir()):
            raise SmokeError('Gunakan folder output baru/kosong agar bukti lama tidak tertimpa.')
        self.output.mkdir(parents=True, exist_ok=True)
        started_utc = datetime.now(timezone.utc).isoformat()
        device_start = float(self.run('shell', 'date', '+%s').strip())
        layers = list(dict.fromkeys(normalize_surface_layer(line) for line in
                      self.run('shell', 'dumpsys', 'SurfaceFlinger', '--list').splitlines() if line.strip()))
        candidates = [line for line in layers if self.package in line]
        if surface_layer:
            surface_layer = normalize_surface_layer(surface_layer)
        if surface_layer and surface_layer not in layers:
            raise SmokeError('Surface layer eksplisit tidak ditemukan pada emulator.')
        if surface_layer is None:
            # Ambiguous surfaces stay unselected; never fabricate a preview frame rate.
            surface_layer = select_surface_layer(layers, self.package)
        self.write('metadata.json', {'schemaVersion': 1, 'serial': self.serial, 'package': self.package,
                   'pid': int(pid), 'startedUtc': started_utc, 'requestedSeconds': duration,
                   'surfaceLayer': surface_layer, 'surfaceCandidates': candidates,
                   'deviceMutation': False, 'surfaceMetricKind': 'surface_composition_proxy'})
        (self.output / 'media-camera-start.txt').write_text(self.run('shell', 'dumpsys', 'media.camera'), encoding='utf-8')
        initial_surface = parse_surface_latency(self.run('shell', 'dumpsys', 'SurfaceFlinger', '--latency', shlex.quote(surface_layer))) if surface_layer else None
        seen_frames = set(initial_surface['presentedNs']) if initial_surface else set()
        previous_latest = max(seen_frames) if seen_frames else None
        observed_frames = 0
        surface_gaps = 0
        surface_samples = 0
        memory = []
        tracking = []
        photos = []
        seen_metrics = set()
        errors = []
        reason = 'duration_completed'
        started = time.monotonic()
        self.deadline = started + duration
        next_memory = next_logs = next_surface = started
        print(f'Collecting {duration}s from {self.serial}; read-only. Output: {self.output}', flush=True)
        try:
            while time.monotonic() < self.deadline:
                now = time.monotonic()
                elapsed = round(now - started, 3)
                if now >= next_surface and surface_layer:
                    parsed = parse_surface_latency(self.run('shell', 'dumpsys', 'SurfaceFlinger', '--latency', shlex.quote(surface_layer)))
                    if parsed is not None:
                        timestamps = set(parsed['presentedNs'])
                        new_frames = timestamps - seen_frames
                        # The finite history may omit frames if host polling falls behind.
                        if previous_latest and timestamps and previous_latest not in timestamps and min(timestamps) > previous_latest:
                            surface_gaps += 1
                        observed_frames += len(new_frames)
                        seen_frames.update(timestamps)
                        previous_latest = max(timestamps) if timestamps else previous_latest
                        surface_samples += 1
                        self.sample({'elapsedSeconds': elapsed, 'surface': {'newPresentedFrames': len(new_frames), 'presentedFramesObserved': observed_frames, 'refreshPeriodNs': parsed['refreshPeriodNs']}})
                    next_surface = now + 1
                if now >= next_logs:
                    lines = self.run('logcat', '-d', '-v', 'epoch', '--pid=' + pid, '-t', '800', 'ReactNativeJS:I', '*:S').splitlines()
                    for line in lines:
                        record = parse_metrics(line)
                        if not record or record['deviceEpochSeconds'] is None or record['deviceEpochSeconds'] < device_start:
                            continue
                        key = json.dumps(record, sort_keys=True)
                        if key in seen_metrics:
                            continue
                        seen_metrics.add(key)
                        self.sample({'elapsedSeconds': elapsed, **record})
                        (tracking if record['tag'] == TRACKING_TAG else photos).append(record)
                    next_logs = now + 5
                if now >= next_memory:
                    if self.run('shell', 'pidof', self.package).strip() != pid:
                        reason = 'app_process_changed'
                        break
                    point = {'elapsedSeconds': elapsed, **parse_memory(self.run('shell', 'dumpsys', 'meminfo', self.package))}
                    memory.append(point)
                    self.sample({'memory': point})
                    next_memory = now + 10
                    print(f'{elapsed:.0f}/{duration}s; metrics={len(tracking)}, surface_frames={observed_frames}', flush=True)
                time.sleep(min(0.25, max(0, self.deadline - time.monotonic())))
        except KeyboardInterrupt:
            reason = 'interrupted'
        except SmokeError as error:
            reason = 'duration_completed' if time.monotonic() >= self.deadline else 'collection_error'
            if reason == 'collection_error':
                errors.append(str(error))
        finally:
            elapsed = time.monotonic() - started
            self.deadline = None
            try:
                (self.output / 'media-camera-end.txt').write_text(self.run('shell', 'dumpsys', 'media.camera'), encoding='utf-8')
            except SmokeError as error:
                errors.append(str(error))
            pss = [point['pssKiB'] for point in memory if point['pssKiB'] is not None]
            summary = {'schemaVersion': 1, 'collectionStatus': reason, 'elapsedSeconds': round(elapsed, 3),
                       'trackingReports': len(tracking), 'photoReports': len(photos), 'memorySamples': len(memory),
                       'pssFirstKiB': pss[0] if pss else None, 'pssLastKiB': pss[-1] if pss else None,
                       'pssGrowthKiB': pss[-1] - pss[0] if pss else None,
                       'surfaceSamples': surface_samples, 'presentedFramesObserved': observed_frames if surface_samples else None,
                       'surfaceFpsObserved': round(observed_frames / elapsed, 3) if surface_samples and elapsed > 0 else None,
                       'surfaceHistoryGaps': surface_gaps, 'latestTracking': tracking[-1]['metrics'] if tracking else None,
                       'photoLatenciesMs': [item['metrics']['elapsedMs'] for item in photos],
                       'errors': errors, 'acceptanceResult': 'not_assessed',
                       'limitations': ['Surface composition is a preview FPS proxy, not proof of camera throughput.',
                                       'Callback busy/throttle counters do not include CameraX hardware drops.',
                                       'Latency buckets are cumulative within each mounted tracker, not exact percentiles.',
                                       'No gestures, shutter actions, installation, log clearing or device changes were performed.']}
            self.write('summary.json', summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0 if reason == 'duration_completed' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adb', default=default_adb())
    parser.add_argument('--serial', required=True)
    parser.add_argument('--package', default='com.brailleme.app')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=int, default=600)
    parser.add_argument('--surface-layer', help='Exact layer name from SurfaceFlinger --list, optionally inside its API36 RequestedLayerState wrapper.')
    args = parser.parse_args(argv)
    try:
        return RuntimeCollector(args.adb, args.serial, args.package, args.output).collect(args.duration, args.surface_layer)
    except SmokeError as error:
        parser.exit(2, str(error) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
