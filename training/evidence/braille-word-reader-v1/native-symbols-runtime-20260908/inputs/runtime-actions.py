"""Bounded actions for one authorized 0.3.0 emulator; observations are not pass claims."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/word-android-smoke'))
from smoke import AdbSmoke, SmokeError, find_visible

PACKAGE = 'com.brailleme.app'
AVD = 'BrailleMe_API_36'
SCANNER_STATES = ['Mencari Braille', 'Braille terdeteksi', 'Tahan kamera, lalu tekan Pindai',
                  'Mengambil foto', 'Membaca Braille', 'Hasil terbaca', 'Pindai ulang']
NAVIGATION = ['Buka Beranda', 'Buka pemindai kata', 'Lihat status pemindai', 'Aktifkan kamera', 'Izinkan kamera']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def button(root, label, enabled=True):
    matches = {}
    for attribute in ('content-desc', 'text'):
        for item in find_visible(root, attribute, label, enabled):
            # The bottom tab may also be called Pindai; only the actual button is actionable.
            if item['class'].endswith('Button'):
                matches[tuple(item['bounds'])] = item
    if len(matches) > 1:
        raise SmokeError(f'Ambiguous visible button {label!r}; no tap sent.')
    return next(iter(matches.values()), None)


def observed(root):
    states = [label for label in SCANNER_STATES if find_visible(root, 'text', label)]
    results = sorted({node.get('content-desc')[7:] for node in root.iter('node')
                      if node.get('content-desc', '').startswith('Hasil: ')
                      and find_visible(root, 'content-desc', node.get('content-desc'))})
    scan = button(root, 'Pindai', False)
    return {'visible_states': states, 'result_texts': results,
            'shutter_visible': scan is not None, 'shutter_enabled': bool(scan and scan['enabled']),
            'rescan_visible': button(root, 'Pindai ulang', False) is not None,
            'processing_visible': button(root, 'Memproses foto', False) is not None}


class Actions(AdbSmoke):
    def __init__(self, adb, serial, output, component, ui_dump_dex=None, ui_dump_sha256=None):
        if serial != 'emulator-5554':
            raise SmokeError('Only explicit serial emulator-5554 is authorized.')
        if not re.fullmatch(r'com\.brailleme\.app/[A-Za-z0-9._]+', component):
            raise SmokeError('Use the explicit com.brailleme.app activity reported by aapt.')
        if bool(ui_dump_dex) != bool(ui_dump_sha256):
            raise SmokeError('Provide the remote UI dump DEX path and SHA-256 together.')
        if ui_dump_dex and (not re.fullmatch(r'/data/local/tmp/brailleme-ui-dump-[A-Za-z0-9_-]+\.dex', ui_dump_dex)
                            or not re.fullmatch(r'[0-9a-fA-F]{64}', ui_dump_sha256)):
            raise SmokeError('UI dump DEX must use the task remote path and an exact SHA-256.')
        if Path(output).exists():
            raise SmokeError('Use a new output directory; old evidence is never overwritten.')
        super().__init__(adb, serial, output)
        self.component, self.deadline = component, None
        self.ui_dump_dex = ui_dump_dex
        self.ui_dump_sha256 = ui_dump_sha256.lower() if ui_dump_sha256 else None
        self.observations, self.action_counts = [], {}

    def run(self, *arguments, deadline=None):
        if self.deadline is not None:
            deadline = self.deadline if deadline is None else min(deadline, self.deadline)
        return super().run(*arguments, deadline=deadline)

    def guard(self):
        super().guard()
        avd_lines = [line.strip() for line in self.run('emu', 'avd', 'name').splitlines() if line.strip() and line.strip() != 'OK']
        if avd_lines != [AVD]:
            raise SmokeError(f'AVD mismatch: {avd_lines!r}; mutation rejected.')
        self.log('avd_guard_verified', avd=AVD)

    def dump(self, label='ui', deadline=None):
        if not self.ui_dump_dex:
            return super().dump(label, deadline)
        self.guard()
        if self.deadline is not None:
            deadline = self.deadline if deadline is None else min(deadline, self.deadline)
        reported = self.run('shell', 'sha256sum', self.ui_dump_dex, deadline=deadline).strip()
        match = re.fullmatch(r'([0-9a-fA-F]{64})\s+\*?' + re.escape(self.ui_dump_dex), reported)
        actual_sha = match.group(1).lower() if match else None
        self.log('ui_dump_dex_identity', remote=self.ui_dump_dex, expected_sha256=self.ui_dump_sha256,
                 actual_sha256=actual_sha, verified=actual_sha == self.ui_dump_sha256)
        if actual_sha != self.ui_dump_sha256:
            raise SmokeError('Remote UI dump DEX hash mismatch; helper execution rejected.')
        local = self.artifact(label, '.xml')
        remote = '/data/local/tmp/brailleme-ui-' + uuid.uuid4().hex + '.xml'
        arguments = ['shell', 'env', 'CLASSPATH=' + self.ui_dump_dex, 'app_process', '/system/bin', 'BrailleUiDump', remote]
        timeout = 15.0 if deadline is None else min(15.0, deadline - time.monotonic())
        if timeout <= 0:
            raise SmokeError('Batas waktu pengamatan tercapai.')
        try:
            result = subprocess.run([self.adb, '-s', self.serial, *arguments], capture_output=True,
                                    text=True, encoding='utf-8', errors='replace', timeout=timeout,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except (OSError, subprocess.TimeoutExpired) as error:
            self.log('adb_error', arguments=arguments, error=str(error), ui_snapshot_mode='no-idle')
            raise SmokeError(str(error)) from error
        self.log('adb', arguments=arguments, exit_code=result.returncode, stdout=result.stdout[-2000:],
                 stderr=result.stderr[-2000:], ui_snapshot_mode='no-idle')
        if result.returncode or not re.search(r'^UI_DUMP_WRITTEN nodes=[1-9]\d* durationMs=\d+\s*$', result.stdout, re.MULTILINE):
            raise SmokeError(f'No-idle UI dump failed: exit={result.returncode}; {result.stderr.strip() or result.stdout.strip()}')
        self.pull_evidence(remote, local, deadline)
        try:
            root = ET.parse(local).getroot()
            if root.tag != 'hierarchy' or root.get('snapshot-mode') != 'no-idle' or not list(root.iter('node')):
                raise ValueError('Missing no-idle hierarchy or nodes')
        except (ET.ParseError, OSError, ValueError) as error:
            raise SmokeError(f'No-idle UI hierarchy is invalid: {local}') from error
        self.log('ui_snapshot_observed', mode='no-idle', hierarchy=str(local), dex_sha256=actual_sha,
                 started_uptime_ms=root.get('started-uptime-ms'), root_wait_ms=root.get('root-wait-ms'),
                 interpretation='Non-atomic current accessibility traversal; no idle condition or app acceptance inferred.')
        return root, local

    def mutation(self, label, *arguments):
        self.guard()
        self.run(*arguments)
        self.action_counts[label] = self.action_counts.get(label, 0) + 1
        self.log('ui_action_sent', label=label, interpretation='command sent, outcome observed separately')
        print(f'Action: {label}', flush=True)

    def pause(self, seconds):
        end = time.monotonic() + seconds
        if self.deadline:
            end = min(end, self.deadline)
        while time.monotonic() < end:
            time.sleep(min(.25, end - time.monotonic()))

    def app_pid(self):
        """pidof exit1 with empty output means no process; other ADB errors still fail."""
        arguments = ['shell', 'pidof', PACKAGE]
        timeout = 8.0 if self.deadline is None else min(8.0, self.deadline - time.monotonic())
        if timeout <= 0:
            raise SmokeError('Batas waktu pengamatan tercapai.')
        try:
            result = subprocess.run([self.adb, '-s', self.serial, *arguments], capture_output=True,
                                    text=True, encoding='utf-8', errors='replace', timeout=timeout,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except (OSError, subprocess.TimeoutExpired) as error:
            self.log('adb_error', arguments=arguments, error=str(error))
            raise SmokeError(str(error)) from error
        self.log('adb', arguments=arguments, exit_code=result.returncode,
                 stdout=result.stdout[-2000:], stderr=result.stderr[-2000:])
        pid = result.stdout.strip()
        if result.returncode == 1 and not pid and not result.stderr.strip():
            self.log('app_pid_observed', pid=None, observation='pidof returned no matching process')
            return None
        if result.returncode != 0 or not re.fullmatch(r'[1-9]\d*', pid):
            raise SmokeError(f'ADB pidof failed or returned an unexpected process list: exit={result.returncode}, {result.stderr.strip() or pid!r}')
        self.log('app_pid_observed', pid=pid)
        return pid

    def snapshot(self, label, *, require_app=True):
        self.guard()
        root, xml = self.dump(label)
        png = self.screenshot(label)
        pid = self.app_pid()
        log_path = None
        if pid is not None:
            logs = self.run('logcat', '-d', '-v', 'epoch', '--pid=' + pid, '-t', '1000')
            log_path = self.artifact(label, '.log')
            log_path.write_text(logs, encoding='utf-8')
            self.log('evidence', path=str(log_path), sha256=sha(log_path))
        record = {'label': label, 'utc': datetime.now(timezone.utc).isoformat(), 'pid': pid or None,
                  'xml': str(xml), 'screenshot': str(png), 'logcat': str(log_path) if log_path else None,
                  'app_process_required': require_app,
                  **observed(root)}
        self.observations.append(record)
        self.log('ui_observed', **record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if require_app and pid is None:
            raise SmokeError(f'App process is missing during {label}; visible scanner UI may be stale. Readiness/action outcome is incomplete.')
        return root

    def tap(self, item, label):
        left, top, right, bottom = item['bounds']
        self.log('tap_target', label=label, element=item)
        self.mutation(label, 'shell', 'input', 'tap', str((left + right) // 2), str((top + bottom) // 2))
        self.pause(.5)
        return self.snapshot('after-' + label)

    def scroll(self, root, label):
        containers = find_visible(root, 'scrollable', 'true', True)
        if not containers:
            return False
        containers.sort(key=lambda item: (item['bounds'][2] - item['bounds'][0]) * (item['bounds'][3] - item['bounds'][1]), reverse=True)
        left, top, right, bottom = containers[0]['bounds']
        x = (left + right) // 2
        y1, y2 = top + (bottom - top) * 3 // 4, top + (bottom - top) // 4
        self.log('scroll_target', container=containers[0], points=[x, y1, x, y2])
        self.mutation(label, 'shell', 'input', 'swipe', str(x), str(y1), str(x), str(y2), '400')
        self.pause(.5)
        self.snapshot('after-' + label)
        return True

    def prepare(self):
        scrolls = 0
        for step in range(20):
            root = self.snapshot(f'prepare-{step:02d}')
            if button(root, 'Pindai', False) or button(root, 'Pindai ulang', False):
                return
            if find_visible(root, 'text', 'Pemindai belum siap'):
                raise SmokeError('Model initialization error is visible; no automatic retry hides this result.')
            acted = False
            for label in NAVIGATION:
                item = button(root, label)
                if item:
                    self.tap(item, label)
                    acted = True
                    break
            if acted:
                continue
            for package in ('com.google.android.permissioncontroller', 'com.android.permissioncontroller'):
                for suffix in ('permission_allow_foreground_only_button', 'permission_allow_button'):
                    items = find_visible(root, 'resource-id', package + ':id/' + suffix, True)
                    if len(items) == 1:
                        self.tap(items[0], 'allow-camera-foreground')
                        acted = True
                        break
                if acted:
                    break
            if acted:
                continue
            if observed(root)['visible_states']:
                return
            if scrolls < 4 and self.scroll(root, f'prepare-scroll-{scrolls}'):
                scrolls += 1
            else:
                self.pause(1)
        raise SmokeError('Scanner controls not reached within bounded onboarding; inspect recorded UI.')

    def cycle(self, index):
        root = self.snapshot(f'cycle-{index:02d}-before')
        retry = button(root, 'Pindai ulang')
        if retry:
            root = self.tap(retry, 'Pindai ulang')
            self.pause(2)
            root = self.snapshot(f'cycle-{index:02d}-tracking')
        shutter = button(root, 'Pindai')
        if shutter:
            self.tap(shutter, 'Pindai')
            finish_by = min(self.deadline, time.monotonic() + 20)
            while time.monotonic() < finish_by:
                self.pause(1)
                root = self.snapshot(f'cycle-{index:02d}-result')
                if button(root, 'Pindai ulang', False):
                    return
            self.log('capture_observation_timeout', cycle=index, interpretation='No terminal control observed; no success inferred.')
        else:
            state = observed(root)
            self.log('capture_not_sent', cycle=index, **state)
            # At large font, one bounded scroll may reveal the actual shutter.
            if not state['shutter_visible'] and not state['processing_visible']:
                self.scroll(root, f'cycle-{index:02d}-scroll')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adb', type=Path, required=True)
    parser.add_argument('--serial', required=True)
    parser.add_argument('--component', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feed-manifest', type=Path)
    parser.add_argument('--feed-name')
    parser.add_argument('--ui-dump-dex', help='Already-pushed /data/local/tmp/brailleme-ui-dump-NAME.dex; no auto-push')
    parser.add_argument('--ui-dump-sha256', help='Exact expected SHA-256, checked remotely before each helper execution')
    parser.add_argument('--duration', type=int, default=600)
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('command', choices=['prepare', 'exercise', 'background', 'cold-start', 'font-200'])
    args = parser.parse_args(argv)
    if not args.adb.is_file() or not 1 <= args.duration <= 600 or not 10 <= args.interval <= 60:
        parser.error('ADB must exist; duration1..600 and interval10..60 seconds are required.')
    if bool(args.feed_manifest) != bool(args.feed_name):
        parser.error('Provide --feed-manifest and --feed-name together.')
    if bool(args.ui_dump_dex) != bool(args.ui_dump_sha256):
        parser.error('Provide --ui-dump-dex and --ui-dump-sha256 together.')
    metadata = {'scope': 'UI commands and actual observations only; no model accuracy, audio, physical-camera or release pass claim.',
                'command': args.command, 'script_sha256': sha(__file__),
                'helper_sha256': sha(REPO / 'scripts/word-android-smoke/smoke.py'),
                'declared_feed': None, 'started_utc': datetime.now(timezone.utc).isoformat(),
                'ui_dump': {'mode': 'no-idle' if args.ui_dump_dex else 'stock-uiautomator',
                            'remote_dex': args.ui_dump_dex, 'expected_sha256': args.ui_dump_sha256,
                            'verification': 'Each no-idle execution records its actual remote hash in actions.jsonl.'}}
    if args.feed_manifest:
        feed = json.loads(args.feed_manifest.read_text(encoding='utf-8'))
        matches = [row for row in feed['rows'] if row['output'] == args.feed_name]
        if len(matches) != 1 or Path(args.feed_name).name != args.feed_name:
            parser.error('Feed must match exactly one entry of the supplied labelled manifest.')
        file = args.feed_manifest.resolve().parent / args.feed_name
        if sha(file) != matches[0]['output_sha256']:
            parser.error('Declared feed pixels differ from frozen manifest.')
        metadata['declared_feed'] = {'manifest': str(args.feed_manifest.resolve()), 'manifest_sha256': sha(args.feed_manifest),
                                     'entry': matches[0], 'binding': 'Operator-declared feed; verify actual camera orientation/placement visually.'}
    helper, original_font, font_changed = None, None, False
    status, error = 'completed_actions_not_assessed', None
    try:
        helper = Actions(args.adb.resolve(), args.serial, args.output, args.component, args.ui_dump_dex, args.ui_dump_sha256)
        helper.deadline = time.monotonic() + (args.duration if args.command == 'exercise' else 120)
        helper.guard()
        package = helper.run('shell', 'dumpsys', 'package', PACKAGE)
        if not re.search(r'\bversionName=0\.3\.0(?:\s|$)', package) or not re.search(r'\bversionCode=3(?:\s|$)', package):
            raise SmokeError('Installed app is not 0.3.0/code3; no UI actions sent.')
        (helper.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        if args.command == 'exercise':
            index, next_cycle = 0, time.monotonic()
            while time.monotonic() < helper.deadline - 10:
                helper.pause(max(0, next_cycle - time.monotonic()))
                if time.monotonic() >= helper.deadline - 10:
                    break
                helper.cycle(index)
                index += 1
                next_cycle += args.interval
            helper.pause(max(0, helper.deadline - time.monotonic()))
        else:
            helper.snapshot('before-' + args.command, require_app=False)
            if args.command == 'cold-start':
                helper.mutation('force-stop', 'shell', 'am', 'force-stop', PACKAGE)
                helper.snapshot('after-force-stop', require_app=False)
            elif args.command == 'background':
                helper.mutation('background-home', 'shell', 'input', 'keyevent', '3')
                helper.snapshot('background-home')
                helper.pause(3)
            elif args.command == 'font-200':
                original_font = helper.run('shell', 'settings', 'get', 'system', 'font_scale').strip()
                if original_font != 'null' and not re.fullmatch(r'(?:[0-2](?:\.\d+)?|3(?:\.0+)?)', original_font):
                    raise SmokeError('Unexpected font_scale value; no font mutation sent.')
                helper.log('font_scale_original', value=original_font)
                font_changed = True
                helper.mutation('font-scale-200', 'shell', 'settings', 'put', 'system', 'font_scale', '2.0')
                configured = helper.run('shell', 'settings', 'get', 'system', 'font_scale').strip()
                helper.log('font_scale_configured', expected='2.0', observed=configured)
                if float(configured) != 2.0:
                    raise SmokeError('Font 200% setting was not confirmed.')
                helper.snapshot('font-200-configured')
            helper.mutation('launch-existing-app', 'shell', 'am', 'start', '-W', '-n', args.component)
            helper.prepare()
            if args.command == 'font-200':
                helper.cycle(0)
    except (SmokeError, OSError, ValueError, KeyboardInterrupt) as exception:
        status, error = 'action_or_observation_incomplete', str(exception)
        print(f'Incomplete: {exception}', flush=True)
    finally:
        if helper:
            if font_changed:
                helper.deadline = time.monotonic() + 25
                try:
                    command = ('delete', 'system', 'font_scale') if original_font == 'null' else ('put', 'system', 'font_scale', original_font)
                    helper.mutation('restore-font-scale', 'shell', 'settings', *command)
                    restored = helper.run('shell', 'settings', 'get', 'system', 'font_scale').strip()
                    helper.log('font_scale_restored', expected=original_font, observed=restored)
                    if restored != original_font:
                        raise SmokeError('Original font_scale restoration was not confirmed.')
                    helper.snapshot('font-restored')
                except SmokeError as restore_error:
                    status, error = 'action_or_observation_incomplete', str(restore_error)
            summary = {'status': status, 'error': error, 'acceptanceResult': 'not_assessed',
                       'action_counts': helper.action_counts, 'observations': helper.observations,
                       'finished_utc': datetime.now(timezone.utc).isoformat(),
                       'command': args.command, 'runtime_collector_coordination': 'operator_managed',
                       'note': 'Run cold-start/font separately; the existing runtime collector stops when the app PID changes.'}
            (helper.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return 0 if status == 'completed_actions_not_assessed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
