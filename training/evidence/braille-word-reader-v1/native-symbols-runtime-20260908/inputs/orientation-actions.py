"""One bounded emulator rotation observation; restores both settings and never infers acceptance."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('runtime_actions', HERE/'runtime-actions.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
SETTING_VALUES = {'accelerometer_rotation': {'0', '1'}, 'user_rotation': {'0', '1', '2', '3', 'null'}}


def installed_apk(helper, expected):
    value = helper.run('shell', 'pm', 'path', runtime.PACKAGE).strip()
    match = re.fullmatch(r'package:(/data/app/[A-Za-z0-9_./=+~\-]+/base\.apk)', value)
    if not match:
        raise runtime.SmokeError('Installed APK path is missing, split or unexpected.')
    apk_path = match.group(1)
    digest = helper.run('shell', 'sha256sum', apk_path).strip()
    match = re.fullmatch(r'([0-9a-fA-F]{64})\s+\*?' + re.escape(apk_path), digest)
    actual = match.group(1).lower() if match else None
    helper.log('installed_apk_observed', path=apk_path, expected_sha256=expected, actual_sha256=actual, matches=actual == expected)
    if actual != expected:
        raise runtime.SmokeError('Installed APK hash differs from the requested artifact.')
    return {'path': apk_path, 'sha256': actual}


def host_metro_observation(helper):
    # Query the host listener table only; do not connect to or launch any daemon.
    command = ['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
               "$ErrorActionPreference='Stop'; @(Get-NetTCPConnection -State Listen | Where-Object { $_.LocalPort -eq 8081 } | Select-Object LocalAddress,LocalPort,OwningProcess) | ConvertTo-Json -Compress"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode:
            raise ValueError('Listener query exited ' + str(result.returncode))
        listeners = json.loads(result.stdout or '[]')
        if isinstance(listeners, dict):
            listeners = [listeners]
        if not isinstance(listeners, list) or any(item.get('LocalPort') != 8081 for item in listeners):
            raise ValueError('Unexpected listener query result')
        observation = {'status': 'observed', 'listeners': listeners, 'port_8081_listening': bool(listeners),
                       'limitation': 'Host listener observation only; this does not identify a Metro process or prove all network use absent.'}
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        observation = {'status': 'unavailable', 'error': str(error)}
    helper.log('host_port_8081_observed', **observation)
    return observation


def rotation_evidence(helper, label):
    paths = {}
    for name, arguments in [('window', ('shell', 'dumpsys', 'window', 'displays')),
                            ('display', ('shell', 'dumpsys', 'display'))]:
        value = helper.run(*arguments)
        path = helper.artifact(label + '-' + name, '.txt')
        path.write_text(value, encoding='utf-8')
        helper.log('evidence', path=str(path), sha256=runtime.sha(path), scope='actual native display/window rotation state')
        paths[name] = str(path)
    return paths


def snapshot_same_pid(helper, label, expected_pid):
    helper.snapshot(label)
    actual = helper.observations[-1]['pid']
    if actual != expected_pid:
        raise runtime.SmokeError(f'App PID changed during orientation QA: expected={expected_pid}, observed={actual}.')


def restore_settings(helper, originals):
    results = {}
    for name in ('user_rotation', 'accelerometer_rotation'):
        original = originals[name]
        try:
            arguments = ('delete', 'system', name) if original == 'null' else ('put', 'system', name, original)
            helper.mutation('restore-' + name, 'shell', 'settings', *arguments)
            actual = helper.run('shell', 'settings', 'get', 'system', name).strip()
            results[name] = {'expected': original, 'observed': actual, 'restored': actual == original}
        except runtime.SmokeError as error:
            results[name] = {'expected': original, 'restored': False, 'error': str(error)}
        helper.log('orientation_setting_restored', name=name, **results[name])
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adb', type=Path, required=True)
    parser.add_argument('--serial', required=True)
    parser.add_argument('--component', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ui-dump-dex', required=True)
    parser.add_argument('--ui-dump-sha256', required=True)
    parser.add_argument('--expected-apk-sha256', required=True)
    args = parser.parse_args(argv)
    if not args.adb.is_file() or not re.fullmatch(r'[0-9a-fA-F]{64}', args.expected_apk_sha256):
        parser.error('ADB must exist and expected APK SHA-256 must be exact.')
    helper = None
    originals, restore_results, rotation_files = {}, {}, {}
    changed = False
    expected_pid = None
    status, error = 'completed_actions_not_assessed', None
    overall_deadline = time.monotonic() + 120
    metadata = {'schema_version': 1, 'started_utc': datetime.now(timezone.utc).isoformat(),
                'script_sha256': runtime.sha(__file__), 'runtime_actions_sha256': runtime.sha(HERE/'runtime-actions.py'),
                'ui_dump_dex': args.ui_dump_dex, 'ui_dump_sha256': args.ui_dump_sha256,
                'total_deadline_seconds': 120, 'restoration_reserve_seconds': 30,
                'scope': 'Orientation command/observation only; no acceptance, camera accuracy, audio or physical claim.'}
    try:
        helper = runtime.Actions(args.adb.resolve(), args.serial, args.output, args.component, args.ui_dump_dex, args.ui_dump_sha256)
        helper.deadline = overall_deadline - 30
        helper.guard()
        if helper.run('shell', 'getprop', 'ro.build.version.sdk').strip() != '36':
            raise runtime.SmokeError('Only API36 emulator is authorized.')
        package = helper.run('shell', 'dumpsys', 'package', runtime.PACKAGE)
        if not re.search(r'\bversionName=0\.3\.0(?:\s|$)', package) or not re.search(r'\bversionCode=3(?:\s|$)', package):
            raise runtime.SmokeError('Installed app must be0.3.0/code3.')
        metadata['installed_apk'] = installed_apk(helper, args.expected_apk_sha256.lower())
        metadata['host_port_8081'] = host_metro_observation(helper)
        expected_pid = helper.app_pid()
        if expected_pid is None:
            raise runtime.SmokeError('App must already be running; this helper never launches it.')
        metadata['expected_pid'] = expected_pid
        for name, allowed in SETTING_VALUES.items():
            original = helper.run('shell', 'settings', 'get', 'system', name).strip()
            if original not in allowed:
                raise runtime.SmokeError('Unexpected original setting ' + name)
            originals[name] = original
        metadata['original_settings'] = dict(originals)
        (helper.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
        snapshot_same_pid(helper, 'orientation-before', expected_pid)
        rotation_files['before'] = rotation_evidence(helper, 'orientation-before')
        changed = True
        helper.mutation('orientation-accel-off', 'shell', 'settings', 'put', 'system', 'accelerometer_rotation', '0')
        helper.mutation('orientation-user-rotation-1', 'shell', 'settings', 'put', 'system', 'user_rotation', '1')
        for name, expected in [('accelerometer_rotation', '0'), ('user_rotation', '1')]:
            actual = helper.run('shell', 'settings', 'get', 'system', name).strip()
            helper.log('orientation_setting_configured', name=name, expected=expected, observed=actual)
            if actual != expected:
                raise runtime.SmokeError('Requested setting was not confirmed: ' + name)
        helper.pause(2)
        snapshot_same_pid(helper, 'orientation-rotated-first', expected_pid)
        rotation_files['rotated_first'] = rotation_evidence(helper, 'orientation-rotated-first')
        helper.pause(3)
        snapshot_same_pid(helper, 'orientation-rotated-second', expected_pid)
        rotation_files['rotated_second'] = rotation_evidence(helper, 'orientation-rotated-second')
    except (runtime.SmokeError, OSError, ValueError, KeyboardInterrupt) as failure:
        status, error = 'action_or_observation_incomplete', str(failure)
    finally:
        if helper:
            if changed:
                helper.deadline = overall_deadline
                restore_results = restore_settings(helper, originals)
                if not all(row['restored'] for row in restore_results.values()):
                    status, error = 'action_or_observation_incomplete', 'Both original rotation settings were not confirmed restored.'
                try:
                    helper.pause(2)
                    snapshot_same_pid(helper, 'orientation-restored', expected_pid)
                    rotation_files['restored'] = rotation_evidence(helper, 'orientation-restored')
                except (runtime.SmokeError, OSError) as failure:
                    status, error = 'action_or_observation_incomplete', str(failure)
            summary = {'schema_version': 1, 'status': status, 'error': error, 'acceptanceResult': 'not_assessed',
                       'finished_utc': datetime.now(timezone.utc).isoformat(), 'original_settings': originals,
                       'restoration': restore_results, 'expected_pid': expected_pid, 'observations': helper.observations,
                       'rotation_evidence': rotation_files, 'action_counts': helper.action_counts,
                       'interpretation': 'Requested settings, actual display dumps and UI observations retained; no rotation/camera acceptance automatically inferred.'}
            (helper.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            print(json.dumps({'status': status, 'error': error, 'summary': str(helper.output/'summary.json')}, ensure_ascii=False))
    return 0 if status == 'completed_actions_not_assessed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
