"""Small, emulator-only ADB evidence helper. Does not decide whether a test passed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
import uuid
import xml.etree.ElementTree as ET


class SmokeError(RuntimeError):
    pass


def bounds(value):
    match = re.fullmatch(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]", value or "")
    if not match:
        return None
    left, top, right, bottom = map(int, match.groups())
    return (left, top, right, bottom) if right > left and bottom > top else None


def find_visible(root, attribute, expected, require_enabled=False):
    """Match exact UI text/description; clip bounds to every visible ancestor."""
    matches = {}

    def visit(node, clip=None, enabled=True, visible=True):
        visible = visible and node.get('visible-to-user') != 'false'
        enabled = enabled and node.get('enabled') != 'false'
        rectangle = bounds(node.get('bounds'))
        if node.get('bounds') is not None and rectangle is None:
            visible = False
        if rectangle and clip:
            rectangle = (max(rectangle[0], clip[0]), max(rectangle[1], clip[1]),
                         min(rectangle[2], clip[2]), min(rectangle[3], clip[3]))
        if rectangle and (rectangle[2] <= rectangle[0] or rectangle[3] <= rectangle[1]):
            visible = False
        if visible and rectangle and node.get(attribute) == expected and (enabled or not require_enabled):
            matches[rectangle] = {'text': node.get('text', ''),
                                  'description': node.get('content-desc', ''),
                                  'bounds': list(rectangle), 'enabled': enabled,
                                  'class': node.get('class', '')}
        for child in node:
            visit(child, rectangle or clip, enabled, visible)

    visit(root)
    return list(matches.values())


class AdbSmoke:
    def __init__(self, adb, serial, output):
        if not re.fullmatch(r'emulator-\d{4,5}', serial):
            raise SmokeError('Serial harus eksplisit emulator-NNNN; perangkat fisik ditolak.')
        self.adb, self.serial = str(adb), serial
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output/'actions.jsonl'

    def log(self, action, **details):
        entry = {'utc': datetime.now(timezone.utc).isoformat(),
                 'serial': self.serial, 'action': action, **details}
        with self.log_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(entry, ensure_ascii=False)+'\n')

    def run(self, *arguments, deadline=None):
        timeout = 8.0 if deadline is None else min(8.0, deadline-time.monotonic())
        if timeout <= 0:
            raise SmokeError('Batas waktu pengamatan tercapai.')
        command = [self.adb, '-s', self.serial, *arguments]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    encoding='utf-8', errors='replace', timeout=timeout,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except (OSError, subprocess.TimeoutExpired) as error:
            self.log('adb_error', arguments=list(arguments), error=str(error))
            raise SmokeError(str(error)) from error
        self.log('adb', arguments=list(arguments), exit_code=result.returncode,
                 stdout=result.stdout[-2000:], stderr=result.stderr[-2000:])
        if result.returncode:
            raise SmokeError(f'ADB gagal: {result.stderr.strip() or result.stdout.strip()}')
        return result.stdout

    def guard(self):
        # The serial prefix alone is insufficient; verify ADB state and Android's QEMU flag.
        if self.run('get-state').strip() != 'device':
            raise SmokeError('Emulator belum tersambung dalam state device.')
        if self.run('shell', 'getprop', 'ro.kernel.qemu').strip() != '1':
            raise SmokeError('Target bukan emulator QEMU; semua tindakan UI ditolak.')
        self.log('emulator_guard_verified')

    def artifact(self, label, suffix):
        label = re.sub(r'[^a-zA-Z0-9_-]+', '-', label).strip('-')[:48] or 'evidence'
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        return self.output/f'{stamp}-{label}-{uuid.uuid4().hex[:6]}{suffix}'

    def pull_evidence(self, remote, local, deadline=None):
        self.run('pull', remote, str(local), deadline=deadline)
        self.log('evidence', path=str(local), sha256=hashlib.sha256(local.read_bytes()).hexdigest())
        # Remove only the unique temporary file this command created on the emulator.
        try:
            self.run('shell', 'rm', '-f', remote, deadline=deadline)
        except SmokeError:
            self.log('temporary_cleanup_incomplete', remote=remote)

    def dump(self, label='ui', deadline=None):
        local = self.artifact(label, '.xml')
        remote = f'/sdcard/brailleme-smoke-{uuid.uuid4().hex}.xml'
        self.run('shell', 'uiautomator', 'dump', '--compressed', remote, deadline=deadline)
        self.pull_evidence(remote, local, deadline)
        try:
            return ET.parse(local).getroot(), local
        except (ET.ParseError, OSError) as error:
            raise SmokeError(f'Hierarki UI tidak valid: {local}') from error

    def locate(self, attribute, expected, timeout, enabled):
        deadline = time.monotonic()+timeout
        last_error = 'Elemen belum terlihat.'
        while time.monotonic() < deadline:
            try:
                root, path = self.dump('locate', deadline)
                matches = find_visible(root, attribute, expected, enabled)
                if len(matches) > 1:
                    raise SmokeError(f'Label ambigu pada {len(matches)} elemen: {matches}')
                if matches:
                    self.log('element_observed', attribute=attribute, exact_value=expected,
                             hierarchy=str(path), element=matches[0])
                    return matches[0]
                last_error = 'Tidak ada elemen terlihat yang cocok persis dan memenuhi status enabled.'
            except SmokeError as error:
                last_error = str(error)
                if last_error.startswith('Label ambigu'):
                    raise
            remaining = deadline-time.monotonic()
            if remaining > 0:
                time.sleep(min(.5, remaining))
        raise SmokeError(f'Pengamatan berakhir setelah {timeout:g}s: {last_error}')

    def screenshot(self, label):
        local = self.artifact(label, '.png')
        remote = f'/sdcard/brailleme-smoke-{uuid.uuid4().hex}.png'
        self.run('shell', 'screencap', '-p', remote)
        self.pull_evidence(remote, local)
        if local.read_bytes()[:8] != b'\x89PNG\r\n\x1a\n':
            raise SmokeError('Hasil screenshot bukan PNG yang valid.')
        return local


def default_adb():
    found = shutil.which('adb')
    if found:
        return found
    sdk = os.environ.get('ANDROID_HOME') or os.environ.get('ANDROID_SDK_ROOT')
    sdk = Path(sdk) if sdk else Path(os.environ.get('LOCALAPPDATA', ''))/'Android/Sdk'
    candidate = sdk/'platform-tools'/('adb.exe' if os.name == 'nt' else 'adb')
    return str(candidate) if candidate.is_file() else 'adb'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adb', default=default_adb())
    parser.add_argument('--serial', required=True, help='Explicit emulator serial, e.g. emulator-5554')
    parser.add_argument('--output', type=Path, required=True, help='Directory for XML/PNG/log/action evidence')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['dump', 'screenshot']:
        commands.add_parser(name).add_argument('--label', default=name)
    for name in ['tap', 'wait']:
        command = commands.add_parser(name)
        selector = command.add_mutually_exclusive_group(required=True)
        selector.add_argument('--text')
        selector.add_argument('--description')
        command.add_argument('--timeout', type=float, default=10)
        if name == 'wait':
            command.add_argument('--enabled', action='store_true')
        else:
            command.add_argument('--count', type=int, choices=[1, 2], default=1,
                                 help='Use2 for two consecutive taps at the same observed target')
    commands.add_parser('key').add_argument('name', choices=['HOME', 'BACK', 'APP_SWITCH'])
    commands.add_parser('start').add_argument('--component', required=True)
    commands.add_parser('logcat').add_argument('--lines', type=int, default=500)
    args = parser.parse_args(argv)
    if args.command in ['tap', 'wait'] and not .25 <= args.timeout <= 30:
        parser.error('--timeout must be between 0.25 and 30 seconds')
    if args.command == 'start' and not re.fullmatch(r'[A-Za-z0-9._]+/[A-Za-z0-9._]+', args.component):
        parser.error('--component must be the explicit package/activity reported by aapt')
    if args.command == 'logcat' and not 1 <= args.lines <= 2000:
        parser.error('--lines must be between 1 and 2000')
    return args


def main(argv=None):
    args = parse_args(argv)
    helper = None
    try:
        helper = AdbSmoke(args.adb, args.serial, args.output)
        helper.guard()
        if args.command == 'dump':
            root, path = helper.dump(args.label)
            nodes = [node.attrib for node in root.iter('node') if node.get('text') or node.get('content-desc')]
            print(json.dumps({'hierarchy': str(path), 'nodes': nodes}, ensure_ascii=False, indent=2))
        elif args.command in ['tap', 'wait']:
            attribute = 'text' if args.text is not None else 'content-desc'
            expected = args.text if args.text is not None else args.description
            element = helper.locate(attribute, expected, args.timeout,
                                    args.command == 'tap' or args.enabled)
            if args.command == 'tap':
                left, top, right, bottom = element['bounds']
                x, y = (left+right)//2, (top+bottom)//2
                for index in range(args.count):
                    helper.run('shell', 'input', 'tap', str(x), str(y))
                    helper.log('tap', element=element, point=[x, y], tap_number=index+1)
            print(json.dumps({'action': args.command, 'element': element}, ensure_ascii=False))
        elif args.command == 'screenshot':
            print(helper.screenshot(args.label))
        elif args.command == 'key':
            helper.run('shell', 'input', 'keyevent', {'HOME':'3','BACK':'4','APP_SWITCH':'187'}[args.name])
        elif args.command == 'start':
            helper.run('shell', 'am', 'start', '-W', '-n', args.component)
        elif args.command == 'logcat':
            value = helper.run('logcat', '-d', '-v', 'threadtime', '-t', str(args.lines))
            path = helper.artifact('logcat', '.log')
            path.write_text(value, encoding='utf-8')
            helper.log('evidence', path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            print(path)
        helper.log('command_completed', command=args.command, interpretation='action/observation only; no test pass decision')
        return 0
    except SmokeError as error:
        if helper:
            helper.log('command_failed', command=args.command, error=str(error))
        print(f'ERROR: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
