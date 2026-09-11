"""Compile a standalone no-idle UI snapshot DEX locally; never use ADB."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--java-home', type=Path, required=True)
    parser.add_argument('--android-jar', type=Path, required=True)
    parser.add_argument('--d8', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.exists():
        parser.error('Output must be a new directory')
    for path in [args.java_home/'bin/javac.exe', args.java_home/'bin/java.exe', args.android_jar, args.d8]:
        if not path.is_file():
            parser.error(f'Missing tool/input: {path}')
    output.mkdir(parents=True)
    classes = output/'classes'
    classes.mkdir()
    env = dict(os.environ, JAVA_HOME=str(args.java_home))
    commands = [
        [str(args.java_home/'bin/javac.exe'), '-source', '8', '-target', '8', '-encoding', 'UTF-8', '-classpath', str(args.android_jar), '-d', str(classes), str(source/'BrailleUiDump.java'), str(source/'BrailleUiDumpHostTest.java')],
        [str(args.java_home/'bin/java.exe'), '-cp', os.pathsep.join([str(classes), str(args.android_jar)]), 'BrailleUiDumpHostTest'],
        [str(args.d8), '--min-api', '29', '--lib', str(args.android_jar), '--output', str(output), str(classes/'BrailleUiDump.class')],
    ]
    steps = []
    for index, command in enumerate(commands):
        result = subprocess.run(command, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        (output/f'step-{index}.stdout.txt').write_text(result.stdout, encoding='utf-8')
        (output/f'step-{index}.stderr.txt').write_text(result.stderr, encoding='utf-8')
        steps.append({'command': command, 'exit_code': result.returncode})
        if result.returncode:
            raise RuntimeError(f'Build step{index} failed; see retained logs')
    def identity(path):
        return {'file': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    report = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'compiled_host_guards_passed_android_execution_pending', 'steps': steps,
              'inputs': [identity(source/name) for name in ['BrailleUiDump.java', 'BrailleUiDumpHostTest.java', 'build.py']] + [identity(args.android_jar)],
              'dex': identity(output/'classes.dex'), 'limitation': 'No ADB or Android execution. Reflection connection and live hierarchy still require emulator verification.'}
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report['dex']))


if __name__ == '__main__':
    main()
