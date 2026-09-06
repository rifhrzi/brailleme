"""Execute a frozen C API bundle on an already running, explicitly selected Android device."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import uuid


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiled', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adb', type=Path, required=True)
    parser.add_argument('--serial', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new evidence output directory')
    compiled = json.loads((args.compiled/'compile-report.json').read_text())
    manifest = json.loads((args.bundle/'manifest.json').read_text())
    if manifest['status'] != 'trained_candidate_ready_native_execution_pending':
        raise ValueError('Actual trained candidate and its desktop tensors are required')
    for base, rows in ((args.compiled, compiled['files']), (args.bundle, manifest['files'])):
        for row in rows:
            path = base/row['path']
            if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
                raise ValueError(f'Frozen file changed: {path}')
    if sha(args.bundle/'model.tflite') != manifest['model_sha256']:
        raise ValueError('Model identity mismatch')
    adb = [str(args.adb), '-s', args.serial]
    subprocess.run(adb+['get-state'], check=True, capture_output=True)
    abi = subprocess.check_output(adb+['shell', 'getprop', 'ro.product.cpu.abi'], text=True).strip()
    if abi != compiled['abi']:
        raise ValueError(f'Compiled ABI {compiled["abi"]} differs from device ABI {abi}')
    args.output.mkdir(parents=True)
    remote = '/data/local/tmp/word-interpreter-'+uuid.uuid4().hex
    commands = []
    def call(parts, **kwargs):
        command = adb+parts
        commands.append(command)
        return subprocess.run(command, check=True, capture_output=True, **kwargs)
    call(['shell', 'mkdir', '-p', remote, remote+'/inputs', remote+'/outputs'])
    inputs = [remote+'/'+row['input'] for row in manifest['cases']]
    input_list = args.output/'inputs.txt'
    input_list.write_text('\n'.join(inputs)+'\n', encoding='ascii', newline='\n')
    files = [(args.compiled/'word-interpreter', 'word-interpreter'),
             (args.compiled/'libtensorflowlite_jni.so', 'libtensorflowlite_jni.so'),
             (args.bundle/'model.tflite', 'model.tflite'), (input_list, 'inputs.txt')]
    files += [(args.bundle/row['input'], row['input']) for row in manifest['cases']]
    transferred = []
    for local, name in files:
        target = remote+'/'+name
        call(['push', str(local), target])
        observed = call(['shell', 'sha256sum', target], text=True).stdout.split()[0]
        if observed != sha(local):
            raise ValueError(f'Android transfer hash mismatch: {name}')
        transferred.append({'path': name, 'sha256': observed, 'bytes': local.stat().st_size})
    call(['shell', 'chmod', '755', remote+'/word-interpreter'])
    fingerprint = call(['shell', 'getprop', 'ro.build.fingerprint'], text=True).stdout.strip()
    command = adb+['shell', 'env', 'LD_LIBRARY_PATH='+remote, remote+'/word-interpreter',
                   remote+'/model.tflite', remote+'/inputs.txt', remote+'/outputs']
    commands.append(command)
    process = subprocess.run(command, capture_output=True, timeout=1800)
    (args.output/'native-stdout.jsonl').write_bytes(process.stdout)
    (args.output/'native-stderr.txt').write_bytes(process.stderr)
    call(['pull', remote+'/outputs', str(args.output)])
    report = {'status': 'native_process_completed_comparison_pending' if process.returncode == 0 else 'native_process_failed',
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'returncode': process.returncode,
              'serial': args.serial, 'abi': abi, 'fingerprint': fingerprint, 'remote': remote,
              'model_sha256': manifest['model_sha256'], 'bundle_manifest_sha256': sha(args.bundle/'manifest.json'),
              'compile_report_sha256': sha(args.compiled/'compile-report.json'), 'source_sha256': sha(__file__),
              'transferred_files_verified_on_device': transferred, 'commands': commands,
              'files': [{'path': p.relative_to(args.output).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size}
                        for p in sorted(args.output.rglob('*')) if p.is_file()],
              'limits': ['Standalone Android C API invocation; no Java/Nitro bridge, main app, camera or physical-accuracy claim.',
                         'No emulator was started or stopped by this script. Remote diagnostic files are retained for review.']}
    (args.output/'execution.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'returncode': process.returncode}, indent=2))
    process.check_returncode()


if __name__ == '__main__':
    main()
