"""Compile one standalone Android C runner with cached exact LiteRT; never use ADB."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_AAR_SHA256 = '54b567572cb2d1295571c785198f87597809cbf22d6cb8acfeac23869041887d'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--aar', type=Path, required=True)
    parser.add_argument('--ndk', type=Path, required=True)
    parser.add_argument('--abi', choices=['x86_64', 'arm64-v8a'], default='x86_64')
    args = parser.parse_args()
    if sha(args.aar) != EXPECTED_AAR_SHA256:
        raise ValueError('Cached LiteRT 1.4.0 AAR bytes differ from the frozen dependency')
    if args.stage.exists():
        raise ValueError('Use a new isolated stage; existing evidence must be retained')
    args.stage.mkdir(parents=True)
    stage = args.stage.resolve()
    source = Path(__file__).with_name('word_interpreter.c')
    package = ROOT/'node_modules/react-native-fast-tflite'
    if json.loads((package/'package.json').read_text())['version'] != '3.0.1':
        raise ValueError('Expected fast-tflite 3.0.1 source')
    if 'com.google.ai.edge.litert:litert:1.4.0' not in (package/'android/build.gradle').read_text():
        raise ValueError('Expected exact LiteRT 1.4.0 dependency')
    with zipfile.ZipFile(args.aar) as archive:
        for name in archive.namelist():
            if name.startswith('headers/') and not name.endswith('/'):
                target = stage/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
        library = stage/'libtensorflowlite_jni.so'
        library.write_bytes(archive.read(f'jni/{args.abi}/libtensorflowlite_jni.so'))
    shutil.copyfile(source, stage/source.name)
    tools = args.ndk/'toolchains/llvm/prebuilt/windows-x86_64/bin'
    triple = {'x86_64': 'x86_64-linux-android29', 'arm64-v8a': 'aarch64-linux-android29'}[args.abi]
    command = [str(tools/'clang.exe'), f'--target={triple}', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
               '-fPIE', '-pie', '-I'+str(stage/'headers'), '-I'+str(stage/'headers/external/org_tensorflow'),
               str(stage/source.name), '-L'+str(stage), '-ltensorflowlite_jni', '-lm',
               '-Wl,-rpath,$ORIGIN', '-Wl,-z,max-page-size=16384', '-o', str(stage/'word-interpreter')]
    compiled = subprocess.run(command, capture_output=True, text=True)
    (stage/'compiler.log').write_text(compiled.stdout+compiled.stderr, encoding='utf-8')
    compiled.check_returncode()
    readelf = [str(tools/'llvm-readelf.exe'), '-h', '-d', '-Ws', str(stage/'word-interpreter')]
    elf = subprocess.check_output(readelf, text=True)
    (stage/'runner-elf.txt').write_text(elf, encoding='utf-8')
    library_elf = subprocess.check_output([str(tools/'llvm-readelf.exe'), '-h', '-d', str(library)], text=True)
    (stage/'library-elf.txt').write_text(library_elf, encoding='utf-8')
    files = [p for p in stage.rglob('*') if p.is_file()]
    source_paths = [Path(__file__), source, package/'cpp/HybridTfliteModule.cpp',
                    package/'cpp/HybridTfliteModel.cpp', package/'android/build.gradle',
                    package/'package.json', ROOT/'src/ml/word-model-loader.ts', ROOT/'package-lock.json']
    report = {'status': 'compiled_only_not_executed_on_android', 'abi': args.abi, 'minimum_api': 29,
              'backend': 'Exact LiteRT 1.4.0 C API with default options and no setter calls, matching fast-tflite 3.0.1 delegates=[] creation.',
              'aar': {'path': str(args.aar.resolve()), 'sha256': sha(args.aar)},
              'ndk': str(args.ndk.resolve()), 'ndk_source_properties_sha256': sha(args.ndk/'source.properties'),
              'compiler_version': subprocess.check_output([str(tools/'clang.exe'), '--version'], text=True),
              'command': command, 'elf_command': readelf,
              'sources': [{'path': str(p.resolve()), 'sha256': sha(p)} for p in source_paths],
              'files': [{'path': p.relative_to(stage).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in files],
              'limits': ['No interpreter invocation, operator execution or tensor parity has been established yet.',
                         'Standalone C API omits Nitro/JS bridge, worklet, loader, camera and main APK lifecycle.',
                         'No Java Interpreter default-options equivalence is assumed.']}
    (stage/'compile-report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'stage': str(stage), 'runner_sha256': sha(stage/'word-interpreter'),
                      'library_sha256': sha(library), 'report_sha256': sha(stage/'compile-report.json')}, indent=2))


if __name__ == '__main__':
    main()
