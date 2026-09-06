"""Bounded adb controls for the isolated camera harness (no model)."""
from pathlib import Path
import argparse, json, re, subprocess, time, xml.etree.ElementTree as ET
ADB='C:/Users/rifqi/AppData/Local/Android/Sdk/platform-tools/adb.exe'
def adb(*args): return subprocess.check_output([ADB,'-s','emulator-5554',*args])
def click(label):
    adb('shell','uiautomator','dump','/sdcard/geometry-window.xml')
    raw=adb('shell','cat','/sdcard/geometry-window.xml')
    tree=ET.fromstring(raw)
    for node in tree.iter('node'):
        if node.get('text','').casefold()==label.casefold():
            n=list(map(int,re.findall(r'\d+',node.get('bounds'))))
            adb('shell','input','tap',str((n[0]+n[2])//2),str((n[1]+n[3])//2))
            return
    raise ValueError(f'Cannot find {label}: '+raw.decode())
parser=argparse.ArgumentParser(); parser.add_argument('--directory', default='camera-capture'); parser.add_argument('action', choices=['click','screenshot','rotate','collect']); parser.add_argument('value',nargs='?',default='')
args=parser.parse_args()
OUT=Path(__file__).parent/args.directory
if args.action=='click': click(args.value)
elif args.action=='screenshot':
    (OUT/(args.value+'.png')).write_bytes(adb('exec-out','screencap','-p'))
elif args.action=='rotate':
    adb('shell','settings','put','system','accelerometer_rotation','0')
    adb('shell','settings','put','system','user_rotation',args.value)
elif args.action=='collect':
    print(adb('logcat','-d','-v','raw','ReactNativeJS:I','*:S').decode(errors='replace'))
