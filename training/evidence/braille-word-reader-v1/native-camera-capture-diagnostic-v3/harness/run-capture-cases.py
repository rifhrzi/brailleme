"""Eight bounded UI captures; preserves every success/error and never invokes recognition."""
from pathlib import Path
import json, subprocess, time, xml.etree.ElementTree as ET, re
HERE=Path(__file__).parent
OUT=HERE/'camera-capture-v3'
ADB='C:/Users/rifqi/AppData/Local/Android/Sdk/platform-tools/adb.exe'
def adb(*args): return subprocess.check_output([ADB,'-s','emulator-5554',*args])
def events():
    raw=adb('logcat','-d','-v','raw','ReactNativeJS:I','*:S').decode(errors='replace')
    return [json.loads(line.split('WORD_GEOMETRY_DIAGNOSTIC ',1)[1]) for line in raw.splitlines() if 'WORD_GEOMETRY_DIAGNOSTIC ' in line]
def click(label):
    for attempt in range(4):
        adb('shell','uiautomator','dump','/sdcard/geometry-window.xml')
        tree=ET.fromstring(adb('shell','cat','/sdcard/geometry-window.xml'))
        nodes=list(tree.iter('node'))
        candidate=next((n for n in nodes if (n.get('content-desc','').casefold()==label.casefold() or n.get('text','').casefold()==label.casefold()) and n.get('enabled')=='true' and n.get('clickable')=='true'),None)
        if candidate is None:
            candidate=next((n for n in nodes if n.get('text')=='Wait'),None)
            if candidate is None: time.sleep(2);continue
            retry=True
        else: retry=False
        b=list(map(int,re.findall(r'\d+',candidate.get('bounds'))))
        adb('shell','input','tap',str((b[0]+b[2])//2),str((b[1]+b[3])//2))
        if not retry:return
        time.sleep(2)
    raise RuntimeError('Cannot click '+label)
plan=[(0,1),(0,2),(1,1),(1,2),(1,3),(0,3),(2,1),(3,1)]
results=[]
for rotation,case in plan:
    adb('shell','settings','put','system','accelerometer_rotation','0')
    adb('shell','settings','put','system','user_rotation',str(rotation));time.sleep(3)
    click('Case '+str(case));time.sleep(3)
    adb('exec-out','screencap','-p') # ensure rendering has completed before the shutter
    before=sum(e['kind'] in ('mapped','error') for e in events())
    click('Capture')
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        fresh=[e for e in events() if e['kind'] in ('mapped','error')]
        if len(fresh)>before:break
        time.sleep(1)
    else: raise RuntimeError(f'Capture timed out: rotation{rotation} case{case}')
    name=f'rotation{rotation}-case{case}'
    (OUT/(name+'-frozen.png')).write_bytes(adb('exec-out','screencap','-p'))
    results.append({'rotation_setting':rotation,'case':case,'event':fresh[-1]})
    (OUT/'case-control-results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(name,json.dumps(fresh[-1]),flush=True)
