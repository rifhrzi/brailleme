"""Dismiss only the observed System UI boot ANR on the owned emulator."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/word-android-smoke'))
from smoke import AdbSmoke, find_visible

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
args = p.parse_args()
h = AdbSmoke('C:/Users/rifqi/AppData/Local/Android/Sdk/platform-tools/adb.exe', 'emulator-5554', args.output)
h.guard()
assert h.run('emu', 'avd', 'name').splitlines()[0] == 'BrailleMe_API_36'
root, xml = h.dump('before-boot-wait')
assert find_visible(root, 'text', "System UI isn't responding"), 'No matching System UI ANR; no tap sent'
items = find_visible(root, 'text', 'Wait', True)
assert len(items) == 1, 'No unique Wait control'
left, top, right, bottom = items[0]['bounds']
h.log('tap_observed_boot_wait', ui=str(xml), target=items[0])
h.run('shell', 'input', 'tap', str((left+right)//2), str((top+bottom)//2))
print(h.screenshot('after-boot-wait'))
