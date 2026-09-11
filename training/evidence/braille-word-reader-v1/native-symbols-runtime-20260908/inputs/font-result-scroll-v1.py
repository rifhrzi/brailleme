"""Additional bounded UI observation after the existing 200% font capture."""
import importlib.util
from pathlib import Path

path = Path(__file__).with_name('runtime-actions.py')
spec = importlib.util.spec_from_file_location('runtime_actions', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original_cycle = module.Actions.cycle

def cycle_with_result_scroll(self, index):
    original_cycle(self, index)
    root = self.snapshot('large-font-result-before-scroll')
    for attempt in range(4):
        retry = module.button(root, 'Pindai ulang')
        if retry:
            self.log('large_font_rescan_reachable', bounds=retry['bounds'], scrolls=attempt)
            self.tap(retry, 'Pindai ulang')
            self.pause(2)
            self.snapshot('large-font-after-rescan')
            return
        if not module.find_visible(root, 'text', 'Pindai ulang'):
            self.log('large_font_result_not_observed', interpretation='No result inferred.')
            return
        if attempt == 3 or not self.scroll(root, f'large-font-result-scroll-{attempt}'):
            self.log('large_font_rescan_unconfirmed', interpretation='No accessibility pass inferred.')
            return
        root = self.snapshot(f'large-font-scrolled-result-{attempt}')

module.Actions.cycle = cycle_with_result_scroll
if __name__ == '__main__':
    raise SystemExit(module.main())
