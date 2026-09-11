"""Explicit inference-module choices shared by export and final evaluation."""
from importlib import import_module

DEFAULT_RUNTIME = 'word_overall_v5_tta'
RUNTIME_VIEWS = {
    DEFAULT_RUNTIME: ('identity', 'horizontal', 'vertical', 'half_turn'),
    'word_overall_v5_zoom': ('identity', 'horizontal', 'vertical', 'half_turn', 'zoom125', 'zoom150'),
}


def get_runtime(name=DEFAULT_RUNTIME):
    if name not in RUNTIME_VIEWS:
        raise ValueError('Only the reviewed flip or flip-plus-zoom runtime is allowed')
    return import_module(name)


def get_views(name=DEFAULT_RUNTIME):
    if name not in RUNTIME_VIEWS:
        raise ValueError('Unknown inference runtime')
    return RUNTIME_VIEWS[name]
