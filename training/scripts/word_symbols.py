"""Pure Indonesian uncontracted one-token parser, mirrored in TypeScript."""
from __future__ import annotations

from math import isfinite
from numbers import Real

DECODER_VERSION = 'six-dot-id-v3'
SYMBOL_PROTOCOL = 'indonesian-basic-v1'
MAX_CHARACTERS = 12
MAX_CELLS = 24
CAPITAL_MASK = 32
NUMBER_MASK = 60
LATIN_MASKS = (1, 3, 9, 25, 17, 11, 27, 19, 10, 26, 5, 7, 13,
               29, 21, 15, 31, 23, 14, 30, 37, 39, 58, 45, 61, 53)


def parse_word_masks(masks):
    """Keep indicator cells; never guess omitted indicators or emit partial text."""
    def rejected(reason):
        return {'status': 'rejected', 'reason': reason}

    if not len(masks):
        return rejected('invalid_sequence')
    if len(masks) > MAX_CELLS:
        return rejected('too_many_cells')
    symbols, text = [], ''
    numeric = masks[0] == NUMBER_MASK
    capitals = len(masks) > 1 and masks[0] == masks[1] == CAPITAL_MASK
    capital_next = False
    for index, value in enumerate(masks):
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or not 0 <= value <= 63 or value != int(value):
            return rejected('unsupported_pattern')
        mask = int(value)
        if mask == NUMBER_MASK:
            if index != 0:
                return rejected('invalid_sequence')
            symbols.append({'mask': mask, 'role': 'number_indicator', 'output': ''})
            continue
        if mask == CAPITAL_MASK:
            if numeric or (index > 1 if capitals else capital_next):
                return rejected('invalid_sequence')
            if not capitals:
                capital_next = True
            symbols.append({'mask': mask, 'role': 'capital_indicator', 'output': ''})
            continue
        if mask not in LATIN_MASKS:
            return rejected('unsupported_pattern')
        letter = LATIN_MASKS.index(mask)
        if numeric and letter >= 10:
            return rejected('invalid_sequence')
        output = str((letter + 1) % 10) if numeric else chr((65 if capitals or capital_next else 97) + letter)
        symbols.append({'mask': mask, 'role': 'digit' if numeric else 'letter', 'output': output})
        text += output
        capital_next = False
        if len(text) > MAX_CHARACTERS:
            return rejected('too_many_characters')
    if not text or capital_next:
        return rejected('invalid_sequence')
    return {'status': 'success', 'text': text, 'symbols': symbols}
