"""Experimental Indonesian uncontracted token grammar; not the deployed decoder.

Source: World Braille Usage, third edition (2013), Indonesia, printed page 70
(PDF page index 85), referencing Indonesian Braille 2000. Retrieved 2026-09-10:
https://www.perkins.org/wp-content/uploads/2021/07/world-braille-usage-third-edition.pdf

The punctuation table supplies dot patterns, not this application's scope. This
bounded protocol supports one ASCII word or unsigned integer, a final . , ? ! :
or ;, and one balanced parenthesis pair. Alphabetic hyphenated words are allowed.
It preserves literal punctuation; it does not interpret decimals, thousands,
dates, contractions, quotes, spaces, signs, or mixed alphanumerics. Parentheses
share dots 2356; only the complete pair is distinguishable here. Dots 236 also
open a quotation: accept them as a question mark only after a complete token.
Bare question/parenthesis cells are ambiguous and are rejected. Standalone
unambiguous . , ! : ; - are supported. The 12-character limit includes all
printed punctuation; the 24-cell limit includes every indicator.
"""
from __future__ import annotations

from math import isfinite
from numbers import Real
import re

from word_symbols import (
    CAPITAL_MASK, LATIN_MASKS, MAX_CELLS, MAX_CHARACTERS, NUMBER_MASK,
    parse_word_masks as parse_basic_masks,
)

DECODER_VERSION = 'six-dot-id-v4-experimental'
SYMBOL_PROTOCOL = 'indonesian-punctuation-token-v1'
SOURCE_URL = 'https://www.perkins.org/wp-content/uploads/2021/07/world-braille-usage-third-edition.pdf'
SOURCE_VERSION = 'World Braille Usage third edition 2013; Indonesia p70; Indonesian Braille 2000'
PUNCTUATION_MASKS = {'.': 50, ',': 2, '?': 38, '!': 22, ':': 18, ';': 6, '-': 36, '(': 54, ')': 54}
_FINAL = {PUNCTUATION_MASKS[char]: char for char in '.,?!:;'}
_STANDALONE = {PUNCTUATION_MASKS[char]: char for char in '.,!:;-'}
_PAREN = 54
_HYPHEN = 36
_LETTER_MASKS = dict(zip('abcdefghijklmnopqrstuvwxyz', LATIN_MASKS))
_DIGIT_MASKS = dict(zip('1234567890', LATIN_MASKS[:10]))


def _rejected(reason='invalid_sequence'):
    return {'status': 'rejected', 'reason': reason}


def _punctuation(mask, output):
    return {'mask': mask, 'role': 'punctuation', 'output': output}


def parse_word_masks(masks):
    """Decode complete bounded tokens without inferring missing indicators."""
    if not len(masks):
        return _rejected()
    if len(masks) > MAX_CELLS:
        return _rejected('too_many_cells')
    if any(isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value)
           or not 0 <= value <= 63 or value != int(value) for value in masks):
        return _rejected('unsupported_pattern')
    cells = [int(value) for value in masks]
    if len(cells) == 1 and cells[0] in _STANDALONE:
        output = _STANDALONE[cells[0]]
        return {'status': 'success', 'text': output, 'symbols': [_punctuation(cells[0], output)]}
    if len(cells) == 1 and cells[0] in (38, _PAREN):
        return _rejected('ambiguous_symbol')

    before, after, terminal = [], [], False
    if cells[-1] in _FINAL:
        mask = cells.pop()
        after.append(_punctuation(mask, _FINAL[mask]))
        terminal = True
    if _PAREN in cells:
        if len(cells) < 3 or cells.count(_PAREN) != 2 or cells[0] != _PAREN or cells[-1] != _PAREN:
            return _rejected('ambiguous_symbol')
        cells = cells[1:-1]
        before.append(_punctuation(_PAREN, '('))
        after.insert(0, _punctuation(_PAREN, ')'))
        if cells and cells[-1] in _FINAL:
            if terminal:
                return _rejected()
            mask = cells.pop()
            after.insert(0, _punctuation(mask, _FINAL[mask]))
    if not cells:
        return _rejected()

    # A hyphen joins full alphabetic components, never fragments or numbers.
    segments = [[]]
    for mask in cells:
        if mask == _HYPHEN:
            segments.append([])
        else:
            segments[-1].append(mask)
    if any(not segment for segment in segments):
        return _rejected()
    symbols = list(before)
    for index, segment in enumerate(segments):
        result = parse_basic_masks(segment)
        if result['status'] != 'success':
            return result
        if len(segments) > 1 and not result['text'].isascii():
            return _rejected()
        if len(segments) > 1 and not result['text'].isalpha():
            return _rejected()
        if index:
            symbols.append(_punctuation(_HYPHEN, '-'))
        symbols.extend(result['symbols'])
    symbols.extend(after)
    text = ''.join(symbol['output'] for symbol in symbols)
    if len(text) > MAX_CHARACTERS:
        return _rejected('too_many_characters')
    return {'status': 'success', 'text': text, 'symbols': symbols}


def encode_text(text):
    """Encode a supported exact string, raising ValueError outside the protocol."""
    if not isinstance(text, str) or not text:
        raise ValueError('Text must be a nonempty supported token')
    if len(text) > MAX_CHARACTERS:
        raise ValueError('too_many_characters')
    masks = []
    for token in re.findall(r'[A-Za-z]+|[0-9]+|.', text, flags=re.DOTALL):
        if re.fullmatch(r'[A-Za-z]+', token):
            all_capitals = len(token) > 1 and token.isupper()
            if all_capitals:
                masks.extend([CAPITAL_MASK, CAPITAL_MASK])
            for char in token:
                if char.isupper() and not all_capitals:
                    masks.append(CAPITAL_MASK)
                masks.append(_LETTER_MASKS[char.lower()])
        elif token.isascii() and token.isdigit():
            masks.append(NUMBER_MASK)
            masks.extend(_DIGIT_MASKS[char] for char in token)
        elif token in PUNCTUATION_MASKS:
            masks.append(PUNCTUATION_MASKS[token])
        else:
            raise ValueError('unsupported_pattern')
    result = parse_word_masks(masks)
    if result['status'] != 'success':
        raise ValueError(result['reason'])
    if result['text'] != text:
        raise ValueError('ambiguous_symbol')
    return masks
