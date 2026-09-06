"""Strict, side-effect-free parser for DSBI recto grid annotations.

Coordinates refer to the supplied deskewed ``+recto.jpg``. The recorded angle
describes the original scan; do not rotate the deskewed image again. Dot order
is left top/middle/bottom, then right top/middle/bottom (bits 0 through 5).

Format: https://github.com/yeluo1994/DSBI/tree/6937eb8973efe5ff337e5c018485cb626303f5c2
Cell margin convention: IlyaOvodov/AngelinaReader at
328b0e0d30353d76ec06cde8560a876da5c46ab7, data_utils/dsbi.py.
This module does not add records to a training manifest.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re


DSBI_REVISION = '6937eb8973efe5ff337e5c018485cb626303f5c2'
DSBI_COORDINATE_SPACE = 'provided_deskewed_recto_image_pixels'
DEFAULT_RECT_MARGIN = .3
_NUMBER = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z')
_INDEX = re.compile(r'[1-9][0-9]*\Z')


@dataclass(frozen=True)
class DSBICell:
    row: int
    col: int
    bits: tuple[int, ...]
    dot_centers: tuple[tuple[float, float], ...]
    annotation_line: int

    @property
    def mask(self) -> int:
        return sum(value << bit for bit, value in enumerate(self.bits))

    @property
    def is_background(self) -> bool:
        return self.mask == 0

    def box(self, rect_margin: float = DEFAULT_RECT_MARGIN) -> tuple[float, float, float, float]:
        """Full 2x3 cell, expanded on every edge by margin * column pitch.

        Keep all six grid positions even for A or a blank annotation. Never
        clamp the source box to an image: a later crop must preserve knowledge
        that the source cell is incomplete.
        """
        if not math.isfinite(rect_margin) or rect_margin < 0:
            raise ValueError('Cell margin must be a finite nonnegative number')
        left, top = self.dot_centers[0]
        right, bottom = self.dot_centers[5]
        extension = rect_margin * (right-left)
        box = left-extension, top-extension, right+extension, bottom+extension
        if not all(math.isfinite(value) for value in box):
            raise ValueError('Expanded cell box must remain finite')
        return box


@dataclass(frozen=True)
class DSBIAnnotation:
    angle_degrees: float | None
    vertical_lines: tuple[float, ...]
    horizontal_lines: tuple[float, ...]
    cells: tuple[DSBICell, ...]

    @property
    def is_empty(self) -> bool:
        """An empty side annotation is explicitly documented by DSBI."""
        return self.angle_degrees is None


def _numbers(line: str, name: str) -> tuple[float, ...]:
    tokens = line.split()
    if not tokens or any(not _NUMBER.fullmatch(token) for token in tokens):
        raise ValueError(f'{name}: expected plain numeric tokens')
    values = tuple(float(token) for token in tokens)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f'{name}: values must be finite')
    return values


def parse_dsbi_annotation(text: str, *, source: str = '<text>') -> DSBIAnnotation:
    """Parse without eval; reject malformed grids, duplicates and unknown bits.

    Explicit mask-zero cells are retained as background. Unlisted row/column
    slots are not silently synthesized as labeled cells.
    """
    text = text.removeprefix('\ufeff')
    if not text.strip():
        return DSBIAnnotation(None, (), (), ())
    lines = text.splitlines()
    if len(lines) < 3:
        raise ValueError(f'{source}: a nonempty annotation needs three header lines')
    angle = _numbers(lines[0], f'{source}: angle')
    if len(angle) != 1:
        raise ValueError(f'{source}: angle must contain one number')
    vertical = _numbers(lines[1], f'{source}: vertical grid')
    horizontal = _numbers(lines[2], f'{source}: horizontal grid')
    if len(vertical) % 2 or len(horizontal) % 3:
        raise ValueError(f'{source}: vertical grid must pair columns; horizontal grid must group three rows')
    for name, values in (('vertical', vertical), ('horizontal', horizontal)):
        if values[0] < 0 or any(right <= left for left, right in zip(values, values[1:])):
            raise ValueError(f'{source}: {name} grid must be nonnegative and strictly increasing')
    cells, occupied_slots = [], set()
    for line_number, line in enumerate(lines[3:], 4):
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) != 8 or not _INDEX.fullmatch(tokens[0]) or not _INDEX.fullmatch(tokens[1]):
            raise ValueError(f'{source}:{line_number}: expected one-based row/column and six bits')
        row, col = int(tokens[0]), int(tokens[1])
        if row > len(horizontal)//3 or col > len(vertical)//2:
            raise ValueError(f'{source}:{line_number}: cell index exceeds its grid')
        if any(value not in ('0', '1') for value in tokens[2:]):
            raise ValueError(f'{source}:{line_number}: each dot must be exactly 0 or 1')
        if (row, col) in occupied_slots:
            raise ValueError(f'{source}:{line_number}: duplicate cell row/column')
        occupied_slots.add((row, col))
        xs = vertical[(col-1)*2:col*2]
        ys = horizontal[(row-1)*3:row*3]
        # Use the actual middle grid line, which need not equal the arithmetic
        # midpoint of the top and bottom lines.
        points = tuple((x, y) for x in xs for y in ys)
        cells.append(DSBICell(row, col, tuple(map(int, tokens[2:])), points, line_number))
    return DSBIAnnotation(angle[0], vertical, horizontal, tuple(cells))


def read_dsbi_annotation(path: Path | str) -> DSBIAnnotation:
    path = Path(path)
    return parse_dsbi_annotation(path.read_text(encoding='utf-8-sig'), source=str(path))


def validate_dsbi_image_bounds(annotation: DSBIAnnotation, width: int, height: int) -> None:
    """Validate center-grid coordinates against an already deskewed image.

    Expanded full-cell boxes may extend beyond its edges; callers must retain
    those coordinates and mark them incomplete rather than clipping labels.
    """
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError('Image dimensions must be positive integers')
    if any(value >= width for value in annotation.vertical_lines) or any(value >= height for value in annotation.horizontal_lines):
        raise ValueError('Annotation grid extends outside the deskewed image')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotation', type=Path)
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    args = parser.parse_args()
    if (args.width is None) != (args.height is None):
        parser.error('--width and --height must be provided together')
    annotation = read_dsbi_annotation(args.annotation)
    if args.width is not None:
        validate_dsbi_image_bounds(annotation, args.width, args.height)
    result = asdict(annotation)
    result['coordinate_space'] = DSBI_COORDINATE_SPACE
    result['applied_rotation_degrees'] = 0
    result['cells'] = [dict(asdict(cell), mask=cell.mask, is_background=cell.is_background, box=cell.box()) for cell in annotation.cells]
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
