"""NumPy reference decoder; thresholds must match the deployed manifest."""
from __future__ import annotations

from collections import Counter
import numpy as np
from word_data import WIDTH, HEIGHT, GRID_W, GRID_H, STRIDE
from word_symbols import DECODER_VERSION, MAX_CELLS, parse_word_masks

DEFAULT_THRESHOLDS = dict(proposalScore=.25, acceptScore=.65, dotConfidence=.9, nmsIou=.25, minCellWidth=.008, minCellHeight=.06, edgeMargin=.008, maxRowDeviation=.55, maxGapRatio=1.2, maxOverlapRatio=.5)


def iou(a, b, a_size, b_size):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = a_size[0]*a_size[1]+b_size[0]*b_size[1]-area
    return area/union if union>0 else 0.


def decode(output, thresholds=None, selection=None):
    t = thresholds or DEFAULT_THRESHOLDS
    selection = selection or [0., 0., 1., 1.]
    # Fixture xyxy is transported to the app as x/y/width/height. Preserve the
    # same subtraction/addition order when reconstructing its right/bottom.
    selection = [selection[0],selection[1],selection[0]+(selection[2]-selection[0]),selection[1]+(selection[3]-selection[1])]
    if output.shape != (GRID_H, GRID_W, 11) or not np.isfinite(output).all() or output.min()<0 or output.max()>1:
        return {"text": "", "reason": "malformed", "cells": []}
    # JavaScript reads Float32Array elements into double-precision Numbers.
    # Promote before threshold/geometry arithmetic instead of rounding the
    # decimal manifest threshold down to the array's float32 dtype.
    output = output.astype(np.float64, copy=False)
    candidates = []
    sizes = {}
    for gy, gx in np.argwhere(output[..., 0] >= t["proposalScore"]):
        row = output[gy, gx]
        cx, cy = (gx+row[1])/GRID_W, (gy+row[2])/GRID_H
        x, y = cx-row[3]/2, cy-row[4]/2
        box = [x, y, x+row[3], y+row[4]]
        if box[2]<=selection[0] or box[3]<=selection[1] or box[0]>=selection[2] or box[1]>=selection[3]:
            continue
        candidate={"box": box, "score": float(row[0]), "dots": row[5:].tolist()}
        candidates.append(candidate)
        sizes[id(candidate)]=(row[3],row[4])
    selected = []
    for cell in sorted(candidates, key=lambda cell: -cell["score"]):
        if all(iou(cell["box"], old["box"], sizes[id(cell)], sizes[id(old)]) <= t["nmsIou"] for old in selected):
            selected.append(cell)
    selected.sort(key=lambda cell: cell["box"][0]+sizes[id(cell)][0]/2)
    result = {"text": "", "reason": "", "cells": selected}
    if not selected:
        result["reason"] = "no_grid"
        return result
    if len(selected)>MAX_CELLS:
        result["reason"] = "too_many_cells"
        return result
    heights = [sizes[id(cell)][1] for cell in selected]
    mid_y = [cell["box"][1]+sizes[id(cell)][1]/2 for cell in selected]
    for reason, invalid in (
        ("too_far", any(sizes[id(cell)][0]<t["minCellWidth"] or sizes[id(cell)][1]<t["minCellHeight"] for cell in selected)),
        ("clipped", any(cell["box"][0]<selection[0] or cell["box"][1]<selection[1] or cell["box"][2]>selection[2] or cell["box"][3]>selection[3] for cell in selected)),
        ("clipped", any(cell["box"][0]<t["edgeMargin"] or cell["box"][1]<t["edgeMargin"] or cell["box"][2]>1-t["edgeMargin"] or cell["box"][3]>1-t["edgeMargin"] for cell in selected)),
        ("low_confidence", any(cell["score"]<t["acceptScore"] for cell in selected)),
        ("multiple_rows", max(mid_y)-min(mid_y)>np.median(heights)*t["maxRowDeviation"]),
    ):
        if invalid:
            result["reason"] = reason
            return result
    for left, right in zip(selected, selected[1:]):
        a, b = left["box"], right["box"]
        gap = (b[0]+sizes[id(right)][0]/2-a[0]-sizes[id(left)][0]/2)*WIDTH
        if gap > t["maxGapRatio"]*np.median(heights)*HEIGHT:
            result["reason"] = "gap"
            return result
        overlap = max(0, min(a[2], b[2])-max(a[0], b[0]))
        if gap <= 0 or overlap > t["maxOverlapRatio"]*min(sizes[id(left)][0], sizes[id(right)][0]):
            result["reason"] = "overlapping_cells"
            return result
    masks = []
    for cell in selected:
        dot_confidence = min(max(dot, 1-dot) for dot in cell["dots"])
        if dot_confidence<t["dotConfidence"]:
            result["reason"] = "low_confidence"
            return result
        mask = sum((int(dot>=.5) << bit) for bit, dot in enumerate(cell["dots"]))
        cell["mask"] = mask
        cell["dot_confidence"] = dot_confidence
        masks.append(mask)
    parsed = parse_word_masks(masks)
    if parsed['status'] == 'rejected':
        result['reason'] = parsed['reason']
        return result
    result['text'] = parsed['text']
    for cell, symbol in zip(selected, parsed['symbols']):
        cell.update(role=symbol['role'], output=symbol['output'])
    return result


def summarize(predictions, records, thresholds=None):
    results = [decode(p, thresholds, row.get("selection")) for p, row in zip(predictions, records)]
    accepted = sum(bool(row["text"]) for row in results)
    correct = sum(row["text"]==record["expected"] for row, record in zip(results, records))
    by_length = {}
    for length in range(1, 13):
        chosen = [(row, record) for row, record in zip(results, records) if len(record["expected"])==length]
        if chosen:
            valid = sum(bool(row["text"]) for row, _ in chosen)
            right = sum(row["text"]==record["expected"] for row, record in chosen)
            by_length[str(length)] = dict(total=len(chosen), accepted=valid, correct=right, exact_word_accuracy=right/len(chosen), accepted_accuracy=right/valid if valid else None)
    return dict(total=len(records), accepted=accepted, correct=correct, exact_word_accuracy=correct/len(records) if records else 0., accepted_accuracy=correct/accepted if accepted else None, by_length=by_length, rejection_reasons=dict(Counter(row["reason"] for row in results if not row["text"])))


def prepare_calibration_outputs(predictions, minimum_proposal=.15):
    """Cache only the minimum score filter; preserve every positional proposal.

    All calibration trials must use proposalScore >= minimum_proposal. Geometry,
    dots and malformed rows remain exact. Never apply local-grid suppression.
    """
    prepared=np.asarray(predictions).copy()
    score=prepared[...,0]
    valid=np.isfinite(prepared).all(axis=(1,2,3)) & (prepared.min(axis=(1,2,3))>=0) & (prepared.max(axis=(1,2,3))<=1)
    prepared[...,0]=np.where(valid[:,None,None],np.where(score>=minimum_proposal,score,0),score)
    return prepared


def calibrate(predictions, records):
    predictions=prepare_calibration_outputs(predictions)
    best = None
    for proposal in (.15, .25, .35):
        for accept in (.4, .55, .65, .75):
            if accept < proposal:
                continue
            for dot in (.7, .8, .9, .95):
                thresholds = dict(DEFAULT_THRESHOLDS, proposalScore=proposal, acceptScore=accept, dotConfidence=dot)
                metrics = summarize(predictions, records, thresholds)
                precision = metrics["accepted_accuracy"] or 0.
                eligible = precision >= .95
                rank = (eligible, metrics["exact_word_accuracy"] if eligible else precision, metrics["correct"])
                if best is None or rank > best[0]:
                    best = (rank, thresholds, metrics)
    return best[1], best[2]
