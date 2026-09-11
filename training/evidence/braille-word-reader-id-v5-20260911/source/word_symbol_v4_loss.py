"""Opt-in minority-symbol gradients for the versioned experimental training run.

Only fully known six-bit positive centers receive extra weight. Background,
ignored slots, unknown dot targets and all denominator definitions are retained.
Weight 1 and dot scale 1 delegate directly to the original native relative loss.
"""
from __future__ import annotations

import math
import torch
from word_torch_port import word_relative_loss as original_word_relative_loss

MINORITY_MASKS = (32, 60, 50, 2, 38, 22, 18, 6, 36, 54)


def validate_minority_weight(value):
    if isinstance(value, bool):
        raise ValueError('minority_weight must be a finite number >= 1')
    number = float(value)
    if not math.isfinite(number) or number < 1:
        raise ValueError('minority_weight must be a finite number >= 1')
    return number


def validate_dot_loss_scale(value):
    try:
        return validate_minority_weight(value)
    except (TypeError, ValueError) as error:
        raise ValueError('dot_loss_scale must be a finite number >= 1') from error


def minority_centers(truth):
    """Identify complete known minority labels, never model predictions."""
    bits = truth[..., 5:]
    known_binary = ((bits == 0) | (bits == 1)).all(dim=-1, keepdim=True)
    powers = torch.tensor([1, 2, 4, 8, 16, 32], dtype=torch.int64, device=truth.device)
    masks = ((bits >= .5).to(torch.int64) * powers).sum(dim=-1, keepdim=True)
    selected = masks == MINORITY_MASKS[0]
    for mask in MINORITY_MASKS[1:]:
        selected = selected | (masks == mask)
    return (selected & known_binary & (truth[..., :1] >= .999)).to(torch.float32)


def word_relative_loss(truth, prediction, minority_weight=1., dot_loss_scale=1.):
    """Weight minority positives and optionally all known positive dot BCE."""
    weight = validate_minority_weight(minority_weight)
    dot_scale = validate_dot_loss_scale(dot_loss_scale)
    baseline = original_word_relative_loss(truth, prediction)
    if weight == 1. and dot_scale == 1.:
        return baseline
    centers = (truth[..., :1] >= .999).to(torch.float32)
    bits = prediction[..., 5:].clamp(1e-5, 1 - 1e-5)
    known_bits = (truth[..., 5:] >= 0).to(torch.float32)
    labels = truth[..., 5:].clamp_min(0.)
    bit_bce = -(labels * bits.log() + (1 - labels) * (1 - bits).log())
    dot_normalizer = (centers * known_bits).sum().clamp_min(1.)
    if dot_scale != 1.:
        general_dot_loss = (bit_bce * centers * known_bits).sum() / dot_normalizer
        baseline = baseline + 3 * (dot_scale - 1.) * general_dot_loss
    if weight == 1.:
        return baseline
    selected = minority_centers(truth)
    normalizer = centers.sum().clamp_min(1.)
    probability = prediction[..., :1].clamp(1e-5, 1 - 1e-5)
    positive = -selected * (1 - probability).square() * probability.log()
    localization = positive.sum() / normalizer
    delta = torch.cat((truth[..., 1:3] - prediction[..., 1:3],
                       truth[..., 3:5].clamp_min(1e-3).log() - prediction[..., 3:5].clamp_min(1e-3).log()), dim=-1)
    absolute = delta.abs()
    huber = torch.where(absolute < 1., .5 * delta.square(), absolute - .5)
    geometry = (huber * selected).sum() / normalizer
    dot_loss = (bit_bce * selected * known_bits).sum() / dot_normalizer
    return baseline + (weight - 1.) * (localization + 5 * geometry + 3 * dot_loss * dot_scale)
