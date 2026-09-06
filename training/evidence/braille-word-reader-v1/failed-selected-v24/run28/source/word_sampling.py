"""Deterministic source-family balancing that retains every training record."""
from __future__ import annotations

import hashlib
import math
from pathlib import PurePosixPath

import numpy as np


class BalancedFamilyOrder:
    def __init__(self, records, seed):
        self.seed = int(seed)
        self.negative_indices = np.asarray([i for i, row in enumerate(records) if not row['boxes']], dtype=np.int64)
        families = {}
        for index, row in enumerate(records):
            if row['boxes']:
                family = str(PurePosixPath(row['source'].replace('\\', '/')).parent)
                families.setdefault(family, []).append(index)
        if not families:
            raise ValueError('Source balancing requires at least one positive source family')
        self.positive_count = sum(map(len, families.values()))
        quota, remainder = divmod(self.positive_count, len(families))
        self.family_orders = {}
        self.quotas = {}
        for number, (family, indices) in enumerate(sorted(families.items())):
            family_seed = int.from_bytes(hashlib.sha256(f'{seed}:{family}'.encode()).digest()[:8], 'little')
            self.family_orders[family] = np.random.default_rng(family_seed).permutation(np.asarray(indices, np.int64))
            self.quotas[family] = quota+(number < remainder)
        self.record_count = len(records)

    def order(self, epoch):
        if epoch < 0:
            raise ValueError('Sampling epoch must be nonnegative')
        parts = [self.negative_indices]
        for family, records in self.family_orders.items():
            quota = self.quotas[family]
            # Continuous cyclic cursors ensure the large family is eventually
            # fully visited. Small families repeat; no record is deleted.
            positions = (epoch*quota+np.arange(quota)) % len(records)
            parts.append(records[positions])
        result = np.concatenate(parts)
        assert len(result) == self.record_count
        return np.random.default_rng(self.seed+epoch).permutation(result)

    def describe(self):
        return {
            'strategy': 'equal_positive_source_family_patches',
            'epoch_records': self.record_count,
            'positive_records_per_epoch': self.positive_count,
            'negative_records_per_epoch': len(self.negative_indices),
            'negative_fraction_unchanged': len(self.negative_indices)/self.record_count,
            'complete_cycle_epochs': max(math.ceil(len(rows)/self.quotas[family]) for family, rows in self.family_orders.items()),
            'families': {family: {'unique_records': len(rows), 'records_per_epoch': self.quotas[family],
                                 'epochs_to_visit_all': math.ceil(len(rows)/self.quotas[family])}
                         for family, rows in self.family_orders.items()},
            'ordering': 'Fixed seeded permutation within each positive source family, continuous cyclic epoch cursors, then seeded whole-epoch shuffle. Every existing negative record index appears exactly once per epoch, including authorized negative repeats.',
            'scope': 'Balances patch counts, not guaranteed equal numbers of cells or independent physical examples. All source records remain in the manifest.',
        }

    def audit(self, order, cumulative_indices):
        selected = set(map(int, order))
        cumulative_indices.update(selected)
        return {
            'order_int64_sha256': hashlib.sha256(np.asarray(order, dtype='<i8').tobytes()).hexdigest(),
            'records': len(order),
            'unique_record_indices_this_epoch': len(selected),
            'cumulative_unique_record_indices': len(cumulative_indices),
            'families': {family: {'records_this_epoch': int(np.isin(order, rows).sum()),
                                 'unique_records_this_epoch': len(selected.intersection(map(int, rows))),
                                 'cumulative_unique_records': len(cumulative_indices.intersection(map(int, rows))),
                                 'total_unique_records': len(rows)}
                         for family, rows in self.family_orders.items()},
        }
