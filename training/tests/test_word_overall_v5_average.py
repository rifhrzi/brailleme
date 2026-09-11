from pathlib import Path
import json
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from word_overall_v5_average import average_models, top_three, inspect_epochs
from word_overall_v5_protocol import sha


class Tiny(torch.nn.Module):
    def __init__(self, value, frozen=7., buffer=2):
        super().__init__()
        self.architecture = dict(kind='fixed-test')
        self.weight = torch.nn.Parameter(torch.tensor([value], dtype=torch.float32))
        self.frozen_bn = torch.nn.Parameter(torch.tensor([frozen], dtype=torch.float32), requires_grad=False)
        self.register_buffer('counter', torch.tensor(buffer, dtype=torch.int64))


class AverageTests(unittest.TestCase):
    def test_epoch_ranking_recounts_truth_and_binds_checkpoint_bytes(self):
        rows = [dict(source=f'{group}/{i}', symbol_group=group, expected='a' if i < 4 else 'ab')
                for group in ('capital_single', 'capital_word', 'lowercase', 'numbers', 'punctuation') for i in range(100)]
        rows += [dict(source=f'invalid/{i}', symbol_group='rejection', expected='') for i in range(40)]
        decisions = [dict(source=row['source'], group=row['symbol_group'], expected=row['expected'], predicted=row['expected']) for row in rows]
        overall = dict(correct=500, total=500, exact_accuracy=1., accepted=500,
                       accepted_precision=1., invalid_false_accepts=0, lowest_group_exact=1.)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run/'selection.json').write_text(json.dumps(dict(epochs_completed=3)))
            history = []
            for epoch in range(1, 4):
                checkpoint = run/f'epoch{epoch:02d}.keras'
                checkpoint.write_bytes(str(epoch).encode())
                item = dict(epoch=epoch, overall=overall, checkpoint_sha256=sha(checkpoint))
                history.append(item)
                (run/f'epoch{epoch:02d}-validation.json').write_text(json.dumps(dict(item, decisions=decisions)))
            (run/'history.json').write_text(json.dumps(history))
            entries, bindings = inspect_epochs(run, rows)
            self.assertEqual([row['rank'] for row in entries], [(1., 1., 0)]*3)
            self.assertEqual(len(bindings), 6)
            wrong_rows = [dict(row) for row in rows]
            wrong_rows[0]['expected'] = 'different'
            with self.assertRaisesRegex(ValueError, 'identities/truth'):
                inspect_epochs(run, wrong_rows)
            (run/'epoch02.keras').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                inspect_epochs(run, rows)

    def test_fixed_top_three_excludes_baseline_and_breaks_ties_by_earlier_epoch(self):
        entries = [dict(epoch=0, rank=(1., 1., 0))]
        entries += [dict(epoch=epoch, rank=rank) for epoch, rank in
            [(1, (.7, .8, 0)), (2, (.7, .8, 0)), (3, (.7, .8, -1)), (4, (.7, .7, 0)), (5, (.69, 1., 0))]]
        self.assertEqual([item['epoch'] for item in top_three(entries)], [1, 2, 3])
        with self.assertRaises(ValueError):
            top_three(entries[:3])
        with self.assertRaises(ValueError):
            top_three([entries[1]]*3)

    def test_float64_accumulation_preserves_small_term_and_frozen_values(self):
        models = [Tiny(value) for value in (1e8, 1., -1e8)]
        result, audit = average_models(models)
        self.assertEqual(result.weight.dtype, torch.float32)
        self.assertEqual(result.weight.item(), torch.tensor(1/3, dtype=torch.float32).item())
        self.assertEqual(result.frozen_bn.item(), 7.)
        self.assertEqual(result.counter.item(), 2)
        self.assertEqual(audit['averaged_parameter_tensors'], 1)
        self.assertEqual(audit['frozen_parameter_tensors'], 1)
        self.assertEqual(audit['buffer_tensors'], 1)

    def test_frozen_bn_buffer_architecture_dtype_and_nonfinite_mismatch_fail_before_mutation(self):
        for change in ('bn', 'buffer', 'architecture', 'dtype', 'nonfinite', 'trainability'):
            with self.subTest(change=change):
                models = [Tiny(1.), Tiny(2.), Tiny(3.)]
                if change == 'bn':
                    models[1].frozen_bn.data.add_(1.)
                elif change == 'buffer':
                    models[1].counter.add_(1)
                elif change == 'architecture':
                    models[1].architecture = dict(kind='different')
                elif change == 'dtype':
                    models[1].weight.data = models[1].weight.data.double()
                elif change == 'nonfinite':
                    models[1].weight.data.fill_(float('nan'))
                else:
                    models[1].weight.requires_grad_(False)
                with self.assertRaises(ValueError):
                    average_models(models)
                self.assertEqual(models[0].weight.item(), 1.)

    def test_frozen_signed_zero_mismatch_is_rejected_bitwise(self):
        models = [Tiny(1., frozen=0.), Tiny(2., frozen=-0.), Tiny(3., frozen=0.)]
        with self.assertRaisesRegex(ValueError, 'Frozen parameter'):
            average_models(models)


if __name__ == '__main__':
    unittest.main()
