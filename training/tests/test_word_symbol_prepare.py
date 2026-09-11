"""Regression tests for truthful synthetic symbol source/bit provenance."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from word_symbol_prepare import cell_pool,complete_synthetic_pool


class SymbolCorpusTests(unittest.TestCase):
    def test_split_pool_is_not_populated_from_another_source(self):
        train=[dict(source='train/page1',image='a.png',boxes=[[20,20,40,60]],masks=[1])]
        held=[dict(source='held/page2',image='b.png',boxes=[[20,20,40,60]],masks=[1])]
        training=complete_synthetic_pool(cell_pool(Path('.'),train),{1,32})
        validation=complete_synthetic_pool(cell_pool(Path('.'),held),{1,32})
        self.assertEqual({row['source'] for values in training.values() for row,_ in values},{'train/page1'})
        self.assertEqual({row['source'] for values in validation.values() for row,_ in values},{'held/page2'})

    def test_missing_indicator_is_explicit_transformation_not_relabeling(self):
        original=dict(source='held/page',image='b.png',boxes=[[20,20,40,60]],masks=[1])
        pool=complete_synthetic_pool(cell_pool(Path('.'),[original]),{32})
        transformed,index=pool[32][0]
        self.assertEqual(transformed['synthetic_transform'],'half_turn')
        self.assertEqual(transformed['masks'][index],1)
        self.assertEqual(transformed['original_mask'],1)
        self.assertNotIn('synthetic_transform',original)

    def test_horizontal_number_indicator_to_v_uses_correct_bits(self):
        original=dict(source='held/page',image='b.png',boxes=[[20,20,40,60]],masks=[60])
        pool=complete_synthetic_pool(cell_pool(Path('.'),[original]),{39})
        transformed,_=pool[39][0]
        self.assertEqual(transformed['synthetic_transform'],'horizontal')
        self.assertEqual(transformed['original_mask'],60)

    def test_partial_or_too_small_donors_are_excluded(self):
        rows=[dict(source='held/page',image='b.png',boxes=[[-1,20,40,60],[20,20,24,60],[20,20,40,60]],masks=[1,3,9])]
        pool=cell_pool(Path('.'),rows)
        self.assertEqual(set(pool),{9})


if __name__=='__main__':
    unittest.main()
