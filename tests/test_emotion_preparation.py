"""Check decisions, split isolation and deterministic preparation on generated data."""
import unittest

from facial_attendance.emotion.prepare import LABELS, make_plan


class EmotionPreparationTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for label in LABELS:
            for i in range(10):
                key = f'{label}-{i}'
                rows.append(dict(source_path=f'train/{label}/{i}.jpg', original_split='train',
                                 original_label=label, file_sha256=key, pixel_sha256=key))
        return rows

    def test_overlap_conflicts_and_exclusions(self):
        rows = self.fixture()
        def add(key, split, label, filename):
            rows.append(dict(source_path=f'{split}/{label}/{filename}.jpg', original_split=split,
                             original_label=label, file_sha256=key, pixel_sha256=key))
        add('conflict','train','sad','c'); add('conflict','test','fear','c')
        add('exclude','train','angry','e'); add('exclude','test','happy','e')
        add('angry-0','test','angry','duplicate')
        choices={k:dict(group_id=i,decision=d,notes='fixture') for i,(k,d) in
                 enumerate((('conflict','neutral'),('exclude','exclude')),1)}
        plan=make_plan(rows,choices)
        conflict=[r for r in plan if r['pixel_sha256']=='conflict']
        self.assertEqual({r['revised_label'] for r in conflict},{'neutral'})
        self.assertEqual([r['output_split'] for r in conflict if r['status']=='retained'],['test'])
        self.assertTrue(all(r['status']=='excluded' for r in plan if r['pixel_sha256']=='exclude'))
        kept=[r for r in plan if r['status']=='retained']
        self.assertEqual(len(kept),len({r['pixel_sha256'] for r in kept}))
        self.assertEqual(plan,make_plan(list(reversed(rows)),choices))

    def test_review_must_cover_conflicts_exactly(self):
        rows=self.fixture()
        rows.append(dict(source_path='test/sad/conflict.jpg',original_split='test',
                         original_label='sad',file_sha256='angry-0',pixel_sha256='angry-0'))
        with self.assertRaises(ValueError):make_plan(rows,{})

    def test_stratified_validation_is_reproducible(self):
        rows=self.fixture()
        plan=make_plan(rows,{})
        for label in LABELS:
            self.assertEqual(sum(r['output_split']=='val' and r['revised_label']==label for r in plan),2)
        self.assertEqual(plan,make_plan(rows,{}))
        self.assertNotEqual([r['output_split'] for r in plan],
                            [r['output_split'] for r in make_plan(rows,{},seed=7)])


if __name__=='__main__':unittest.main()
