import unittest
from facial_attendance.emotion.prepare_sixclass import LABELS,make_plan


class EmotionSixClassTests(unittest.TestCase):
    def fixture(self):
        return [dict(source_path=f'train/{label}/{i}.jpg',original_split='train',original_label=label,
                     file_sha256=f'{label}{i}',pixel_sha256=f'{label}{i}') for label in LABELS for i in range(10)]

    def add(self,rows,key,split,label,name):
        rows.append(dict(source_path=f'{split}/{label}/{name}.jpg',original_split=split,
                         original_label=label,file_sha256=key,pixel_sha256=key))

    def test_surprise_is_removed_before_conflicts(self):
        rows=self.fixture();self.add(rows,'happy0','test','surprise','s')
        plan=make_plan(rows)
        happy=[r for r in plan if r['pixel_sha256']=='happy0']
        self.assertEqual(sum(r['status']=='retained' for r in happy),1)
        self.assertEqual(next(r for r in happy if r['original_label']=='surprise')['reason'],'surprise_class')

    def test_conflicts_excluded_without_relabeling(self):
        rows=self.fixture();self.add(rows,'sad0','test','angry','c')
        plan=make_plan(rows)
        self.assertTrue(all(r['reason']=='conflicting_labels' for r in plan if r['pixel_sha256']=='sad0'))
        self.assertTrue(all(r['label']==r['original_label'] for r in plan))

    def test_test_priority_and_split_isolation(self):
        rows=self.fixture();self.add(rows,'angry0','test','angry','t')
        plan=make_plan(rows);group=[r for r in plan if r['pixel_sha256']=='angry0']
        self.assertEqual([r['output_split'] for r in group if r['status']=='retained'],['test'])
        kept=[r for r in plan if r['status']=='retained']
        self.assertEqual(len(kept),len({r['pixel_sha256'] for r in kept}))
        self.assertTrue(all(r['model_id']==5 for r in kept if r['label']=='neutral'))

    def test_deterministic_stratification(self):
        rows=self.fixture();plan=make_plan(rows)
        self.assertEqual(plan,make_plan(list(reversed(rows))))
        for label in LABELS:
            self.assertEqual(sum(r['output_split']=='val' and r['label']==label for r in plan),2)
        self.assertNotEqual(plan,make_plan(rows,seed=7))


if __name__=='__main__':unittest.main()
