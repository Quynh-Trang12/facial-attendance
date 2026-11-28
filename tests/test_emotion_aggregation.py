import unittest
from facial_attendance.emotion.aggregate import aggregate_plan,EMOTION_TO_CLASS,CLASS_TO_ID


class EmotionAggregationTests(unittest.TestCase):
    def row(self,label,split='train',status='retained'):
        return dict(label=label,original_label=label,original_split=split,output_split=split,
                    status=status,reason='kept',pixel_sha256=label+split,
                    output_path=f'{split}/{label}/sample.jpg',model_id=0)

    def test_mapping_and_partition_preservation(self):
        rows=[self.row(label,split) for label in EMOTION_TO_CLASS for split in ('train','val','test')]
        plan=aggregate_plan(rows)
        for source,row in zip(rows,plan):
            self.assertEqual(row['label'],EMOTION_TO_CLASS[source['label']])
            self.assertEqual(row['emotion_label'],source['original_label'])
            self.assertEqual(row['output_split'],source['output_split'])
            self.assertEqual(row['model_id'],CLASS_TO_ID[row['label']])

    def test_excluded_surprise_is_not_assigned_a_target(self):
        row=self.row('surprise',status='excluded');row.update(output_split='',output_path='',reason='surprise_class')
        result=aggregate_plan([row])[0]
        self.assertEqual(result['label'],'');self.assertEqual(result['reason'],'surprise_class')
        self.assertEqual(result['emotion_label'],'surprise')

    def test_surprise_cannot_be_retained(self):
        with self.assertRaises(ValueError):aggregate_plan([self.row('surprise')])


if __name__=='__main__':unittest.main()
