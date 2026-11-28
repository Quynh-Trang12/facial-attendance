"""Known-answer checks for SYS-06 metrics and input integrity."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from facial_attendance.evaluation import metrics as m

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('evaluation_cli', ROOT/'scripts/evaluate.py')
cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)


class EvaluationTests(unittest.TestCase):
    def test_known_auc_and_ties(self):
        self.assertEqual(m.binary_metrics([0,0,1,1],[0.1,0.4,0.35,0.8])['auc'],0.75)
        self.assertEqual(m.binary_metrics([0,1],[0.5,0.5])['auc'],0.5)
        self.assertEqual(m.binary_metrics([0,1],[1,0])['auc'],0)

    def test_roc_rejects_invalid_inputs(self):
        for y,s in [([0,0],[1,2]),([0,1],[1,float('nan')]),([0,1],[1]),([0,2],[0,1])]:
            with self.subTest(y=y,s=s),self.assertRaises(ValueError):
                m.binary_metrics(y,s)

    def test_threshold_ties_and_validation_only(self):
        selected=m.select_face_threshold([0,1,1],[0.5,0.5,0.8],split='validation')
        self.assertEqual(selected['threshold'],0.8)
        self.assertEqual(selected['far'],0)
        self.assertEqual(selected['tar'],0.5)
        rejected=m.select_face_threshold([0,1],[0.8,0.4],split='validation')
        self.assertTrue(rejected['reject_all'])
        self.assertGreater(rejected['threshold'],0.8)
        for split in ['official','fixture','test']:
            with self.assertRaises(ValueError):
                m.select_face_threshold([0,1],[0,1],split=split)

    def test_classification_macro_f1_and_missing_class(self):
        result=m.classification_metrics([0,1,2],[0,0,2],[0,1,2],['N','U','P'])
        self.assertAlmostEqual(result['accuracy'],2/3)
        self.assertAlmostEqual(result['macro_f1'],5/9)
        self.assertEqual(result['confusion_matrix'],[[1,0,0],[1,0,0],[0,0,1]])
        self.assertEqual(m.classification_metrics([0],[0],[0,1,2],['N','U','P'])['macro_f1'],1/3)
        with self.assertRaises(ValueError):
            m.classification_metrics([0],[3],[0,1,2],['N','U','P'])

    def test_pair_alignment_and_distance_sign(self):
        trials=[('a','b',1),('a','c',0)]
        rows=[dict(left='a',right='b',label=1,cosine=1,euclidean_distance=0),
              dict(left='a',right='c',label=0,cosine=0,euclidean_distance=2)]
        checked=m.pair_scores(trials,rows)
        self.assertEqual(m.binary_metrics([1,0],[r['negative_euclidean'] for r in checked])['auc'],1)
        with self.assertRaises(ValueError): m.pair_scores(trials,rows[::-1])
        with self.assertRaises(ValueError): m.pair_scores(trials,rows[:1])
        rows[0]['euclidean_distance']=-1
        with self.assertRaises(ValueError): m.pair_scores(trials,rows)

    def test_raw_embeddings_not_normalized_distance(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'e.npz'
            np.savez(p,paths=np.array(['a','b','c']),raw_embeddings=np.array([[1,0],[2,0],[0,1]]))
            rows=m.score_embeddings([('a','b',1),('a','c',0)],p)
            self.assertEqual(rows[0]['cosine'],1)
            self.assertEqual(rows[0]['euclidean_distance'],1)
            self.assertAlmostEqual(rows[1]['euclidean_distance'],2**0.5)
            with self.assertRaises(ValueError): m.score_embeddings([('a','missing',0)],p)
            np.savez(p,paths=np.array(['a','b']),raw_embeddings=np.array([[0,0],[1,0]]))
            with self.assertRaises(ValueError): m.score_embeddings([('a','b',0)],p)

    def test_liveness_failures_remain_in_denominators(self):
        rows=[dict(id=str(i),label=y,score=s,status=status,family=f)
              for i,(y,s,status,f) in enumerate([(0,'0.1','live','live'),(0,'0.4','uncertain','live'),
                  (0,'','failure','live'),(1,'0.2','live','print'),(1,'0.9','spoof','screen'),
                  (1,'','failure','video_replay')])]
        r=m.liveness_metrics(rows)
        self.assertEqual(r['count'],6)
        self.assertEqual(r['unscored_count'],2)
        self.assertAlmostEqual(r['spoof_acceptance_rate'],1/3)
        self.assertEqual(r['live_rejection_rate'],0)
        self.assertAlmostEqual(r['genuine_user_blocking_rate'],2/3)
        self.assertEqual(r['failure_count'],2)
        self.assertEqual(r['outcome_table'],[[1,0,1,1],[1,1,0,1]])
        self.assertIsNone(m.liveness_metrics([rows[0]])['spoof_acceptance_rate'])
        rows[1]['score']=''
        with self.assertRaises(ValueError): m.liveness_metrics(rows)

    def test_cli_output_and_official_count(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);trials=root/'trials.txt';scores=root/'scores.csv'
            trials.write_text('a b 1\na c 0\n')
            scores.write_text('left,right,label,cosine,euclidean_distance\na,b,1,1,0\na,c,0,0,2\n')
            with self.assertRaises(ValueError): m.read_trials(trials,official=True)
            args=cli.parser().parse_args(['verification','--trials',str(trials),'--scores',str(scores),
                '--approach','classification','--mode','fixture','--run-id','test','--checkpoint-id','synthetic',
                '--output',str(root/'out')])
            result=cli.evaluate(args)
            self.assertEqual(result['metrics']['cosine']['auc'],1)
            for filename in ['metrics.json','pair_scores.csv','roc_cosine.csv','roc_negative_euclidean.csv','roc.png']:
                self.assertTrue((root/'out'/filename).is_file())
            json.loads((root/'out'/'metrics.json').read_text())
            with self.assertRaises(FileExistsError): cli.evaluate(args)


if __name__=='__main__': unittest.main()
