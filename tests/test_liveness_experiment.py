"""Training readiness, direct candidate policy and evidence-integrity regressions."""

from contextlib import closing
import csv
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sqlite3
import unittest

from PIL import Image

FULL = all(importlib.util.find_spec(p) for p in ('numpy', 'cv2', 'torch', 'torchvision'))
if FULL:
    import numpy as np
    from facial_attendance.liveness.candidate_policy import select_candidates
    from facial_attendance.liveness.data_contract import LivenessSample
    from facial_attendance.liveness.data_loader import LivenessImages
    from facial_attendance.liveness.dataset_files import file_sha256
    from facial_attendance.liveness.experimental_readiness import prepare_experiment
    from facial_attendance.liveness.image_inventory import write_inventory
    from facial_attendance.liveness.near_search import complete_near_search
    from facial_attendance.liveness.quality_audit import scan, exact_duplicates
    from facial_attendance.liveness.quality_review import visual_queue


@unittest.skipUnless(FULL, 'Experimental preparation requires the existing full project environment')
class CandidatePolicyTests(unittest.TestCase):
    def select(self, samples, edges, review=None, crop_edges=(), crop_hashes=None):
        with closing(sqlite3.connect(':memory:')) as db:
            db.execute('CREATE TABLE members(view,signature,path)')
            db.execute('CREATE TABLE edges(view,left_hash,right_hash,hamming)')
            for view in ('full_phash', 'crop_phash'):
                db.executemany('INSERT INTO members VALUES (?,?,?)',
                               [(view, f'{i:016x}', r.relative_path) for i, r in enumerate(samples)])
            for view, pairs in (('full_phash', edges), ('crop_phash', crop_edges)):
                db.executemany('INSERT INTO edges VALUES (?,?,?,?)',
                               [(view, f'{a:016x}', f'{b:016x}', 2) for a, b in pairs])
            fingerprints = {r.relative_path: {'error': '', 'crop_hash': (crop_hashes or {}).get(r.relative_path, r.relative_path)}
                            for r in samples}
            return select_candidates(samples, fingerprints, db, review or {})

    def samples(self):
        return [LivenessSample(p, p, 'train', 0, 0, 'live', p, p) for p in ('a', 'b', 'c')]

    def test_direct_chain_preserves_non_neighbors(self):
        excluded, retained, policy = self.select(self.samples(), [(0, 1), (1, 2)])
        self.assertEqual(retained, {'a', 'c'})
        self.assertEqual(excluded, {'b': ('near_candidate_quarantine', 'a')})
        self.assertEqual(policy['remaining_actionable_pairs'], 0)
        # A component-based rule would incorrectly discard c as well.

    def test_weak_crop_and_source_type_contrast_are_not_duplicate_verdicts(self):
        samples = self.samples()[:2]
        _, retained, policy = self.select(samples, [], crop_edges=[(0, 1)])
        self.assertEqual(retained, {'a', 'b'})
        self.assertEqual(policy['dispositions']['crop_similarity_diagnostic'], 1)
        samples[1] = replace(samples[1], source_type=7, binary_label=1, attack_family='screen')
        self.assertEqual(self.select(samples, [(0, 1)])[1], {'a', 'b'})

    def test_cross_split_full_image_constraint_and_weak_crop_counterpart(self):
        samples = self.samples()[:2]
        samples[1] = replace(samples[1], split='test')
        excluded, retained, _ = self.select(samples, [(0, 1)])
        self.assertEqual(retained, {'b'})
        self.assertEqual(excluded['a'], ('near_candidate_quarantine', 'b'))
        self.assertEqual(samples[0].split, 'train')
        self.assertEqual(self.select(samples, [], crop_edges=[(0, 1)])[1], {'a', 'b'})
        review = {'reviewer': 'fixture', 'pairs': [
            {'path_a': 'a', 'path_b': 'b', 'view': 'crop_phash', 'verdict': 'related_source', 'evidence': 'fixture'}]}
        self.assertEqual(self.select(samples, [], review, crop_edges=[(0, 1)])[1], {'b'})

    def test_reviewed_distinct_and_related_presentations_remain(self):
        samples = self.samples()[:2]
        for verdict in ('distinct', 'related_source'):
            review = {'reviewer': 'SYNTHETIC reviewer', 'pairs': [
                {'path_a': 'a', 'path_b': 'b', 'view': 'full_phash',
                 'verdict': verdict, 'evidence': 'SYNTHETIC review fixture'}]}
            self.assertEqual(self.select(samples, [(0, 1)], review)[1], {'a', 'b'})

    def test_exact_crop_label_conflict_is_quarantined(self):
        samples = self.samples()[:2]
        samples[1] = replace(samples[1], source_type=7, binary_label=1, attack_family='screen')
        excluded, retained, _ = self.select(samples, [], crop_hashes={'a': 'identical', 'b': 'identical'})
        self.assertFalse(retained)
        self.assertEqual({reason for reason, _ in excluded.values()}, {'crop_label_conflict'})

    def test_contradictory_review_is_rejected(self):
        pairs = [{'path_a': a, 'path_b': b, 'view': 'full_phash', 'verdict': verdict, 'evidence': 'fixture'}
                 for a, b, verdict in [('a', 'b', 'duplicate'), ('b', 'c', 'duplicate'), ('a', 'c', 'distinct')]]
        with self.assertRaisesRegex(ValueError, 'Contradictory'):
            self.select(self.samples(), [(0, 1), (1, 2)], {'reviewer': 'fixture', 'pairs': pairs})


@unittest.skipUnless(FULL, 'Experimental preparation requires the existing full project environment')
class ExperimentalReadinessTests(unittest.TestCase):
    def fixture(self, known_overlap=False):
        from test_liveness_data import LivenessDataTests
        case = LivenessDataTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        rng = np.random.default_rng(142)
        for split, labels in case.annotations.items():
            for name in labels:
                Image.fromarray(rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)).save(case.root/name)
        if known_overlap:
            paths = [next(iter(case.annotations[s])) for s in ('train', 'test')]
            with (case.root/'relationships.csv').open('w', newline='') as f:
                w = csv.writer(f); w.writerow(['relative_path', 'source_media_id', 'session_id'])
                w.writerows((p, 'known-shared-video', '') for p in paths)
            case.description['relationships'] = 'relationships.csv'
            case.save_data_description()
        records, _ = case.inventory()
        inventory = case.root/'inventory.csv'; write_inventory(inventory, records)
        scanned = case.root/'scan'; scan(case.root, inventory, case.data_description, scanned)
        exact_duplicates(scanned)
        graph = case.root/'graph'; complete_near_search(scanned, graph)
        return case, inventory, scanned, graph

    def test_unknown_provenance_and_incomplete_visual_review_allow_bound_loading(self):
        case, inventory, scanned, graph = self.fixture()
        visual_queue(inventory, scanned)
        visual = scanned/'visual-review.csv'
        with visual.open(newline='') as f:rows = list(csv.DictReader(f)); fields = list(rows[0])
        # A real rejected annotation must be excluded, while empty reviews
        # remain a limitation instead of a blanket prohibition on training.
        rejected = next(r for r in rows if r['split'] == 'train' and r['source_type'] == '1')
        rejected.update(label_verdict='incorrect', reviewer='fixture', evidence='synthetic observed error')
        with visual.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        output = case.root/'experiment'
        (scanned/'multiview-crop-control.json').write_text(json.dumps({'trials': 28, 'recovered': 26}))
        report = prepare_experiment(case.root, inventory, case.data_description, scanned, graph, output,
                                    visual_review=visual)
        self.assertTrue(report['experimental_ready'])
        self.assertEqual(report['selected_rows'], 20)
        self.assertIn('visual_review_rejection', report['exclusion_counts'])
        self.assertEqual(report['diagnostic_evidence']['multiview-crop-control.json']['readiness_effect'], 'informational')
        receipt = output/'experimental-readiness.json'
        dataset = LivenessImages(case.root, inventory, 'train', audit_file=receipt)
        self.assertEqual(len(dataset), 6)
        self.assertEqual(dataset[0][0].mode, 'RGB')
        self.assertFalse(any(r.relative_path == rejected['relative_path'] for r in dataset.samples))
        image = case.root/dataset.samples[0].relative_path
        image_bytes = image.read_bytes(); image.write_bytes(b'changed image')
        with self.assertRaisesRegex(ValueError, 'changed since audit'):
            dataset[0]
        image.write_bytes(image_bytes)
        source = case.root/'train.json'
        source_bytes = source.read_bytes(); source.write_bytes(source_bytes+b'\n')
        with self.assertRaisesRegex(ValueError, 'source_bindings changed'):
            LivenessImages(case.root, inventory, 'train', audit_file=receipt)
        source.write_bytes(source_bytes)
        original = receipt.read_bytes()
        forged = dict(report); forged['readiness_checks'] = dict(report['readiness_checks'])
        forged['readiness_checks']['near_candidate_policy'] = 'FAIL'
        receipt.write_text(json.dumps(forged))
        with self.assertRaisesRegex(ValueError, 'readiness checks'):
            LivenessImages(case.root, inventory, 'train', audit_file=receipt)
        receipt.write_bytes(original)
        selection = output/'selection.csv'; selection.write_bytes(selection.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError, 'selection changed'):
            LivenessImages(case.root, inventory, 'train', audit_file=receipt)

    def test_known_common_video_overlap_still_blocks_experiment(self):
        # The existing inventory contract rejects known overlaps even before
        # experimental preparation can start; that protection remains intact.
        with self.assertRaisesRegex(ValueError, 'split/family checks'):
            self.fixture(known_overlap=True)


if __name__ == '__main__':
    unittest.main()
