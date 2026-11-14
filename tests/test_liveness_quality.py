"""Regressions for LIV-03 acceptance, content fingerprints and subset policy."""

from dataclasses import replace
from contextlib import closing
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from PIL import Image

from facial_attendance.liveness.audit_report import AuditReport, write_summary
from facial_attendance.liveness.face_crop import read_face
from facial_attendance.liveness.data_loader import LivenessImages
from facial_attendance.liveness.data_contract import QUALITY_REQUIREMENTS
from facial_attendance.liveness.dataset_files import file_sha256
from facial_attendance.liveness.image_inventory import write_inventory, read_quality_report

FULL_ENVIRONMENT = all(importlib.util.find_spec(name) for name in ("numpy", "cv2"))
if FULL_ENVIRONMENT:
    from facial_attendance.liveness.image_fingerprints import pixel_sha256, perceptual_hash, near_candidates, complete_hamming_pairs
    from facial_attendance.liveness.quality_audit import exact_duplicates, scan
    from facial_attendance.liveness.quality_review import (
        review_near, provenance, visual_queue, calibrate_search, assess_reviews,
    )


class AcceptanceTests(unittest.TestCase):
    def test_public_loader_binds_inventory_sources_and_boxes(self):
        from test_liveness_data import LivenessDataTests
        case = LivenessDataTests()
        case.setUp()
        try:
            records, _ = case.inventory()
            inventory = case.root / 'inventory.csv'
            write_inventory(inventory, records)
            with self.assertRaisesRegex(ValueError, 'audit report required'):
                LivenessImages(case.root, inventory, 'train')
            # Controlled receipt tests the binding contract, not visual truth
            # or acceptance of any real dataset.
            proof = case.root / 'synthetic-proof.txt'
            proof.write_text('SYNTHETIC fixture contract only')
            selection = case.root / 'selection.csv'
            selection.write_text('relative_path,exclusion_reason\n')
            quality = case.root / 'quality.json'
            quality.write_text(json.dumps({
                'data_root':str(case.root), 'inventory_sha256':file_sha256(inventory),
                'requirements':{key:'PASS' for key in QUALITY_REQUIREMENTS},
                'source_bindings':{str(case.data_description):file_sha256(case.data_description),
                                   str(case.root/'train.json'):file_sha256(case.root/'train.json')},
                'evidence_bindings':{str(proof):file_sha256(proof)},
                'selection_file':str(selection), 'selection_sha256':file_sha256(selection),
                'selected_rows':21, 'result':'PASS','accepted':True,
                'implementation_sha256':{str(proof):file_sha256(proof)},
                'cleaned_subset_counts':{'train':7,'validation':7,'test':7},
            }))
            report=AuditReport.start(case.root,224,1)
            report.audit={'known_common_media_groups':0}
            report.inventory_sha256=file_sha256(inventory)
            report.complete(quality)
            (case.root/'audit.json').write_text(json.dumps(report.to_dict()))
            dataset=LivenessImages(case.root,inventory,'train')
            self.assertEqual(dataset[0][1],0)
            changed=[replace(r,source_type=2) if r.source_type==1 else r for r in records]
            modified=case.root/'modified.csv'
            write_inventory(modified,changed)
            with self.assertRaisesRegex(ValueError,'Inventory changed'):
                LivenessImages(case.root,modified,'train')
            annotation=case.root/'train.json'
            contents=annotation.read_bytes()
            annotation.write_bytes(contents+b'\n')
            with self.assertRaisesRegex(ValueError,'source_bindings changed'):
                LivenessImages(case.root,inventory,'train')
            annotation.write_bytes(contents)
            box=case.root/dataset.samples[0].relative_path
            box.with_name(box.stem+'_BB.txt').write_text('0 0 20 20 1')
            with self.assertRaisesRegex(ValueError,'changed since audit'):
                dataset[0]
        finally:
            case.doCleanups()

    def test_unknown_evidence_prevents_acceptance_even_with_known_group(self):
        for groups in (0, 1):
            report = AuditReport.start(Path('.'), 224, 1)
            report.audit = {"known_common_media_groups": groups}
            report.complete()
            self.assertTrue(report.pipeline_completed)
            self.assertFalse(report.accepted)
            self.assertEqual(report.result, "INCOMPLETE")
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                audit = root / "audit.json"
                audit.write_text(json.dumps(report.to_dict()))
                write_summary(audit, root / "summary.json")
                self.assertFalse(json.loads((root / "summary.json").read_text())["all_task_requirements_verified"])

    def test_extreme_box_rejected_small_clipping_preserved(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'image.png'
            Image.new('RGB', (224, 224)).save(path)
            box = path.with_name('image_BB.txt')
            box.write_text('-10000 -10000 20000 20000 0.99')
            with self.assertRaisesRegex(ValueError, 'over half'):
                read_face(path)
            box.write_text('-10 20 100 80 1')
            self.assertEqual(read_face(path).size, (90, 80))


@unittest.skipUnless(FULL_ENVIRONMENT, "Quality scan requires the existing full project environment")
class QualityTests(unittest.TestCase):
    def test_default_near_review_has_no_neighbor_or_output_cap(self):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            (output / 'selection.csv').write_text('relative_path,exclusion_reason\n')
            with sqlite3.connect(output / 'fingerprints.sqlite') as db:
                db.execute('CREATE TABLE images(path,full_phash,crop_phash,exclusion)')
                db.executemany('INSERT INTO images VALUES (?,?,?,?)',
                               [(f'image-{i:02}', f'{i:016x}', '0000000000000000', '') for i in range(40)])
            report = review_near(output)
            self.assertEqual(report['candidate_rows'], 2 * 40 * 39 // 2)
            self.assertFalse(report['truncated'])
            self.assertEqual(report['saturated_queries'], 0)

    def test_reviewed_duplicate_crosses_subject_ids_but_not_label_semantics(self):
        from types import SimpleNamespace
        from facial_attendance.liveness.quality_review import _review_near_verdicts
        samples = {'train/a': SimpleNamespace(subject_id='9225', split='train', source_type=0, binary_label=0),
                   'test/b': SimpleNamespace(subject_id='7082', split='test', source_type=0, binary_label=0)}
        pair = {'path_a': 'train/a', 'path_b': 'test/b', 'verdict': 'duplicate',
                'reviewer': 'fixture reviewer', 'evidence': 'SYNTHETIC content adjudication'}
        near = {'candidate_rows': 1, 'truncated': False, 'saturated_queries': 0,
                'views_started': ['full_phash', 'crop_phash']}
        calibration = {'controlled_variants': 1, 'within_radius_6': 1}
        selected, _, passed, excluded, _ = _review_near_verdicts([pair], samples, near, calibration)
        self.assertTrue(passed)
        self.assertEqual(excluded, {'train/a'})
        self.assertEqual(set(selected), {'test/b'})
        # Shared portrait content in a genuine image and a screen attack
        # cannot be merged as two interchangeable training examples.
        samples['train/a'] = SimpleNamespace(subject_id='9225', split='train', source_type=7, binary_label=1)
        _, verified, passed, excluded, _ = _review_near_verdicts([pair], samples, near, calibration)
        self.assertFalse(passed)
        self.assertEqual(verified, 0)
        self.assertFalse(excluded)
        _, _, passed, excluded, _ = _review_near_verdicts([{**pair, 'verdict': 'distinct'}], samples, near, calibration)
        self.assertTrue(passed)
        self.assertFalse(excluded)

    def test_complete_search_rejects_selected_input_error(self):
        from facial_attendance.liveness.near_search import complete_near_search
        with tempfile.TemporaryDirectory() as name:
            source = Path(name) / 'source'
            source.mkdir()
            index = source / 'fingerprints.sqlite'
            with sqlite3.connect(index) as db:
                db.execute('CREATE TABLE images(path,split,source_type,label,full_phash,crop_phash,exclusion,error)')
                db.execute('INSERT INTO images VALUES (?,?,?,?,?,?,?,?)',
                           ('bad.png', 'train', 0, 0, '0000000000000000', '', '', 'invalid box'))
            selection = source / 'selection.csv'
            selection.write_text('relative_path,exclusion_reason\n')
            (source / 'exact.json').write_text(json.dumps({'selection_sha256': file_sha256(selection)}))
            (source / 'scan.json').write_text(json.dumps({
                'fingerprints_sha256': file_sha256(index), 'source_bindings': {}, 'inventory_sha256': 'fixture'}))
            target = Path(name) / 'complete'
            with self.assertRaisesRegex(ValueError, 'Selected row has an input error'):
                complete_near_search(source, target)
            self.assertFalse((target / 'near-complete.json').exists())

    def test_complete_search_matches_brute_force_and_dense_buckets(self):
        import random
        rng = random.Random(42)
        values = {rng.getrandbits(64) for _ in range(120)}
        # Dense buckets, boundary distances, and a pair that matches only
        # the last block expose dropping, duplication and partition errors.
        values.update(range(64))
        values.update((0, (1 << 6) - 1, (1 << 7) - 1, (1 << 8) - 1,
                       3 | (3 << 16) | (3 << 32)))
        values = sorted(values)
        signatures = [f'{v:016x}' for v in values]
        for radius in (0, 1, 6, 7):
            expected = {(f'{a:016x}', f'{b:016x}', (a ^ b).bit_count())
                        for i, a in enumerate(values) for b in values[i + 1:]
                        if (a ^ b).bit_count() <= radius}
            actual = list(complete_hamming_pairs(signatures + signatures[:3], radius=radius))
            self.assertEqual(set(actual), expected)
            self.assertEqual(len(actual), len(expected))
        self.assertEqual(list(complete_hamming_pairs([])), [])
        with self.assertRaises(ValueError):
            list(complete_hamming_pairs(signatures, radius=8))
        with self.assertRaises(ValueError):
            list(complete_hamming_pairs(['not-a-hash']))

    def test_pixel_hash_includes_size_but_not_encoding(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            image = Image.new('RGB', (16, 12), (30, 20, 200))
            image.save(root / 'a.png', compress_level=0)
            image.save(root / 'b.png', compress_level=9)
            self.assertNotEqual((root / 'a.png').read_bytes(), (root / 'b.png').read_bytes())
            with Image.open(root / 'a.png') as a, Image.open(root / 'b.png') as b:
                self.assertEqual(pixel_sha256(a), pixel_sha256(b))
            self.assertNotEqual(pixel_sha256(image), pixel_sha256(Image.new('RGB', (12, 16), (30, 20, 200))))

    def test_exact_subset_policy_and_label_conflicts(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with closing(sqlite3.connect(root / 'fingerprints.sqlite')) as db, db:
                db.execute('CREATE TABLE images(path, split, source_type, label, exclusion, file_hash, pixel_hash, crop_hash)')
                db.executemany('INSERT INTO images VALUES (?,?,?,?,?,?,?,?)', [
                    ('train/a', 'train', 0, 0, '', 'a', 'same', 'crop-a'),
                    ('train/b', 'train', 0, 0, '', 'a', 'same', 'crop-a'),
                    ('test/a', 'test', 0, 0, '', 'b', 'same', 'crop-a'),
                    ('train/c', 'train', 1, 1, '', 'c', 'conflict', 'crop-c'),
                    ('train/d', 'train', 0, 0, '', 'd', 'conflict', 'crop-d'),
                    ('validation/z', 'validation', 0, 0, '', 'z', 'unique', 'crop-z'),
                ])
            report = exact_duplicates(root)
            self.assertEqual(report['additional_exclusions'], 4)
            self.assertEqual(report['cleaned_subset_counts'], {'test': 1, 'validation': 1, 'train': 0})
            self.assertEqual(report['result'], 'UNVERIFIED')
            selection = (root / 'selection.csv').read_text()
            self.assertIn('train/a,exact_duplicate', selection)
            self.assertIn('train/c,duplicate_label_conflict', selection)
            self.assertNotIn('test/a,', selection)
            with self.assertRaises(FileExistsError):
                exact_duplicates(root)

    def test_hamming_neighbors_and_distinct_counterpart(self):
        pairs, saturated = near_candidates(['0000000000000000', '0000000000000001', 'ffffffffffffffff'])
        self.assertEqual(pairs, [('0000000000000000', '0000000000000001', 1)])
        self.assertEqual(saturated, 0)
        self.assertEqual(near_candidates([]), ([], 0))

    def test_scan_bound_to_source_and_stages_leave_unknowns_unverified(self):
        from test_liveness_data import LivenessDataTests
        from facial_attendance.liveness.image_inventory import write_inventory
        case = LivenessDataTests()
        case.setUp()
        try:
            records, _ = case.inventory()
            inventory = case.root / 'inventory.csv'
            write_inventory(inventory, records)
            output = case.root / 'quality'
            result = scan(case.root, inventory, case.data_description, output, workers=2)
            self.assertEqual(result['inventory_rows'], len(records))
            exact_duplicates(output)
            from facial_attendance.liveness.near_search import complete_near_search
            complete_dir = case.root / 'complete'
            complete = complete_near_search(output, complete_dir)
            self.assertTrue(complete['search_complete'])
            self.assertEqual(complete['result'], 'UNVERIFIED')
            self.assertFalse('accepted' in complete)
            with sqlite3.connect(complete_dir / 'candidates.sqlite') as graph:
                for view, summary in complete['views'].items():
                    members = graph.execute('SELECT path,signature FROM members WHERE view=?', (view,)).fetchall()
                    brute = sum((int(a[1], 16) ^ int(b[1], 16)).bit_count() <= 6
                                for i, a in enumerate(members) for b in members[i + 1:])
                    self.assertEqual(summary['implied_image_pairs'], brute)
            with self.assertRaises(FileExistsError):
                complete_near_search(output, complete_dir)
            near = review_near(output, max_pairs=10)
            self.assertEqual(near['result'], 'UNVERIFIED')
            self.assertTrue(near['truncated'])
            self.assertEqual(provenance(case.root, inventory, case.data_description, output)['source_media_known'], 0)
            visual = visual_queue(inventory, output)
            self.assertEqual(visual['reviewed_rows'], 0)
            self.assertEqual(visual['result'], 'UNVERIFIED')
            calibration = calibrate_search(output)
            self.assertEqual(calibration['controlled_variants'], 84)
            attachment = case.root / 'review-attachment.txt'
            attachment.write_text('SYNTHETIC review attachment; not real visual evidence')
            assessed = assess_reviews(case.root, inventory, case.data_description, output, output/'quality.json',
                                      supporting_evidence=[attachment])
            self.assertEqual(assessed['result'], 'INCOMPLETE')
            self.assertFalse(assessed['accepted'])
            self.assertEqual(assessed['reviewed_visual_rows'], 0)
            self.assertEqual(assessed['confirmed_provenance_rows'], 0)
            self.assertEqual(read_quality_report(case.root,output/'quality.json',file_sha256(inventory)),assessed)
            attachment.write_text('changed attachment')
            with self.assertRaisesRegex(ValueError, 'evidence_bindings changed'):
                read_quality_report(case.root, output/'quality.json', file_sha256(inventory))
            attachment.write_text('SYNTHETIC review attachment; not real visual evidence')
            failed_calibration = dict(calibration)
            failed_calibration['measurements'] = [dict(row) for row in calibration['measurements']]
            failed_calibration['measurements'][0]['hamming'] = 64
            failed_calibration['measurements'][0]['within_radius_6'] = False
            failed_calibration['within_radius_6'] -= 1
            (output/'calibration.json').write_text(json.dumps(failed_calibration))
            failed = assess_reviews(case.root, inventory, case.data_description, output, output/'missed-transform.json')
            self.assertEqual(failed['result'], 'INCOMPLETE')
            self.assertIn('UNVERIFIED', failed['requirements']['near_duplicate_review'])
            # A status edit is permitted; replacing candidate identities is not.
            with (output/'near-candidates.csv').open('a') as handle:
                handle.write('full_phash,changed,changed,0,distinct,reviewer,evidence\n')
            with self.assertRaisesRegex(ValueError,'identities changed'):
                assess_reviews(case.root, inventory, case.data_description, output, output/'changed-quality.json')
            changed = case.root / 'changed.csv'
            write_inventory(changed, [replace(r, source_type=2) if r.source_type == 1 else r for r in records])
            with self.assertRaisesRegex(ValueError, 'differs from source'):
                scan(case.root, changed, case.data_description, case.root / 'bad', workers=1)
        finally:
            case.doCleanups()


if __name__ == '__main__':
    unittest.main()
