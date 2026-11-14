"""Prepare a reproducible training subset with explicit evidence limits."""

from collections import Counter
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
import csv
from pathlib import Path
import sqlite3

from facial_attendance.liveness.audit_report import RuntimeInfo
from facial_attendance.liveness.candidate_policy import select_candidates, VIEWS
from facial_attendance.liveness.data_contract import EXPERIMENT_REQUIREMENTS, EXPERIMENT_IMPLEMENTATION_FILES, SPLITS
from facial_attendance.liveness.data_loader import LivenessImages
from facial_attendance.liveness.dataset_audit import audit_splits
from facial_attendance.liveness.dataset_files import file_sha256, read_json
from facial_attendance.liveness.image_inventory import read_selection
from facial_attendance.liveness.quality_audit import source_records, write_json
from facial_attendance.liveness.sample_batches import sample_batch
from facial_attendance.recognition.data_loader import image_transform


def _bindings(paths):
    return {str(Path(p).resolve()): file_sha256(p) for p in paths}


def _check_bindings(bindings):
    for name, digest in bindings.items():
        if file_sha256(Path(name)) != digest:
            raise ValueError(f'Experimental input changed: {name}')


def _visual_rejections(path, samples):
    rejected, seen = set(), set()
    if path is None:
        return rejected
    by_path = {r.relative_path: r for r in samples}
    with Path(path).open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            name = row['relative_path']
            if name not in by_path or name in seen:
                raise ValueError('Unexpected or repeated visual-review path')
            seen.add(name)
            sample = by_path[name]
            if any(str(getattr(sample, key)) != row[key] for key in
                   ('sha256', 'bbox_sha256', 'split', 'source_type', 'binary_label')):
                raise ValueError(f'Visual review refers to changed input: {name}')
            if any(row[key] in {'incorrect', 'invalid', 'rejected'} for key in ('label_verdict', 'box_verdict')):
                if not row['reviewer'].strip() or not row['evidence'].strip():
                    raise ValueError('Visual rejection requires an attributed evidence record')
                rejected.add(name)
    return rejected


def prepare_experiment(root, inventory, description, scan_dir, candidate_dir, output,
                       *, near_reviews=None, visual_review=None, batch_size=8, image_size=224,
                       supporting_evidence=()):
    """Build experimental readiness independently of exhaustive audit acceptance.

    Public annotations remain the label source. Unknown capture provenance,
    limited visual review and measured pHash misses remain limitations rather
    than fabricated PASS results. Known errors and direct candidate constraints
    are handled before constructing a loadable receipt.
    """
    root, inventory, description = (Path(p).resolve() for p in (root, inventory, description))
    scan_dir, candidate_dir, output = (Path(p).resolve() for p in (scan_dir, candidate_dir, output))
    if image_size < 1 or batch_size < 1:
        raise ValueError('image-size and batch-size must be positive')
    if output.exists():
        raise FileExistsError(output)
    scan, exact = read_json(scan_dir/'scan.json'), read_json(scan_dir/'exact.json')
    graph_report = read_json(candidate_dir/'near-complete.json')
    if (scan['inventory_sha256'] != file_sha256(inventory)
            or Path(scan['data_root']).resolve() != root):
        raise ValueError('Fingerprint scan belongs to another inventory/data root')
    _check_bindings(scan['source_bindings'])
    _check_bindings(graph_report['input_bindings'])
    _check_bindings(graph_report['source_bindings'])
    if (graph_report.get('search_complete') is not True or graph_report.get('radius') != 6
            or graph_report.get('inventory_sha256') != scan['inventory_sha256']
            or graph_report['input_bindings'].get(str((scan_dir/'scan.json').resolve())) != file_sha256(scan_dir/'scan.json')
            or graph_report['graph_sha256'] != file_sha256(candidate_dir/'candidates.sqlite')
            or scan['fingerprints_sha256'] != file_sha256(scan_dir/'fingerprints.sqlite')
            or exact['selection_sha256'] != file_sha256(scan_dir/'selection.csv')):
        raise ValueError('Complete candidate graph or exact selection is not bound to this scan')

    records = source_records(root, inventory, description)
    exact_excluded = read_selection(scan_dir/'selection.csv', records)
    samples = [r for r in records if not r.exclusion_reason and r.relative_path not in exact_excluded]
    by_path = {r.relative_path: r for r in samples}
    fingerprints = {}
    with closing(sqlite3.connect((scan_dir/'fingerprints.sqlite').as_uri()+'?mode=ro', uri=True)) as db:
        fields = ('path', 'file_hash', 'pixel_hash', 'crop_hash', 'full_phash', 'crop_phash', 'error')
        for values in db.execute('SELECT '+','.join(fields)+' FROM images'):
            item = dict(zip(fields, values))
            path = item.pop('path')
            if path in by_path:
                if item['file_hash'] != by_path[path].sha256:
                    raise ValueError(f'Fingerprint/inventory image hash differs: {path}')
                fingerprints[path] = item
    if set(fingerprints) != set(by_path):
        raise ValueError('Fingerprint coverage differs from selection')
    review = read_json(near_reviews) if near_reviews else {}
    review_attachments = []
    if review.get('input_bindings'):
        _check_bindings(review['input_bindings'])
        review_attachments.extend(Path(p) for p in review['input_bindings'])
        for attachment in tuple(review_attachments):
            if attachment.suffix == '.json':
                item = read_json(attachment)
                _check_bindings(item.get('input_bindings', {}))
                review_attachments.extend(Path(p) for p in item.get('input_bindings', {}))
                for name, digest in item.get('previews', {}).items():
                    if Path(name).name != name:
                        raise ValueError('Review preview must be local to its sample record')
                    preview = attachment.parent/name
                    _check_bindings({str(preview): digest})
                    review_attachments.append(preview)
    rejected = _visual_rejections(visual_review, samples)
    with closing(sqlite3.connect((candidate_dir/'candidates.sqlite').as_uri()+'?mode=ro', uri=True)) as db:
        for view in VIEWS:
            seen = set()
            for path, split, code, label, signature in db.execute(
                    'SELECT path,split,source_type,label,signature FROM members WHERE view=?', (view,)):
                if path not in by_path or path in seen:
                    raise ValueError('Candidate member coverage differs from selection')
                seen.add(path)
                row = by_path[path]
                if ((split, code, label) != (row.split, row.source_type, row.binary_label)
                        or signature != fingerprints[path][view]):
                    raise ValueError('Candidate members differ from bound fingerprints/labels')
            if seen != set(by_path):
                raise ValueError('Candidate member coverage differs from selection')
        decisions, retained, policy = select_candidates(samples, fingerprints, db, review, rejected)
    for view in VIEWS:
        if policy['candidate_occurrences_by_view'].get(view, 0) != graph_report['views'][view]['implied_image_pairs']:
            raise ValueError('Expanded candidate count differs from complete search receipt')
    selected = [r for r in samples if r.relative_path in retained]
    splits = audit_splits(selected)
    exact_remaining = {field: sum(n > 1 for n in Counter(fingerprints[p][field] for p in retained).values())
                       for field in ('file_hash', 'pixel_hash', 'crop_hash')}
    checks = dict.fromkeys(EXPERIMENT_REQUIREMENTS, 'UNVERIFIED')
    checks.update(source_annotation_mapping='PASS',
                  selected_input_integrity='PASS' if selected and all(not fingerprints[p]['error'] for p in retained) else 'FAIL',
                  partition_integrity='PASS' if not splits['overlaps'] else 'FAIL',
                  selected_family_coverage='PASS' if not splits['missing_families'] else 'FAIL',
                  exact_deduplication='PASS' if not any(exact_remaining.values()) else 'FAIL',
                  near_candidate_policy='PASS' if policy['remaining_actionable_pairs'] == 0 else 'FAIL')
    output.mkdir(parents=True, exist_ok=False)
    selection = output/'selection.csv'
    with selection.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['relative_path', 'exclusion_reason'])
        with (scan_dir/'selection.csv').open(newline='', encoding='utf-8') as old:
            writer.writerows((r['relative_path'], r['exclusion_reason']) for r in csv.DictReader(old))
        writer.writerows((p, reason) for p, (reason, _) in sorted(decisions.items()))
    queue = output/'review-priority.csv'
    with queue.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['relative_path', 'reason', 'reference_path', 'split', 'source_type'])
        entries = sorted(decisions.items(), key=lambda item: (
            item[1][0] != 'near_candidate_quarantine',
            not item[1][1] or by_path[item[0]].split == by_path[item[1][1]].split,
            item[0]))
        writer.writerows((p, reason, witness, by_path[p].split, by_path[p].source_type)
                         for p, (reason, witness) in entries)
    anchors = {}
    for path, (reason, witness) in decisions.items():
        if not witness:
            continue
        entry = anchors.setdefault(witness, Counter())
        entry['excluded_images'] += 1
        entry['cross_split_images'] += by_path[path].split != by_path[witness].split
    anchor_file = output/'review-anchors.csv'
    with anchor_file.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['reference_path', 'excluded_images', 'cross_split_images'])
        writer.writerows((path, counts['excluded_images'], counts['cross_split_images'])
                         for path, counts in sorted(anchors.items(), key=lambda item: (
                             -item[1]['cross_split_images'], -item[1]['excluded_images'], item[0])))
    policy['optional_review_anchors'] = len(anchors)
    policy['review_requirement'] = 'No exhaustive human review is required for readiness. Review direct anchors to consider reinstatement; each exemption needs an explicit pair decision. A preview of some members does not adjudicate unseen members.'
    write_json(output/'candidate-policy.json', policy)
    batches = {}
    if all(value == 'PASS' for key, value in checks.items() if key != 'sample_batch_alignment_and_traceability'):
        for split in SPLITS:
            dataset = LivenessImages._for_audit(root, selected, split, transform=image_transform(image_size))
            batches[split] = asdict(sample_batch(dataset, output, batch_size, image_size))
        checks['sample_batch_alignment_and_traceability'] = 'PASS'
    ready = all(value == 'PASS' for value in checks.values())
    evidence = [scan_dir/name for name in ('scan.json', 'exact.json', 'selection.csv', 'fingerprints.sqlite')]
    evidence += [candidate_dir/'near-complete.json', candidate_dir/'candidates.sqlite', queue, anchor_file, output/'candidate-policy.json']
    evidence += review_attachments + [Path(p) for p in supporting_evidence]
    diagnostic_files = [scan_dir/name for name in ('calibration.json', 'multiview-crop-control.json', 'provenance.json')
                        if (scan_dir/name).is_file()]
    evidence += diagnostic_files
    for path in (near_reviews, visual_review):
        if path:
            evidence.append(Path(path))
    evidence += [output/f'{split}.{suffix}' for split in batches for suffix in ('pt', 'png')]
    report = {
        'report_type': 'experimental_readiness', 'schema_version': 1,
        'recorded_at': datetime.now(timezone.utc).isoformat(),
        'result': 'PASS' if ready else 'FAIL', 'experimental_ready': ready, 'pipeline_completed': True,
        'readiness_checks': checks, 'data_root': str(root), 'inventory_sha256': file_sha256(inventory),
        'source_bindings': {**scan['source_bindings'], str(description): file_sha256(description)},
        'evidence_bindings': _bindings(evidence),
        'implementation_sha256': _bindings(
            [Path(__file__).parent/name for name in EXPERIMENT_IMPLEMENTATION_FILES]
            + [Path(__file__).parent.parent/'recognition/data_loader.py']),
        'selection_file': str(selection), 'selection_sha256': file_sha256(selection),
        'selected_rows': len(selected), 'cleaned_subset_counts': {split: sum(r.split == split for r in selected) for split in SPLITS},
        'exclusion_counts': dict(Counter(reason for reason, _ in decisions.values())),
        'candidate_policy': policy, 'remaining_exact_groups': exact_remaining,
        'family_counts': {split: splits['counts'][split]['families'] for split in SPLITS},
        'provenance_coverage': {'source_media_known': sum(bool(r.source_media_id) for r in selected),
                                'source_media_unknown': sum(not r.source_media_id for r in selected),
                                'session_known': sum(bool(r.session_id) for r in selected),
                                'session_unknown': sum(not r.session_id for r in selected)},
        'diagnostic_evidence': {p.name: {'path': str(p), 'sha256': file_sha256(p),
                                       'readiness_effect': 'informational'} for p in diagnostic_files},
        'batches': batches, 'runtime': asdict(RuntimeInfo.capture(1)),
        'limitations': [
            'Labels use published source annotations unless an evidence-bearing review rejects them; visual review is not exhaustive.',
            'Unknown source media and capture-session IDs remain unknown; supplied IDs are checked for split overlap.',
            'Complete radius-6 search does not establish natural-pair recall or invariance to all crops; measured pHash misses remain diagnostic.',
            'Candidate quarantine can exclude distinct images. Crop-only pHash similarity is diagnostic in every split and is not a duplicate verdict.',
            'These are cleaned experimental subsets; metrics must not be described as the unmodified original benchmark.',
            'The full-scan fingerprints are reused with verified bindings; loading rechecks image/box hashes on access.',
            'Experimental readiness does not establish model effectiveness, real-time integration or full visual/provenance verification.',
        ],
    }
    write_json(output/'experimental-readiness.json', report)
    return report
