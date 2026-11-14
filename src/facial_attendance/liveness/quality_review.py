"""Candidate review and provenance coverage; unknown evidence stays unknown."""

from collections import Counter, defaultdict
from contextlib import closing
import csv
from hashlib import sha256
from itertools import chain, combinations, product
import json
from io import BytesIO
from pathlib import Path
import sqlite3

from PIL import Image, ImageDraw, ImageOps

from facial_attendance.liveness.dataset_files import file_sha256, local_path, read_json
from facial_attendance.liveness.image_inventory import build_inventory, read_inventory, read_experimental_report
from facial_attendance.liveness.quality_audit import source_records, write_json
from facial_attendance.liveness.face_crop import bbox_path, face_box, read_face
from facial_attendance.liveness.image_fingerprints import complete_hamming_pairs, perceptual_hash


def exclusions(output):
    with (output / "selection.csv").open(newline="", encoding="utf-8") as handle:
        return {r["relative_path"] for r in csv.DictReader(handle)}


def candidate_identity(rows):
    """Bind candidate identity while permitting separate review columns to change."""
    digest = sha256()
    for row in rows:
        digest.update((json.dumps([row[key] for key in
                      ("view", "path_a", "path_b", "hamming")]) + "\n").encode("utf-8"))
    return digest.hexdigest()


def review_near(output, *, max_pairs=None):

    if max_pairs is not None and max_pairs < 1:
        raise ValueError("max-pairs must be positive")
    selected_exclusions = exclusions(output)
    candidates, saturated, truncated = 0, 0, False
    views_started = []
    view_results = {}
    with closing(sqlite3.connect(output / "fingerprints.sqlite")) as db:
        with (output / "near-candidates.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["view", "path_a", "path_b", "hamming", "verdict", "reviewer", "evidence"])
            for view in ("full_phash", "crop_phash"):
                views_started.append(view)
                budget = None if max_pairs is None else ((max_pairs + 1) // 2 if view == "full_phash" else max_pairs // 2)
                view_count, view_truncated = 0, False
                groups = defaultdict(list)
                for path, signature in db.execute(f"SELECT path,{view} FROM images WHERE exclusion='' AND {view}!='' ORDER BY path"):
                    if path not in selected_exclusions:
                        groups[signature].append(path)
                pairs, dense = complete_hamming_pairs(groups), 0
                saturated += dense
                same = ((s, s, 0) for s, members in groups.items() if len(members) > 1)
                for left, right, distance in chain(same, pairs):
                    members = combinations(groups[left], 2) if left == right else product(groups[left], groups[right])
                    for a, b in members:
                        if budget is not None and view_count == budget:
                            truncated = True
                            view_truncated = True
                            break
                        writer.writerow([view, *sorted((a, b)), distance, "", "", ""])
                        candidates += 1
                        view_count += 1
                    if view_truncated:
                        break
                view_results[view] = {"candidate_rows": view_count, "truncated": view_truncated,
                                      "saturated_queries": dense}
    with (output / "near-candidates.csv").open(newline="", encoding="utf-8") as handle:
        identity = candidate_identity(csv.DictReader(handle))
    report = {"result": "UNVERIFIED", "candidate_rows": candidates,
              "candidate_identity_sha256": identity,
              "radius": 6, "index": "Exact four-block multi-index; all 16-bit blocks at distance <=1; full 64-bit verification",
              "views_requested": ["full_phash", "crop_phash"],
              "views_started": views_started,
              "view_results": view_results,
              "saturated_queries": saturated, "truncated": truncated,
              "reviewed_candidate_rows": 0, "calibration": "UNVERIFIED: independent real-pair adjudication required",
              "candidates_sha256": file_sha256(output / "near-candidates.csv"),
              "limitation": "Complete within radius 6 only when not truncated. Perceptual candidates do not establish visual independence or shared media/session. All retained candidates require content review."}
    write_json(output / "near.json", report)
    return report


def provenance(root, inventory, description, output):
    """Inspect packaged metadata without inventing session/video identifiers."""
    source = build_inventory(root, description)
    records = read_inventory(inventory)
    selected_exclusions = exclusions(output)
    selected = [r for r in records if not r.exclusion_reason and r.relative_path not in selected_exclusions]
    inspected = []
    for path in sorted((root / "metas").rglob("*")):
        if path.is_file():
            inspected.append({"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)})
    if (root / "README").is_file():
        inspected.append({"path": "README", "sha256": file_sha256(root / "README")})
    known_media = sum(bool(r.source_media_id) for r in selected)
    known_sessions = sum(bool(r.session_id) for r in selected)
    with (output / 'relationship-review.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['relative_path', 'source_media_id', 'session_id',
                         'source_media_status', 'session_status', 'reviewer', 'evidence'])
        writer.writerows((r.relative_path, r.source_media_id, r.session_id,
                          'unknown', 'unknown', '', '') for r in selected)
    condition_counts = Counter()
    if source.assignments is not None:
        for annotations in source.assignments.original_annotations.values():
            condition_counts.update(
                f"illumination={attributes[41]},environment={attributes[42]}"
                for attributes in annotations.values() if attributes[40] != 0
            )
    report = {"result": "UNVERIFIED", "selected_rows": len(selected),
              "source_media_known": known_media, "source_media_unknown": len(selected) - known_media,
              "session_known": known_sessions, "session_unknown": len(selected) - known_sessions,
              "relationships": source.source.get("relationships"), "inspected_metadata": inspected,
              "spoof_condition_counts": dict(sorted(condition_counts.items())),
              "condition_definition_source": "https://arxiv.org/html/2007.12342#S1.T1",
              "condition_limit": "The paper describes session counts by illumination/environment; condition categories are not capture-session identities",
              "required_evidence": "Verified media/session mapping with provenance, or documented not-applicable evidence for individual still-image sources; filenames and image similarity are not relationship IDs"}
    write_json(output / "provenance.json", report)
    return report


def visual_queue(inventory, output):
    """One review row for every selected image; never prefill a verdict."""
    selected_exclusions = exclusions(output)
    records = read_inventory(inventory)
    counters = Counter()
    with closing(sqlite3.connect(output / "fingerprints.sqlite")) as db:
        findings = {path: (retained, error) for path, retained, error in db.execute("SELECT path,retained,error FROM images WHERE retained<1 OR error!=''")}
    with (output / "visual-review.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "sha256", "bbox_sha256", "split", "source_type", "binary_label", "flags", "label_verdict", "box_verdict", "reviewer", "evidence"])
        for row in records:
            if row.exclusion_reason or row.relative_path in selected_exclusions:
                continue
            flags = []
            if ("/live/" in row.relative_path) != (row.binary_label == 0):
                flags.append("folder_json_disagreement")
            retained, error = findings.get(row.relative_path, (1.0, ""))
            if retained < 1:
                flags.append("clipped_box")
            if error:
                flags.append("invalid_box_or_decode")
            counters.update(flags)
            counters["selected_rows"] += 1
            writer.writerow([row.relative_path, row.sha256, row.bbox_sha256, row.split,
                             row.source_type, row.binary_label, ";".join(flags), "", "", "", ""])
    report = {"result": "UNVERIFIED", **counters, "reviewed_rows": 0,
              "review_sha256": file_sha256(output / "visual-review.csv"),
              "scope": "All selected images; inspect original, box overlay, crop and transformed input; label uncertainty requires provenance or exclusion",
              "limitation": "Geometry and annotations do not independently verify visible face content or live/spoof ground truth"}
    write_json(output / "visual.json", report)
    return report


def calibrate_search(output):
    """Measure fixed-radius recovery of controlled variants on real source images.

    This establishes only the measured transformation coverage. It cannot replace
    adjudication of naturally occurring duplicate/distinct image pairs.
    """

    root = Path(read_json(output / 'scan.json')['data_root'])
    rows = []
    counts = Counter()
    excluded = exclusions(output)
    with closing(sqlite3.connect(output / 'fingerprints.sqlite')) as db:
        for path, code in db.execute("SELECT path,source_type FROM images WHERE exclusion='' AND error='' ORDER BY file_hash,path"):
            if path in excluded:
                continue
            if counts[code] < 4:
                rows.append(path)
                counts[code] += 1
            if len(counts) == 7 and all(count >= 4 for count in counts.values()):
                break
    measurements, signatures = [], set()
    for path in rows:
        with Image.open(root / path) as source:
            source = source.convert('RGB')
            original = perceptual_hash(source)
            variants = {}
            for quality in (65, 90):
                encoded = BytesIO()
                source.save(encoded, format='JPEG', quality=quality)
                encoded.seek(0)
                with Image.open(encoded) as decoded:
                    variants[f'jpeg_{quality}'] = perceptual_hash(decoded)
            resized = source.resize((max(1, int(source.width * .8)), max(1, int(source.height * .8))))
            variants['resize_80_percent'] = perceptual_hash(resized)
            w, h = source.size
            variants['crop_5_percent_each_edge'] = perceptual_hash(source.crop((int(w*.05), int(h*.05), max(1, int(w*.95)), max(1, int(h*.95)))))
        signatures.add(original)
        for transform, signature in variants.items():
            signatures.add(signature)
            distance = (int(original, 16) ^ int(signature, 16)).bit_count()
            measurements.append({'relative_path': path, 'transform': transform,
                                 'hamming': distance, 'within_radius_6': distance <= 6})
    pairs, saturated = complete_hamming_pairs(signatures), 0
    found = {tuple(sorted((a, b))) for a, b, _ in pairs}
    expected = {tuple(sorted((a, b))) for a, b in combinations(sorted(signatures), 2)
                if (int(a, 16) ^ int(b, 16)).bit_count() <= 6}
    report = {'result': 'UNVERIFIED', 'source_images': len(rows), 'controlled_variants': len(measurements),
              'controlled_transform_check': 'PASS' if measurements and all(r['within_radius_6'] for r in measurements) else 'FAIL',
              'within_radius_6': sum(row['within_radius_6'] for row in measurements),
              'by_transform': {name: {'trials': sum(r['transform'] == name for r in measurements),
                                     'recovered': sum(r['transform'] == name and r['within_radius_6'] for r in measurements)}
                               for name in sorted({r['transform'] for r in measurements})},
              'index_pairs_in_controlled_set': len(expected), 'index_pairs_recovered': len(expected & found),
              'saturated_queries': saturated, 'measurements': measurements,
              'limitation': 'Controlled transformation trials are not independently adjudicated natural pairs; false-positive rate and natural-pair recall remain unverified'}
    write_json(output / 'calibration.json', report)
    calibrate_crop_views(output, report)
    return report


def calibrate_crop_views(output, calibration):
    """Check the same crop trials using known coordinate shifts in both views."""

    root=Path(read_json(output/'scan.json')['data_root'])
    paths=sorted({row['relative_path'] for row in calibration['measurements']})
    measurements=[]
    for path in paths:
        with Image.open(local_path(root,path)) as source:
            source=source.convert('RGB')
            box,_=face_box(local_path(root,str(bbox_path(Path(path)))),source.size)
            left,top,right,bottom=(int(source.width*.05),int(source.height*.05),
                                   int(source.width*.95),int(source.height*.95))
            variant=source.crop((left,top,right,bottom))
            shifted=(max(0,box[0]-left),max(0,box[1]-top),
                     min(variant.width,box[2]-left),min(variant.height,box[3]-top))
            full=(int(perceptual_hash(source),16)^int(perceptual_hash(variant),16)).bit_count()
            face=(int(perceptual_hash(source.crop(box)),16)^int(perceptual_hash(variant.crop(shifted)),16)).bit_count()
            measurements.append({'relative_path':path,'full_hamming':full,'face_hamming':face,
                                 'matched':min(full,face)<=6})
    report={'scope':'Supplementary 5-percent crop control using known coordinate shifts for the supplied box; this ideal-box control does not establish real detector consistency or natural-pair recall',
            'trials':len(measurements),'recovered':sum(row['matched'] for row in measurements),
            'measurements':measurements}
    write_json(output/'multiview-crop-control.json',report)
    return report


def preview_review(root, output, *, offset=0, count=4):
    """Render source/box, face crop and actual baseline input for bounded review."""

    if offset < 0 or count < 1 or count > 8:
        raise ValueError('Preview needs nonnegative offset and count in 1..8')
    with (output / 'visual-review.csv').open(newline='', encoding='utf-8') as handle:
        rows = sorted(csv.DictReader(handle), key=lambda r: (not bool(r['flags']), r['relative_path']))[offset:offset+count]
    if not rows:
        raise ValueError('No review rows at this offset')
    sheet = Image.new('RGB', (3*224, len(rows)*288), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        path = local_path(root, row['relative_path'])
        with Image.open(path) as image:
            image = image.convert('RGB')
            try:
                box, _ = face_box(local_path(root, str(bbox_path(Path(row['relative_path'])))), image.size)
                marked = image.copy()
                ImageDraw.Draw(marked).rectangle(box, outline='red', width=max(1,image.width//150))
                face = read_face(path)
                sheet.paste(ImageOps.contain(face, (224,224)), (224, i*288))
                sheet.paste(face.resize((224,224), Image.Resampling.BILINEAR), (448,i*288))
            except ValueError as error:
                marked = image
                draw.text((224, i*288), str(error)[:70], fill='red')
            sheet.paste(ImageOps.contain(marked,(224,224)), (0,i*288))
        draw.text((3,i*288+225), row['relative_path'], fill='black')
        draw.text((3,i*288+240), f"type={row['source_type']} label={row['binary_label']} flags={row['flags']}", fill='black')
        draw.text((3,i*288+255), 'Original + box | face crop | 224x224 baseline input', fill='black')
    target=output/f'visual-preview-{offset:06}.png'
    if target.exists():
        raise FileExistsError(target)
    sheet.save(target)
    write_json(target.with_suffix('.json'), {'rows':rows,'preview_sha256':file_sha256(target),
               'scope':'Diagnostic preview only; no visual verdict is generated'})
    return target


def preview_quarantine(root, inventory, output, target, *, anchor_index=0, offset=0, count=4):
    """Inspect direct anchor examples without approving an unseen component."""
    target = Path(target)
    if target.suffix.lower() != '.png' or count < 1 or count > 8 or min(anchor_index, offset) < 0:
        raise ValueError('Quarantine preview needs PNG, nonnegative indices and count in 1..8')
    if target.exists() or target.with_suffix('.json').exists():
        raise FileExistsError(target)
    receipt = output/'experimental-readiness.json'
    read_experimental_report(root, receipt, file_sha256(inventory))
    with (output/'review-anchors.csv').open(newline='', encoding='utf-8') as handle:
        anchors = list(csv.DictReader(handle))
    if anchor_index >= len(anchors):
        raise ValueError('Review anchor index is outside the queue')
    anchor = anchors[anchor_index]
    with (output/'review-priority.csv').open(newline='', encoding='utf-8') as handle:
        members = [r for r in csv.DictReader(handle) if r['reference_path'] == anchor['reference_path']]
    rows = members[offset:offset+count]
    if not rows:
        raise ValueError('No review members at this offset')
    paths = {anchor['reference_path'], *(r['relative_path'] for r in rows)}
    samples = {r.relative_path: r for r in read_inventory(inventory) if r.relative_path in paths}
    sheet = Image.new('RGB', (896, len(rows)*288), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        for side, name in enumerate((anchor['reference_path'], row['relative_path'])):
            sample = samples[name]
            path = local_path(root, name)
            box_file = local_path(root, str(bbox_path(Path(name))))
            if file_sha256(path) != sample.sha256 or file_sha256(box_file) != sample.bbox_sha256:
                raise ValueError(f'Preview input changed since scan: {name}')
            with Image.open(path) as source:
                source = source.convert('RGB')
            box, _ = face_box(box_file, source.size)
            marked = source.copy()
            ImageDraw.Draw(marked).rectangle(box, outline='red', width=max(2, source.width//100))
            for j, im in enumerate((marked, read_face(path))):
                sheet.paste(ImageOps.contain(im, (224,224)), ((side*2+j)*224, i*288))
            draw.text((side*448+2, i*288+225), name, fill='black')
            draw.text((side*448+2, i*288+241), f'{sample.split} type={sample.source_type} label={sample.binary_label}', fill='black')
        draw.text((2, i*288+258), row['reason'], fill='black')
    sheet.save(target)
    write_json(target.with_suffix('.json'), {
        'readiness_sha256': file_sha256(receipt), 'preview_sha256': file_sha256(target),
        'anchor': anchor, 'offset': offset, 'displayed': rows,
        'scope': 'Only displayed originals/boxes/crops. No automatic verdict or approval of unseen anchor members.',
    })
    return target


def _review_near_verdicts(pairs, selected, near, calibration):
    """Resolve reviewed duplicate components; conflicting decisions stay unresolved."""
    near_verified=0
    parent={path:path for path in selected}

    def representative(path):
        while parent[path]!=path:
            parent[path]=parent[parent[path]]
            path=parent[path]
        return path

    for pair in pairs:
        a,b=pair['path_a'],pair['path_b']
        if a not in selected or b not in selected:
            raise ValueError('Near-candidate path absent from scanned selection')
        reviewed=bool(pair['reviewer'].strip()) and bool(pair['evidence'].strip())
        if pair['verdict'] == 'distinct' and reviewed:
            near_verified+=1
        elif pair['verdict']=='duplicate' and reviewed:
            if (selected[a].source_type,selected[a].binary_label)==(selected[b].source_type,selected[b].binary_label):
                parent[representative(a)]=representative(b)
                near_verified+=1
    components=defaultdict(list)
    for path in selected:
        components[representative(path)].append(path)
    near_excluded=set()
    priority={'test':0,'validation':1,'train':2}
    for members in components.values():
        keep=min(members,key=lambda p:(priority[selected[p].split],p))
        near_excluded.update(path for path in members if path!=keep)
    contradictions=sum(pair['verdict']=='distinct' and representative(pair['path_a'])==representative(pair['path_b']) for pair in pairs)
    selected={path:row for path,row in selected.items() if path not in near_excluded}
    near_pass = (len(pairs)==near['candidate_rows'] and near_verified==len(pairs) and not contradictions
                 and not near['truncated'] and not near['saturated_queries']
                 and set(near['views_started'])=={'full_phash','crop_phash'}
                 and bool(calibration.get('controlled_variants'))
                 and calibration.get('within_radius_6')==calibration.get('controlled_variants'))
    return selected, near_verified, near_pass, near_excluded, contradictions


def _review_visual_rows(review_file, before_near, selected):
    """Require unchanged inputs and confirmed decisions for all retained images."""
    confirmed_visual=0
    seen=set()
    with review_file.open(newline='',encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            path=row['relative_path']
            if path in seen or path not in before_near:
                raise ValueError('Unexpected or repeated visual review path')
            seen.add(path)
            sample=before_near[path]
            if any(str(getattr(sample,key))!=row[key] for key in ('sha256','bbox_sha256','split','source_type','binary_label')):
                raise ValueError(f'Visual review refers to changed input: {path}')
            confirmed_visual += (path in selected and row['label_verdict']=='confirmed' and row['box_verdict']=='confirmed'
                                 and bool(row['reviewer'].strip()) and bool(row['evidence'].strip()))
    visual_pass = bool(selected) and set(selected)<=seen and confirmed_visual==len(selected)
    return confirmed_visual, visual_pass


def _review_relationship_rows(provenance_file, before_near, selected):
    """Check evidence coverage and cross-split identities without inventing IDs."""
    # Detailed capture provenance is supplied as a separate, evidence-bearing
    # table. Condition categories and inferred similarity groups cannot fill it.
    media_groups, session_groups=defaultdict(set),defaultdict(set)
    provenance_seen=set()
    confirmed_provenance=0
    if provenance_file.is_file():
        with provenance_file.open(newline='',encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                path=row['relative_path']
                if path in provenance_seen or path not in before_near:
                    raise ValueError('Unexpected or repeated provenance path')
                provenance_seen.add(path)
                if path not in selected:
                    continue
                valid=bool(row['reviewer'].strip()) and bool(row['evidence'].strip())
                for field,groups in (('source_media',media_groups),('session',session_groups)):
                    status=row[f'{field}_status']
                    identity=row[f'{field}_id'].strip()
                    supplied=getattr(selected[path], f'{field}_id')
                    if supplied and (status!='confirmed' or identity!=supplied):
                        valid=False
                    valid &= (status=='confirmed' and bool(identity)) or (status=='not_applicable' and not identity)
                    if status=='confirmed' and identity:
                        groups[identity].add(selected[path].split)
                confirmed_provenance+=bool(valid)
    overlaps=[{'field':field,'id':key,'splits':sorted(splits)}
              for field,groups in (('source_media_id',media_groups),('session_id',session_groups))
              for key,splits in groups.items() if len(splits)>1]
    provenance_pass=(bool(selected) and set(selected)<=provenance_seen
                     and confirmed_provenance==len(selected) and not overlaps)
    return confirmed_provenance, overlaps, provenance_pass


def assess_reviews(root, inventory, description, output, target, *, supporting_evidence=()):
    """Recompute acceptance from bound evidence; summary flags are insufficient.

    Missing provenance or review records keep acceptance incomplete. This does
    not authenticate a reviewer's identity or establish unobserved ground truth.
    """

    if target.exists() or target.with_name(target.stem+'-selection.csv').exists():
        raise FileExistsError(target)
    records = source_records(root, inventory, description)
    scan = read_json(output / 'scan.json')
    if scan['inventory_sha256'] != file_sha256(inventory) or Path(scan['data_root']).resolve() != root.resolve():
        raise ValueError('Quality scan belongs to different inventory/data root')
    for path, digest in scan['source_bindings'].items():
        if file_sha256(Path(path)) != digest:
            raise ValueError(f'Source changed since quality scan: {path}')
    if file_sha256(output/'fingerprints.sqlite') != scan['fingerprints_sha256']:
        raise ValueError('Fingerprint index changed since scan')
    exact, near, visual = (read_json(output/name) for name in ('exact.json','near.json','visual.json'))
    if file_sha256(output/'selection.csv') != exact['selection_sha256']:
        raise ValueError('Exact-duplicate selection changed since audit')
    excluded = exclusions(output)
    selected = {r.relative_path:r for r in records if not r.exclusion_reason and r.relative_path not in excluded}
    before_near = dict(selected)
    checks = {'exact_deduplication': 'PASS' if exact['result']=='PASS' else 'UNVERIFIED: duplicate label conflicts require adjudication'}
    with (output/'near-candidates.csv').open(newline='',encoding='utf-8') as handle:
        pairs=list(csv.DictReader(handle))
    if candidate_identity(pairs) != near['candidate_identity_sha256']:
        raise ValueError('Near-candidate identities changed since search')
    calibration=read_json(output/'calibration.json') if (output/'calibration.json').is_file() else {}
    measurements=calibration.get('measurements', [])
    if measurements:
        if any(type(row['hamming']) is not int or not 0<=row['hamming']<=64 for row in measurements):
            raise ValueError('Invalid controlled-transformation distance')
        recovered=sum(row['hamming']<=6 for row in measurements)
        if len(measurements)!=calibration['controlled_variants'] or recovered!=calibration['within_radius_6']:
            raise ValueError('Calibration counters disagree with measurements')
    combined_file=output/'multiview-crop-control.json'
    combined_misses=0
    if combined_file.is_file():
        combined=read_json(combined_file)
        controls={row['relative_path']:row for row in combined['measurements']}
        crop_trials=[row for row in measurements if row['transform']=='crop_5_percent_each_edge']
        if len(controls)!=len(crop_trials) or set(controls)!={row['relative_path'] for row in crop_trials}:
            raise ValueError('Combined crop controls do not cover the original trials')
        adjusted=[]
        for row in measurements:
            item=dict(row)
            if row['transform']=='crop_5_percent_each_edge':
                control=controls[row['relative_path']]
                if control['full_hamming']!=row['hamming'] or not 0<=control['face_hamming']<=64:
                    raise ValueError('Combined crop control differs from original measurement')
                item['hamming']=min(row['hamming'],control['face_hamming'])
                combined_misses+=item['hamming']>6
            adjusted.append(item)
        calibration={**calibration,'measurements':adjusted,
                     'within_radius_6':sum(row['hamming']<=6 for row in adjusted)}
    selected, near_verified, near_pass, near_excluded, contradictions = _review_near_verdicts(pairs, selected, near, calibration)
    checks['near_duplicate_review']=('FAIL: crop transformations missed in both views' if combined_misses else
                                     'PASS' if near_pass else 'UNVERIFIED: unresolved candidates or search coverage')
    confirmed_visual, visual_pass = _review_visual_rows(output/'visual-review.csv', before_near, selected)
    checks['visual_labels_and_boxes']='PASS' if visual_pass else 'UNVERIFIED: visual labels/boxes not confirmed for every selected image'
    provenance_file=output/'relationship-review.csv'
    confirmed_provenance, overlaps, provenance_pass = _review_relationship_rows(provenance_file, before_near, selected)
    checks['source_relationship_coverage']='PASS' if provenance_pass else 'UNVERIFIED: complete evidence-bearing source relationships unavailable'
    checks['known_common_media']='PASS' if provenance_pass else 'UNVERIFIED: source relationship coverage incomplete'
    checks['loader_provenance']='PASS'
    geometry_errors=0
    with closing(sqlite3.connect(output/'fingerprints.sqlite')) as db:
        for (path,) in db.execute("SELECT path FROM images WHERE error!=''"):
            geometry_errors+=path in selected
    checks['face_input_integrity']='PASS' if not geometry_errors else 'FAIL: selected image/box errors'
    families=Counter((row.split,row.attack_family) for row in selected.values())
    checks['selected_family_coverage']='PASS' if all(families[split,family]>0 for split in ('train','validation','test') for family in ('live','print','screen')) else 'FAIL: cleaned subset lacks a split/attack family'
    accepted=all(value=='PASS' for value in checks.values())
    selection_file=target.with_name(target.stem+'-selection.csv')
    with selection_file.open('x',newline='',encoding='utf-8') as handle:
        writer=csv.writer(handle)
        writer.writerow(['relative_path','exclusion_reason'])
        with (output/'selection.csv').open(newline='',encoding='utf-8') as original:
            writer.writerows((r['relative_path'],r['exclusion_reason']) for r in csv.DictReader(original))
        writer.writerows((path,'confirmed_near_duplicate') for path in sorted(near_excluded))
    bound_files={str((output/name).resolve()):file_sha256(output/name)
                 for name in ('scan.json','exact.json','near.json','visual.json','selection.csv','near-candidates.csv','visual-review.csv','fingerprints.sqlite')}
    if provenance_file.is_file():
        bound_files[str(provenance_file.resolve())]=file_sha256(provenance_file)
    if (output/'calibration.json').is_file():
        bound_files[str((output/'calibration.json').resolve())]=file_sha256(output/'calibration.json')
    if combined_file.is_file():
        bound_files[str(combined_file.resolve())]=file_sha256(combined_file)
    for evidence in supporting_evidence:
        evidence = Path(evidence).resolve(strict=True)
        bound_files[str(evidence)] = file_sha256(evidence)
    failed=any(value.startswith('FAIL') for value in checks.values())
    report={'result':'PASS' if accepted else 'FAIL' if failed else 'INCOMPLETE','accepted':accepted,'requirements':checks,
            'inventory_sha256':file_sha256(inventory),'data_root':str(root.resolve()),
            'source_bindings':scan['source_bindings'],'evidence_bindings':bound_files,
            'selection_file':str(selection_file.resolve()),'selection_sha256':file_sha256(selection_file),
            'confirmed_near_exclusions':len(near_excluded),
            'contradictory_near_verdicts':contradictions,
            'controlled_transform_recovery':{'recovered':calibration.get('within_radius_6',0),
                                              'trials':calibration.get('controlled_variants',0),
                                              'combined_crop_misses':combined_misses,
                                              'scope':'Whole-image controls; crop trials use both views only when known-coordinate controls are supplied'},
            'selected_rows':len(selected),'reviewed_visual_rows':confirmed_visual,
            'cleaned_subset_counts':{split:sum(row.split==split for row in selected.values()) for split in ('train','validation','test')},
            'reviewed_near_rows':near_verified,'confirmed_provenance_rows':confirmed_provenance,
            'relationship_overlaps':overlaps,'selected_input_errors':geometry_errors,
            'implementation_sha256':{str(p.resolve()):file_sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    write_json(target,report)
    return report
