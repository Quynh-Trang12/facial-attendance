"""Complete bounded-radius discovery, stored without expanding dense pairs."""

from collections import Counter
from contextlib import closing
import csv
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sqlite3
import time

from facial_attendance.liveness.dataset_files import file_sha256, read_json
from facial_attendance.liveness.image_fingerprints import complete_hamming_pairs
from facial_attendance.liveness.quality_audit import write_json


def complete_near_search(source, target):
    """Reuse verified fingerprints; preserve the prior scan and review records.

    Members plus signature edges represent all image pairs, including equal
    signatures. Neither equality nor an edge is a duplicate verdict. Keeping
    this graph compact avoids quadratic CSV expansion for dense signatures.
    This supplemental evidence does not authorize the training loader.
    """
    source, target = Path(source).resolve(), Path(target).resolve()
    scan = read_json(source / 'scan.json')
    exact = read_json(source / 'exact.json')
    database = source / 'fingerprints.sqlite'
    if file_sha256(database) != scan['fingerprints_sha256']:
        raise ValueError('Fingerprint index changed since scan')
    if file_sha256(source / 'selection.csv') != exact['selection_sha256']:
        raise ValueError('Exact selection changed since scan')
    for path, digest in scan['source_bindings'].items():
        if file_sha256(Path(path)) != digest:
            raise ValueError(f'Source changed since scan: {path}')
    with (source / 'selection.csv').open(newline='', encoding='utf-8') as handle:
        excluded = {r['relative_path'] for r in csv.DictReader(handle)}
    if shutil.disk_usage(target.parent).free < 1024**3:
        raise ValueError('Need at least 1 GiB free for compact candidate graph')
    target.mkdir(exist_ok=False)
    result = {'recorded_at': datetime.now(timezone.utc).isoformat(),
              'result': 'UNVERIFIED', 'search_complete': False, 'radius': 6,
              'source_scan': str(source / 'scan.json'),
              'inventory_sha256': scan['inventory_sha256'],
              'input_bindings': {str(source / name): file_sha256(source / name)
                                 for name in ('scan.json', 'exact.json', 'selection.csv', 'fingerprints.sqlite')},
              'source_bindings': scan['source_bindings'], 'views': {},
              'algorithm': 'Exact four-block multi-index; all 16-bit blocks at Hamming distance <=1; full 64-bit verification; radius 6',
              'limitation': 'Complete only within the fixed pHash radius for full and supplied face crops. Natural-pair recall, crop robustness, duplicate adjudication, source relationships and visual labels remain unverified.'}
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as original, \
            closing(sqlite3.connect(target / 'candidates.sqlite')) as db, db:
        db.execute('CREATE TABLE members(view TEXT, signature TEXT, path TEXT, split TEXT, source_type INTEGER, label INTEGER, PRIMARY KEY(view,path))')
        db.execute('CREATE TABLE edges(view TEXT, left_hash TEXT, right_hash TEXT, hamming INTEGER, PRIMARY KEY(view,left_hash,right_hash))')
        for view in ('full_phash', 'crop_phash'):
            started = time.monotonic()
            rows = []
            for path, split, code, label, signature, error in original.execute(
                    f"SELECT path,split,source_type,label,{view},error FROM images WHERE exclusion='' ORDER BY path"):
                if path in excluded:
                    continue
                if error:
                    raise ValueError(f'Selected row has an input error: {path}')
                rows.append((view, signature, path, split, code, label))
            if any(not row[1] for row in rows):
                raise ValueError('Selected row missing a perceptual fingerprint')
            db.executemany('INSERT INTO members VALUES (?,?,?,?,?,?)', rows)
            counts = Counter(row[1] for row in rows)
            edges, implied = 0, sum(n * (n - 1) // 2 for n in counts.values())
            for a, b, distance in complete_hamming_pairs(counts):
                db.execute('INSERT INTO edges VALUES (?,?,?,?)', (view, a, b, distance))
                edges += 1
                implied += counts[a] * counts[b]
                if edges % 100000 == 0:
                    db.commit()
                    print(f'{view}: {edges} signature edges', flush=True)
            db.commit()
            result['views'][view] = {'selected_rows': len(rows), 'distinct_signatures': len(counts),
                                     'signature_edges': edges, 'equal_signature_groups': sum(n > 1 for n in counts.values()),
                                     'implied_image_pairs': implied, 'seconds': time.monotonic() - started}
            print(f'{view}: complete, {implied} image pairs', flush=True)
        db.execute('CREATE INDEX member_signature ON members(view,signature)')
        db.commit()
    result['search_complete'] = True
    result['graph_sha256'] = file_sha256(target / 'candidates.sqlite')
    result['implementation_sha256'] = {str(p): file_sha256(p) for p in
                                       (Path(__file__).resolve(), Path(__file__).with_name('image_fingerprints.py').resolve())}
    write_json(target / 'near-complete.json', result)
    return result
