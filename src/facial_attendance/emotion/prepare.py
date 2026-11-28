"""Build a deduplicated FER2013 dataset from recorded human label decisions."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import uuid

from PIL import Image

LABELS = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inspect(path, root):
    relative = path.relative_to(root)
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        if image.mode != 'L' or image.size != (48, 48):
            raise ValueError(f'Expected 48x48 grayscale FER2013 image: {relative}')
        # Same definition as the EDA notebook and manual review group hashes.
        content = json.dumps([image.mode, image.size], separators=(',', ':')).encode() + image.tobytes()
    return dict(source_path=relative.as_posix(), original_split=relative.parts[0],
                original_label=relative.parts[1], file_sha256=sha(raw), pixel_sha256=sha(content))


def load_decisions(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    decisions = {}
    for row in rows:
        key = row['group_hash']
        if key in decisions or row['decision'] not in (*LABELS, 'exclude'):
            raise ValueError('Duplicate hash or unfinished/invalid decision in review CSV.')
        decisions[key] = row
    return decisions


def make_plan(records, decisions, seed=42, validation_fraction=.2):
    if not 0 < validation_fraction < 1:
        raise ValueError('Validation fraction must be between 0 and 1.')
    groups = defaultdict(list)
    for row in records:
        groups[row['pixel_sha256']].append(row)
    conflicts = {key for key, group in groups.items() if len({r['original_label'] for r in group}) > 1}
    if conflicts != set(decisions):
        raise ValueError(f'Review hashes differ from conflicting groups: missing={len(conflicts-set(decisions))}, extra={len(set(decisions)-conflicts)}')
    manifest, retained = [], []
    for key, group in sorted(groups.items()):
        review = decisions.get(key)
        label = review['decision'] if review else group[0]['original_label']
        test = [row for row in group if row['original_split'] == 'test']
        representative = min(test or group, key=lambda row: row['source_path'])
        for original in sorted(group, key=lambda row: row['source_path']):
            row = dict(original, revised_label='' if label == 'exclude' else label,
                       output_split='', output_path='', status='excluded', reason='',
                       review_group_id=review['group_id'] if review else '',
                       review_reason=review['notes'] if review else '')
            if label == 'exclude':
                row['reason'] = 'manual_exclusion'
            elif original is not representative:
                row['reason'] = ('train_test_overlap' if test and row['original_split'] == 'train'
                                 else 'duplicate_copy')
            else:
                row.update(status='retained', output_split='test' if test else 'train',
                           reason='manual_label_change' if label != row['original_label'] else 'kept')
                retained.append(row)
            manifest.append(row)
    # One image per pixel group remains: group separation is guaranteed before splitting.
    for label in LABELS:
        candidates = [r for r in retained if r['output_split'] == 'train' and r['revised_label'] == label]
        if len(candidates) < 2:
            raise ValueError(f'Need at least two eligible training images for {label}.')
        candidates.sort(key=lambda r: sha(f"{seed}|{label}|{r['pixel_sha256']}".encode()))
        # Round 20% separately per class; keep every class in both subsets.
        n_validation = max(1, min(len(candidates)-1, int(len(candidates)*validation_fraction + .5)))
        for row in candidates[:n_validation]:
            row['output_split'] = 'val'
    for row in retained:
        extension = Path(row['source_path']).suffix.lower()
        row['output_path'] = f"{row['output_split']}/{row['revised_label']}/{row['pixel_sha256']}{extension}"
    assert len({r['pixel_sha256'] for r in retained}) == len(retained)
    assert len({r['output_path'] for r in retained}) == len(retained)
    return manifest


def build_dataset(source, output, review, *, seed=42, validation_fraction=.2, workers=8, config=None):
    source, output, review = source.resolve(), output.resolve(), review.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Source and output directories must be separate.')
    files = []
    for split in ('train', 'test'):
        folder = source / split
        if {p.name for p in folder.iterdir() if p.is_dir()} != set(LABELS):
            raise ValueError(f'Unexpected class folders in {folder}')
        for label in LABELS:
            for path in sorted((folder / label).rglob('*')):
                if not path.is_file():
                    continue
                if not path.resolve().is_relative_to((folder / label).resolve()):
                    raise ValueError('Source image escapes its class directory.')
                if path.suffix.lower() not in EXTENSIONS:
                    raise ValueError(f'Unexpected non-image file: {path}')
                files.append(path)
    print(f'Auditing {len(files):,} source images...', flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, record in enumerate(pool.map(lambda p: inspect(p, source), files), 1):
            records.append(record)
            if i % 5000 == 0:
                print(f'  {i:,}/{len(files):,}', flush=True)
    decisions = load_decisions(review)
    manifest = make_plan(records, decisions, seed, validation_fraction)
    retained = [r for r in manifest if r['status'] == 'retained']
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + '.building-' + uuid.uuid4().hex[:8])
    staging.mkdir()
    for split in ('train', 'val', 'test'):
        for label in LABELS:
            (staging / split / label).mkdir(parents=True)
    print(f'Copying and verifying {len(retained):,} retained images...', flush=True)
    def copy_and_verify(row):
        raw = (source / row['source_path']).read_bytes()
        if sha(raw) != row['file_sha256']:
            raise ValueError('A source file changed during preparation.')
        dest = staging / row['output_path']
        dest.write_bytes(raw)
        if sha(dest.read_bytes()) != row['file_sha256']:
            raise ValueError('Output copy failed verification.')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(copy_and_verify, retained):
            pass
    with (staging / 'manifest.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader(); writer.writerows(manifest)
    shutil.copyfile(review, staging / 'review_decisions.csv')
    (staging / 'label_mapping.json').write_text(json.dumps(dict(enumerate(LABELS)), indent=2), encoding='utf-8')
    counts = {split: {label: sum(r['output_split']==split and r['revised_label']==label for r in retained)
                      for label in LABELS} for split in ('train', 'val', 'test')}
    changed = [r for r in manifest if r['revised_label'] and r['original_label'] != r['revised_label']]
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), seed=seed,
                   validation_fraction=validation_fraction,
                   split_method='Per-class SHA256(seed|label|pixel_hash) rank; round(n*fraction) validation images.',
                   source_images=len(manifest), retained_images=len(retained), counts=counts,
                   excluded_by_reason=dict(Counter(r['reason'] for r in manifest if r['status']=='excluded')),
                   manual_exclusion_groups=sum(r['decision']=='exclude' for r in decisions.values()),
                   relabeled_source_images=len(changed),
                   relabeled_source_test_images=sum(r['original_split']=='test' for r in changed),
                   relabeled_retained_test_images=sum(r['original_split']=='test' and r['status']=='retained' for r in changed),
                   review_sha256=sha(review.read_bytes()),
                   source_manifest_sha256=sha('\n'.join(sorted(r['source_path']+'|'+r['file_sha256'] for r in records)).encode()),
                   manifest_sha256=sha((staging/'manifest.csv').read_bytes()),
                   implementation_sha256=sha(Path(__file__).read_bytes()),
                   config_sha256=sha(config.read_bytes()) if config else None,
                   validation='Every output copy verified against source SHA256; one retained image per pixel hash; no pixel overlap between splits.',
                   evaluation_note='Revised FER2013 test set: manually changed labels and removed images; not directly comparable to the original benchmark.',
                   limitations='No identity or near-duplicate audit. Manual labels reflect one reviewer; no independent agreement measured.')
    (staging/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (staging/'README.md').write_text('''# Revised FER2013 dataset

Original downloaded files are unchanged. This dataset applies recorded human review decisions,
excludes ambiguous groups, retains one image per identical decoded-pixel group, and gives test
membership priority over training membership. Remaining training images are split 80/20 by class
using seed 42. See summary.json for measured counts and the exact deterministic split method.

The test set has revised labels and fewer images. Report it as a revised FER2013 test set;
do not present results as directly comparable to the original benchmark.

manifest.csv records every original image, label, decision, exclusion reason and output location.
review_decisions.csv records the manual choices, keyed by decoded-pixel hash. Manual review
is subjective, but applying these fixed decisions to unchanged input files is reproducible.
To change a decision, edit a separate decision CSV and build into a new output directory.
label_mapping.json uses source model IDs, not alphabetical folder order.

Source: https://www.kaggle.com/datasets/msambare/fer2013
Distributor's listed license: Database: Open Database, Contents: Database Contents.
No identity or near-duplicate checks were performed.
''', encoding='utf-8')
    staging.rename(output)
    print(json.dumps(summary, indent=2), flush=True)
    return summary
