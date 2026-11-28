"""Prepare six FER2013 classes with original labels and exact duplicate removal."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import platform
import uuid

import PIL
from .prepare import EXTENSIONS, LABELS as SOURCE_LABELS, inspect, sha

LABELS = ('angry', 'disgust', 'fear', 'happy', 'sad', 'neutral')
MODEL_IDS = {label: index for index, label in enumerate(LABELS)}


def make_plan(records, *, seed=42, validation_fraction=.2):
    if not 0 < validation_fraction < 1:
        raise ValueError('Validation fraction must be between 0 and 1.')
    groups = defaultdict(list)
    for record in records:
        if record['original_label'] not in SOURCE_LABELS:
            raise ValueError('Unknown source label.')
        if record['original_split'] not in ('train', 'test'):
            raise ValueError('Unknown source split.')
        groups[record['pixel_sha256']].append(record)
    manifest, retained = [], []
    for key, group in sorted(groups.items()):
        eligible = [r for r in group if r['original_label'] in LABELS]
        conflict = len({r['original_label'] for r in eligible}) > 1
        tests = [r for r in eligible if r['original_split'] == 'test']
        selected = min(tests or eligible, key=lambda r:r['source_path']) if eligible else None
        for original in sorted(group, key=lambda r:r['source_path']):
            row = dict(original, label=original['original_label'], model_id='',
                       status='excluded', reason='', output_split='', output_path='')
            if row['original_label'] == 'surprise':
                row['reason'] = 'surprise_class'
            elif conflict:
                row['reason'] = 'conflicting_labels'
            elif original is not selected:
                row['reason'] = 'train_test_overlap' if tests and row['original_split']=='train' else 'duplicate_copy'
            else:
                row.update(status='retained', reason='kept', model_id=MODEL_IDS[row['label']],
                           output_split='test' if tests else 'train')
                retained.append(row)
            manifest.append(row)
    for label in LABELS:
        candidates = [r for r in retained if r['output_split']=='train' and r['label']==label]
        if len(candidates)<2:
            raise ValueError(f'Not enough eligible training images for {label}.')
        candidates.sort(key=lambda r:sha(f"{seed}|{label}|{r['pixel_sha256']}".encode()))
        count = max(1,min(len(candidates)-1,int(len(candidates)*validation_fraction+.5)))
        for row in candidates[:count]:
            row['output_split']='val'
    for row in retained:
        extension=Path(row['source_path']).suffix.lower()
        row['output_path']=f"{row['output_split']}/{row['label']}/{row['pixel_sha256']}{extension}"
    if len(retained)!=len({r['pixel_sha256'] for r in retained}):
        raise ValueError('Duplicate content remains in the output plan.')
    return manifest


def build_dataset(source, output, *, seed=42, validation_fraction=.2, workers=8, config=None):
    source, output=source.resolve(),output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite output: {output}')
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Source and output must be separate directories.')
    files=[]
    for split in ('train','test'):
        folder=source/split
        if {p.name for p in folder.iterdir() if p.is_dir()}!=set(SOURCE_LABELS):
            raise ValueError(f'Unexpected class folders: {folder}')
        for label in SOURCE_LABELS:
            for path in sorted((folder/label).rglob('*')):
                if not path.is_file():
                    continue
                if not path.resolve().is_relative_to((folder/label).resolve()):
                    raise ValueError('Image escapes its source class folder.')
                if path.suffix.lower() not in EXTENSIONS:
                    raise ValueError(f'Unexpected non-image file: {path}')
                files.append(path)
    records=[]
    print(f'Auditing {len(files):,} source images...',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,record in enumerate(pool.map(lambda p:inspect(p,source),files),1):
            records.append(record)
            if i%5000==0:print(f'Audited {i:,}/{len(files):,}',flush=True)
    manifest=make_plan(records,seed=seed,validation_fraction=validation_fraction)
    retained=[r for r in manifest if r['status']=='retained']
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=output.with_name(output.name+'.building-'+uuid.uuid4().hex[:8])
    staging.mkdir()
    for split in ('train','val','test'):
        for label in LABELS:(staging/split/label).mkdir(parents=True)
    def copy_checked(row):
        raw=(source/row['source_path']).read_bytes()
        if sha(raw)!=row['file_sha256']:raise ValueError('Source changed during build.')
        dest=staging/row['output_path'];dest.write_bytes(raw)
        if sha(dest.read_bytes())!=row['file_sha256']:raise ValueError('Output verification failed.')
    print(f'Copying {len(retained):,} images...',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,_ in enumerate(pool.map(copy_checked,retained),1):
            if i%5000==0:print(f'Copied and verified {i:,}/{len(retained):,}',flush=True)
    with (staging/'manifest.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
    mapping={str(index):label for index,label in enumerate(LABELS)}
    (staging/'label_mapping.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    counts={s:{label:sum(r['output_split']==s and r['label']==label for r in retained) for label in LABELS}
            for s in ('train','val','test')}
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_images=len(records),
        retained_images=len(retained),counts=counts,model_ids=MODEL_IDS,labels_changed=0,
        excluded_by_reason=dict(Counter(r['reason'] for r in manifest if r['status']=='excluded')),
        excluded_conflicting_groups=len({r['pixel_sha256'] for r in manifest if r['reason']=='conflicting_labels'}),
        seed=seed,validation_fraction=validation_fraction,
        split_method='Per-class SHA256(seed|label|pixel_hash) rank; round(n*fraction), halves up.',
        source_manifest_sha256=sha('\n'.join(sorted(r['source_path']+'|'+r['file_sha256'] for r in records)).encode()),
        manifest_sha256=sha((staging/'manifest.csv').read_bytes()),
        implementation_sha256=sha(Path(__file__).read_bytes()),
        image_audit_implementation_sha256=sha(Path(__file__).with_name('prepare.py').read_bytes()),
        config_sha256=sha(config.read_bytes()) if config else None,
        python=platform.python_version(),pillow=PIL.__version__,
        verification='Every retained image verified against source file hash; one copy per decoded-pixel hash; no exact content overlap between splits.',
        evaluation_note='Six-class FER2013 with exclusions and deduplication. Not directly comparable to the original seven-class benchmark.',
        limitations='Original labels may contain errors. No full visual review, identity check or near-duplicate search.')
    (staging/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (staging/'README.md').write_text('''# Six-class FER2013

Classes: Angry, Disgust, Fear, Happy, Sad, Neutral. Original labels are unchanged.
Surprise files are excluded before assessing label conflicts. Identical-image groups
with multiple remaining labels are excluded. One copy of each other image is kept,
preferring test membership over training. Remaining training images are divided into
train/val with class stratification and seed 42. No source files are changed.

manifest.csv accounts for every source image and its output or exclusion reason.
label_mapping.json defines model IDs 0–5; Neutral has model ID 5 and source ID 6.
summary.json records counts, exclusions, settings, hashes and verification.

Run the project Python 3.13 interpreter with scripts/prepare_emotion_sixclass.py.
Use --output to rebuild into a different directory; existing output is never overwritten.
Report this six-class evaluation separately from the original seven-class benchmark.
Image-label correctness, near duplicates and subject overlap are not verified.
Source: https://www.kaggle.com/datasets/msambare/fer2013
Listed license: Database: Open Database, Contents: Database Contents.
''',encoding='utf-8')
    staging.rename(output)
    print(json.dumps(summary,indent=2),flush=True)
    return summary
