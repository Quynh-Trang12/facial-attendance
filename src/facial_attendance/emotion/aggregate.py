"""Aggregate verified expression labels into three training targets."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

EMOTION_TO_CLASS = {'angry':'negative','disgust':'negative','fear':'negative','sad':'negative',
                    'happy':'positive','neutral':'neutral'}
CLASS_TO_ID = {'negative':0,'neutral':1,'positive':2}


def digest(data):return hashlib.sha256(data).hexdigest()


def aggregate_plan(rows):
    result=[]
    for record in rows:
        row=dict(record)
        row['emotion_label']=row.pop('label')
        row.update(label='',model_id='',output_path='')
        if row['status']=='retained':
            if row['emotion_label'] not in EMOTION_TO_CLASS:
                raise ValueError('Retained source expression has no aggregation mapping.')
            if row['output_split'] not in ('train','val','test'):
                raise ValueError('Invalid source partition.')
            row['label']=EMOTION_TO_CLASS[row['emotion_label']]
            row['model_id']=CLASS_TO_ID[row['label']]
            row['output_path']=f"{row['output_split']}/{row['label']}/{row['pixel_sha256']}{Path(record['output_path']).suffix.lower()}"
        result.append(row)
    retained=[r for r in result if r['status']=='retained']
    if len(retained)!=len({r['pixel_sha256'] for r in retained}):
        raise ValueError('Source selection has duplicate pixels.')
    return result


def build_dataset(selection_root,output,*,workers=8):
    selection_root,output=selection_root.resolve(),output.resolve()
    if output.exists():raise FileExistsError(f'Refusing to overwrite {output}')
    if output.is_relative_to(selection_root) or selection_root.is_relative_to(output):
        raise ValueError('Input and output must be separate directories.')
    manifest_path=selection_root/'manifest.csv'
    source_summary=json.loads((selection_root/'summary.json').read_text())
    raw_manifest=manifest_path.read_bytes()
    if digest(raw_manifest)!=source_summary['manifest_sha256']:
        raise ValueError('Selection manifest does not match its recorded hash.')
    with manifest_path.open(encoding='utf-8',newline='') as handle:source_rows=list(csv.DictReader(handle))
    if any(r['label']!=r['original_label'] for r in source_rows):
        raise ValueError('Original expression labels must be unchanged.')
    if len(source_rows)!=source_summary['source_images']:
        raise ValueError('Selection source count mismatch.')
    rows=aggregate_plan(source_rows)
    selected=[(src,dst) for src,dst in zip(source_rows,rows) if dst['status']=='retained']
    if len(selected)!=source_summary['retained_images']:raise ValueError('Selection retained count mismatch.')
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=output.with_name(output.name+'.building-'+uuid.uuid4().hex[:8]);stage.mkdir()
    for split in ('train','val','test'):
        for label in CLASS_TO_ID:(stage/split/label).mkdir(parents=True)
    def copy_checked(pair):
        src,dst=pair
        source=(selection_root/src['output_path']).resolve()
        if not source.is_relative_to(selection_root):raise ValueError('Source path escapes selection directory.')
        raw=source.read_bytes()
        if digest(raw)!=dst['file_sha256']:raise ValueError('Source image hash mismatch.')
        target=stage/dst['output_path'];target.write_bytes(raw)
        if digest(target.read_bytes())!=dst['file_sha256']:raise ValueError('Output image hash mismatch.')
    print(f'Copying and verifying {len(selected):,} images...',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index,_ in enumerate(pool.map(copy_checked,selected),1):
            if index%5000==0:print(f'Verified {index:,}/{len(selected):,}',flush=True)
    with (stage/'manifest.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (stage/'label_mapping.json').write_text(json.dumps({str(v):k for k,v in CLASS_TO_ID.items()},indent=2),encoding='utf-8')
    (stage/'emotion_mapping.json').write_text(json.dumps(EMOTION_TO_CLASS,indent=2),encoding='utf-8')
    retained=[dst for _,dst in selected]
    counts={s:{label:sum(r['output_split']==s and r['label']==label for r in retained) for label in CLASS_TO_ID}
            for s in ('train','val','test')}
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_images=len(rows),retained_images=len(retained),
        counts=counts,model_ids=CLASS_TO_ID,emotion_to_class=EMOTION_TO_CLASS,excluded_class='surprise',
        source_emotion_labels_changed=0,excluded_by_reason=source_summary['excluded_by_reason'],
        seed=source_summary['seed'],validation_fraction=source_summary['validation_fraction'],
        split_method=source_summary['split_method'],
        partition_policy='Preserve verified image membership; validation was stratified by source emotion.',
        source_manifest_sha256=source_summary['source_manifest_sha256'],
        selection_manifest_sha256=digest(raw_manifest),selection_implementation_sha256=source_summary['implementation_sha256'],
        manifest_sha256=digest((stage/'manifest.csv').read_bytes()),implementation_sha256=digest(Path(__file__).read_bytes()),
        verification='All source and copied image hashes checked. One retained image per pixel hash; no exact content overlap between splits.',
        evaluation_note='Three aggregated classes derived from FER2013 annotations; not directly comparable to seven-class benchmark scores.',
        limitations='Surprise excluded. Original annotation errors may remain. Groups describe expression categories, not measured mood or mental health.')
    (stage/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (stage/'README.md').write_text('''# Three-class FER2013

Negative: Angry, Disgust, Fear, Sad. Neutral: Neutral. Positive: Happy.
Surprise is excluded. Model IDs: negative=0, neutral=1, positive=2.

The manifest preserves each source emotion label and records its aggregated target.
Image selection and train/val/test membership come from a verified selection manifest:
Surprise and conflicting-label groups excluded, one copy per pixel hash, with test
membership taking priority over training. Validation uses seed 42 and source-emotion
stratification. Original expression labels are unchanged; training targets are aggregated.

Use scripts/prepare_emotion_3class.py to build. To reproduce from the original download,
first run scripts/prepare_emotion_sixclass.py to produce the selection manifest.
Both commands refuse to overwrite existing outputs; use --output to choose a new location.

manifest.csv accounts for every source image, original expression, target, split and exclusion.
summary.json records counts, settings, hashes and verification. emotion_mapping.json and
label_mapping.json provide the two mappings. These groups are a task definition, not
evidence of a person's mood or mental health. Original annotation errors may remain.
''',encoding='utf-8')
    stage.rename(output);print(json.dumps(summary,indent=2),flush=True)
    return summary
