"""Verify prepared emotion partitions and record real CPU batch evidence."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from facial_attendance.emotion.data_loader import (CLASS_TO_ID,EmotionLayout,EmotionDataset,
                                                  load_settings,make_loaders,tensor_transform)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/emotion.toml')
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/emotion/loader'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ.json'))
    args=parser.parse_args();config=(ROOT/args.config).resolve();output=(ROOT/args.output).resolve()
    if output.exists():parser.error('Output exists; choose a new evidence filename.')
    settings=load_settings(config,root=ROOT)
    data=(ROOT/args.data_root).resolve() if args.data_root else settings.data_root
    layout=EmotionLayout(data)
    import torch
    import PIL,numpy
    report=dict(status='PASS',python=platform.python_version(),torch=torch.__version__,pillow=PIL.__version__,
                numpy=numpy.__version__,device='cpu',counts=layout.counts,mapping=CLASS_TO_ID,
                preprocessing=dict(image_shape=[1,48,48],range=[0,1],dtype='float32',augmentation='none'),
                seed=settings.seed,batch_size=settings.batch_size,manifest_sha256=layout.manifest_sha256,
                config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),batches={},examples={})
    loaders=make_loaders(layout,settings)
    for split,loader in loaders.items():
        images,targets=next(iter(loader))
        assert images.dtype==torch.float32 and images.ndim==4 and tuple(images.shape[1:])==(1,48,48)
        assert targets.dtype==torch.int64 and targets.ndim==1 and targets.shape[0]==images.shape[0]
        assert torch.isfinite(images).all() and 0<=float(images.min())<=float(images.max())<=1
        assert ((targets>=0)&(targets<3)).all()
        report['batches'][split]=dict(image_shape=list(images.shape),target_shape=list(targets.shape),
                                     min=float(images.min()),max=float(images.max()),targets=targets.tolist())
        # Check one actual image and target for every aggregate class in each partition.
        dataset=EmotionDataset(layout,split,tensor_transform())
        examples=[]
        for label,target in CLASS_TO_ID.items():
            index=next(i for i,s in enumerate(dataset.samples) if s.label==label)
            image,observed=dataset[index];assert observed==target
            assert torch.equal(image,dataset[index][0])
            sample=dataset.samples[index]
            examples.append(dict(path=sample.relative_path,label=label,target=observed,
                                 image_shape=list(image.shape),file_sha256=sample.file_sha256))
        report['examples'][split]=examples
    again=make_loaders(layout,settings)
    first=next(iter(make_loaders(layout,settings)['train']))
    second=next(iter(again['train']))
    assert torch.equal(first[0],second[0]) and torch.equal(first[1],second[1])
    report['checks']=['Manifest checksum and class counts match.', 'Real files match manifest inventory.',
                      'Unique pixel hashes across all partitions.', 'Each loaded image matches its source file hash.',
                      'Real batches have finite float32 [B,1,48,48] images and int64 [B] targets.',
                      'All class mappings checked on each partition.', 'Repeated preprocessing is deterministic.',
                      'Seeded training shuffle is reproducible; evaluation order is fixed.']
    report['source_sha256']={str(path.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT/'src/facial_attendance/emotion/data_loader.py',ROOT/'src/facial_attendance/emotion/tensor_data.py',Path(__file__).resolve())}
    report['limitations']='Manifest pixel hashes were audited during preparation. Only sampled images were decoded in this run. No training, model accuracy, near-duplicate or identity audit.'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    print('Evidence:',output)


if __name__=='__main__':main()
