"""Manifest-backed three-class emotion datasets; no eager PyTorch imports."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tomllib

from PIL import Image

CLASS_TO_ID = {'negative': 0, 'neutral': 1, 'positive': 2}
EMOTION_TO_CLASS = {'angry':'negative', 'disgust':'negative', 'fear':'negative',
                    'sad':'negative', 'neutral':'neutral', 'happy':'positive'}
SPLITS = ('train', 'val', 'test')


@dataclass(frozen=True)
class EmotionSettings:
    data_root: Path
    batch_size: int
    workers: int
    seed: int
    image_size: int
    augmentation: str


def load_settings(config, *, root):
    config, root = Path(config), Path(root)
    with config.open('rb') as handle: values = tomllib.load(handle)
    if set(values) != {'data','preprocessing'}:
        raise ValueError('Expected data and preprocessing sections.')
    data, pre = values['data'], values['preprocessing']
    if set(data) != {'root','batch_size','workers','seed'} or set(pre) != {'image_size','channels','scale','augmentation'}:
        raise ValueError('Unexpected or missing emotion configuration keys.')
    if not isinstance(data['root'],str) or not data['root'].strip():
        raise ValueError('Dataset root must be a nonempty path.')
    for key in ('batch_size','workers','seed'):
        if type(data[key]) is not int: raise ValueError(f'{key} must be an integer.')
    if data['batch_size']<1 or data['workers']<0 or data['seed']<0:
        raise ValueError('Invalid batch size, worker count or seed.')
    if pre != dict(image_size=48,channels=1,scale=255,augmentation='none'):
        raise ValueError('Baseline preprocessing is 48x48, one grayscale channel, /255, no augmentation.')
    return EmotionSettings((root/Path(data['root']).expanduser()).resolve(),data['batch_size'],
                           data['workers'],data['seed'],pre['image_size'],pre['augmentation'])


@dataclass(frozen=True)
class EmotionSample:
    path: Path
    relative_path: str
    label: str
    target: int
    file_sha256: str
    pixel_sha256: str


class EmotionLayout:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        mapping=json.loads((self.root/'label_mapping.json').read_text(encoding='utf-8'))
        if mapping != {str(index):label for label,index in CLASS_TO_ID.items()}:
            raise ValueError('Dataset model IDs do not match Negative=0, Neutral=1, Positive=2.')
        self.manifest_bytes=(self.root/'manifest.csv').read_bytes()
        summary=json.loads((self.root/'summary.json').read_text(encoding='utf-8'))
        if hashlib.sha256(self.manifest_bytes).hexdigest()!=summary['manifest_sha256']:
            raise ValueError('Manifest checksum does not match the dataset summary.')
        self.manifest_sha256=summary['manifest_sha256']
        by_split=defaultdict(list);seen_paths=set();seen_pixels=set()
        for row in csv.DictReader(io.StringIO(self.manifest_bytes.decode('utf-8-sig'))):
            if row['status']=='excluded': continue
            if row['status']!='retained': raise ValueError('Unknown manifest status.')
            split,label=row['output_split'],row['label']
            if split not in SPLITS or label not in CLASS_TO_ID:
                raise ValueError('Unknown partition or target label.')
            if row['emotion_label']!=row['original_label'] or EMOTION_TO_CLASS.get(row['emotion_label'])!=label:
                raise ValueError('Source emotion/aggregate target mismatch.')
            if int(row['model_id'])!=CLASS_TO_ID[label]: raise ValueError('Manifest target ID mismatch.')
            relative=PurePosixPath(row['output_path'])
            if relative.is_absolute() or len(relative.parts)!=3 or relative.parts[:2]!=(split,label) or ':' in row['output_path'] or '\\' in row['output_path']:
                raise ValueError('Invalid output image path in manifest.')
            if row['output_path'] in seen_paths or row['pixel_sha256'] in seen_pixels:
                raise ValueError('Duplicate image path/content in manifest; possible split leakage.')
            if any(len(row[key])!=64 or any(c not in '0123456789abcdef' for c in row[key]) for key in ('file_sha256','pixel_sha256')):
                raise ValueError('Invalid image hash.')
            seen_paths.add(row['output_path']);seen_pixels.add(row['pixel_sha256'])
            by_split[split].append(EmotionSample(self.root/relative,row['output_path'],label,
                                                CLASS_TO_ID[label],row['file_sha256'],row['pixel_sha256']))
        self.samples={s:tuple(by_split[s]) for s in SPLITS}
        self.counts={s:dict(Counter(sample.label for sample in self.samples[s])) for s in SPLITS}
        if self.counts!=summary['counts']:
            raise ValueError('Manifest class counts differ from the summary.')
        # Compare the real file inventory without decoding every image at construction time.
        for split in SPLITS:
            directory=self.root/split
            if {p.name for p in directory.iterdir() if p.is_dir()}!=set(CLASS_TO_ID):
                raise ValueError('Unexpected class folders.')
            actual=set()
            for label in CLASS_TO_ID:
                folder=directory/label
                if not folder.resolve().is_relative_to(directory.resolve()):
                    raise ValueError('Class folder escapes its partition.')
                for image in folder.iterdir():
                    if not image.is_file(): raise ValueError('Nested/invalid entries in a class folder.')
                    actual.add(image.relative_to(self.root).as_posix())
            expected={sample.relative_path for sample in self.samples[split]}
            if actual!=expected: raise ValueError(f'Missing or extra images in {split}.')


class EmotionDataset:
    """PyTorch map-style protocol; return PIL images when no transform is supplied."""
    def __init__(self,layout,split,transform=None):
        if split not in SPLITS: raise ValueError('Expected train, val or test.')
        self.layout,self.split,self.transform=layout,split,transform
        self.samples=layout.samples[split]
        self.class_to_idx=dict(CLASS_TO_ID)

    def __len__(self):return len(self.samples)

    def __getitem__(self,index):
        sample=self.samples[index]
        path=sample.path.resolve()
        if not path.is_relative_to((self.layout.root/self.split/sample.label).resolve()):
            raise ValueError('Image path escapes its class folder.')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=sample.file_sha256:
            raise ValueError(f'Image hash changed: {sample.relative_path}')
        try:
            with Image.open(io.BytesIO(raw)) as source:
                source.load()
                if source.mode!='L' or source.size!=(48,48):raise ValueError('Expected prepared 48x48 grayscale image.')
                image=source.copy()
        except (OSError,ValueError) as exc:
            raise ValueError(f'Cannot decode emotion image {sample.relative_path}: {exc}') from exc
        return (self.transform(image) if self.transform else image),sample.target


def tensor_transform(image_size=48):
    if image_size!=48:raise ValueError('Baseline input size is 48.')
    from .tensor_data import GrayscaleTensor
    return GrayscaleTensor(image_size)


def make_loaders(layout,settings):
    import torch
    from torch.utils.data import DataLoader
    transform=tensor_transform(settings.image_size)
    generator=torch.Generator().manual_seed(settings.seed)
    return {split:DataLoader(EmotionDataset(layout,split,transform),batch_size=settings.batch_size,
                            shuffle=split=='train',num_workers=settings.workers,
                            generator=generator if split=='train' else None,drop_last=False)
            for split in SPLITS}
