import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from facial_attendance.emotion.data_loader import (CLASS_TO_ID,EmotionLayout,EmotionDataset,
                                                  EmotionSettings,make_loaders,tensor_transform)


class EmotionLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.rows=[]
        source={'negative':'angry','neutral':'neutral','positive':'happy'}
        self.counts={}
        for si,split in enumerate(('train','val','test')):
            self.counts[split]={}
            for label,target in CLASS_TO_ID.items():
                self.counts[split][label]=2
                folder=self.root/split/label;folder.mkdir(parents=True)
                for i in range(2):
                    path=folder/f'{i}.png';image=Image.new('L',(48,48),si*60+target*10+i)
                    image.save(path)
                    pixel=json.dumps([image.mode,image.size],separators=(',',':')).encode()+image.tobytes()
                    self.rows.append(dict(status='retained',output_split=split,label=label,
                        model_id=target,emotion_label=source[label],original_label=source[label],
                        output_path=path.relative_to(self.root).as_posix(),
                        pixel_sha256=hashlib.sha256(pixel).hexdigest(),
                        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        (self.root/'label_mapping.json').write_text(json.dumps({str(v):k for k,v in CLASS_TO_ID.items()}))
        self.write_manifest()

    def write_manifest(self):
        path=self.root/'manifest.csv'
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(self.rows[0]));w.writeheader();w.writerows(self.rows)
        (self.root/'summary.json').write_text(json.dumps(dict(counts=self.counts,
            manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))

    def test_all_partitions_have_explicit_targets(self):
        layout=EmotionLayout(self.root)
        for split in ('train','val','test'):
            ds=EmotionDataset(layout,split)
            self.assertEqual(len(ds),6)
            self.assertEqual(ds.class_to_idx,CLASS_TO_ID)
            for i,sample in enumerate(ds.samples):
                image,target=ds[i];self.assertEqual(target,CLASS_TO_ID[sample.label])
                self.assertEqual(image.mode,'L');self.assertEqual(image.size,(48,48))

    def test_duplicate_pixels_cannot_cross_partitions(self):
        self.rows[-1]['pixel_sha256']=self.rows[0]['pixel_sha256'];self.write_manifest()
        with self.assertRaisesRegex(ValueError,'Duplicate'):EmotionLayout(self.root)

    def test_aggregation_alignment_and_path_are_validated(self):
        self.rows[0]['emotion_label']='happy';self.write_manifest()
        with self.assertRaises(ValueError):EmotionLayout(self.root)
        self.rows[0]['emotion_label']='angry';self.rows[0]['output_path']='../outside.png';self.write_manifest()
        with self.assertRaises(ValueError):EmotionLayout(self.root)

    def test_wrong_mapping_is_rejected(self):
        (self.root/'label_mapping.json').write_text(json.dumps({'0':'positive','1':'neutral','2':'negative'}))
        with self.assertRaises(ValueError):EmotionLayout(self.root)

    def test_changed_image_bytes_are_rejected(self):
        ds=EmotionDataset(EmotionLayout(self.root),'train')
        ds.samples[0].path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'hash changed'):ds[0]

    def test_missing_files_and_manifest_tampering_are_rejected(self):
        manifest=self.root/'manifest.csv';manifest.write_bytes(manifest.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'checksum'):EmotionLayout(self.root)
        self.write_manifest();(self.root/self.rows[0]['output_path']).unlink()
        with self.assertRaisesRegex(ValueError,'Missing'):EmotionLayout(self.root)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'),'Needs project ML environment')
    def test_tensor_batches_scale_and_repeatability(self):
        import torch
        layout=EmotionLayout(self.root)
        settings=EmotionSettings(self.root,4,0,42,48,'none')
        loaders=make_loaders(layout,settings)
        for split in ('train','val','test'):
            images,targets=next(iter(loaders[split]))
            self.assertEqual(tuple(images.shape),(4,1,48,48));self.assertEqual(targets.dtype,torch.int64)
            self.assertEqual(images.dtype,torch.float32)
            self.assertGreaterEqual(float(images.min()),0);self.assertLessEqual(float(images.max()),1)
        image,target=EmotionDataset(layout,'val',tensor_transform())[0]
        self.assertEqual(target,0);self.assertAlmostEqual(float(image[0,0,0]),60/255,places=6)
        other=next(iter(make_loaders(layout,settings)['val']))
        same=next(iter(make_loaders(layout,settings)['val']))
        self.assertTrue(torch.equal(other[0],same[0]))
        first=next(iter(make_loaders(layout,settings)['train']))
        second=next(iter(make_loaders(layout,settings)['train']))
        self.assertTrue(torch.equal(first[0],second[0]));self.assertTrue(torch.equal(first[1],second[1]))


if __name__=='__main__':unittest.main()
