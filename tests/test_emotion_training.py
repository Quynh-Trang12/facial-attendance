"""Training configuration, checkpoint and selection checks with synthetic inputs."""
import importlib.util
import random
from pathlib import Path
import tempfile
import unittest

if importlib.util.find_spec('torch') is None:
    raise unittest.SkipTest('Training tests require the CPU/CUDA project environment; OS CI installs it.')
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from facial_attendance.emotion.model import EmotionCNN
from facial_attendance.emotion.training import (load_training_config,seed_everything,build_model_optimizer,
    run_epoch,improved,checkpoint_payload,save_checkpoint,load_checkpoint,restore_rng)

ROOT=Path(__file__).resolve().parents[1]


class EmotionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.config=load_training_config(ROOT/'configs/emotion-training.toml')
        seed_everything(42)

    def test_agreed_configuration_and_shapes(self):
        self.assertEqual(self.config['loader']['batch_size'],256)
        self.assertEqual(self.config['optimizer']['learning_rate'],.001)
        model=EmotionCNN()
        self.assertEqual(tuple(model(torch.rand(2,1,48,48)).shape),(2,3))
        self.assertEqual(sum(isinstance(m,torch.nn.MaxPool2d) for m in model.modules()),3)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        with self.assertRaises(ValueError):model(torch.rand(2,3,48,48))

    def test_selection_ties_and_earlier_epoch(self):
        a={'macro_f1':.5,'loss':1}
        self.assertTrue(improved(a,None))
        self.assertFalse(improved(dict(a),a))
        self.assertTrue(improved({'macro_f1':.5,'loss':.9},a))
        self.assertFalse(improved({'macro_f1':.4,'loss':.1},a))

    def test_optimizer_validation_and_checkpoint_resume(self):
        model,opt=build_model_optimizer(self.config,'cpu')
        gen=torch.Generator().manual_seed(42)
        x=torch.rand(6,1,48,48);y=torch.tensor([0,1,2,0,1,2])
        loader=DataLoader(TensorDataset(x,y),batch_size=3,shuffle=True,generator=gen)
        before=model.head[-1].weight.detach().clone()
        run_epoch(model,loader,'cpu',opt)
        self.assertFalse(torch.equal(before,model.head[-1].weight))
        buffers={k:v.clone() for k,v in model.state_dict().items()}
        metrics=run_epoch(model,loader,'cpu')
        self.assertEqual(metrics['count'],6)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in buffers.items()))
        payload=checkpoint_payload(model,opt,self.config,run_id='synthetic',epoch=1,best={**metrics,'epoch':1},
            bad_epochs=0,history=[],manifest_hash='a'*64,generator=gen,source_hashes={})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'checkpoint.pt';save_checkpoint(path,payload)
            # Continue once, then restore the entire state and repeat the same shuffled/dropout epoch.
            expected=run_epoch(model,loader,'cpu',opt)
            continued={k:v.clone() for k,v in model.state_dict().items()}
            restored,restored_opt,saved=load_checkpoint(path)
            restore_rng(saved['rng'],gen)
            actual=run_epoch(restored,loader,'cpu',restored_opt)
            self.assertEqual(expected,actual)
            self.assertTrue(all(torch.equal(v,restored.state_dict()[k]) for k,v in continued.items()))
            self.assertEqual(saved['meta']['label_mapping'],{'negative':0,'neutral':1,'positive':2})


if __name__=='__main__':unittest.main()
