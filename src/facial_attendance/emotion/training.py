"""Training helpers and resumable, weights-only-compatible emotion checkpoints."""
from dataclasses import replace
import hashlib
import json
import math
import random
from pathlib import Path
import tomllib

import numpy as np
import torch
from torch import nn

from .data_loader import CLASS_TO_ID, EmotionLayout, load_settings, make_loaders
from .model import EmotionCNN
from ..evaluation.metrics import classification_metrics


def load_training_config(path):
    with Path(path).open('rb') as handle:
        config = tomllib.load(handle)
    expected = {'loader': {'config','batch_size','manifest_sha256'},
                'model': {'channels','dropout'},
                'optimizer': {'learning_rate','betas','epsilon','weight_decay'},
                'training': {'seed','max_epochs','patience'}}
    if set(config) != set(expected) or any(set(config[k]) != v for k,v in expected.items()):
        raise ValueError('Unexpected or missing training configuration keys.')
    for table,key,minimum in [('loader','batch_size',1),('training','seed',0),
                              ('training','max_epochs',1),('training','patience',1)]:
        value = config[table][key]
        if type(value) is not int or value < minimum:
            raise ValueError(f'Invalid {table}.{key}.')
    if config['model']['channels'] != [32,64,128]:
        raise ValueError('This baseline requires channels [32,64,128].')
    numeric = [('model','dropout'),('optimizer','learning_rate'),('optimizer','epsilon'),('optimizer','weight_decay')]
    for table,key in numeric:
        value = config[table][key]
        if type(value) not in (int,float) or not math.isfinite(value):
            raise ValueError(f'Invalid {table}.{key}.')
    o = config['optimizer']
    if not 0 <= config['model']['dropout'] < 1 or min(o['learning_rate'],o['epsilon']) <= 0 or o['weight_decay'] < 0:
        raise ValueError('Invalid optimizer/dropout range.')
    if not isinstance(o['betas'],list) or len(o['betas']) != 2 or any(type(b) not in (int,float) or not 0 <= b < 1 for b in o['betas']):
        raise ValueError('Adam requires two beta values within [0,1).')
    h = config['loader']['manifest_sha256']
    if not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h):
        raise ValueError('Invalid expected dataset manifest hash.')
    if not isinstance(config['loader']['config'],str) or not config['loader']['config'].strip():
        raise ValueError('Loader config path is required.')
    return config


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def build_model_optimizer(config, device):
    model = EmotionCNN(config['model']['channels'],config['model']['dropout']).to(device)
    o = config['optimizer']
    optimizer = torch.optim.Adam(model.parameters(),lr=o['learning_rate'],betas=tuple(o['betas']),
                                 eps=o['epsilon'],weight_decay=o['weight_decay'])
    return model,optimizer


def build_data(config, root):
    settings = load_settings(root/config['loader']['config'],root=root)
    settings = replace(settings,batch_size=config['loader']['batch_size'],seed=config['training']['seed'])
    layout = EmotionLayout(settings.data_root)
    if layout.manifest_sha256 != config['loader']['manifest_sha256']:
        raise ValueError('Dataset manifest differs from the approved training data.')
    return layout,make_loaders(layout,settings)


def run_epoch(model, loader, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0; truth = []; predictions = []
    with torch.set_grad_enabled(training):
        for index,(images,targets) in enumerate(loader):
            if max_batches is not None and index >= max_batches: break
            images,targets = images.to(device),targets.to(device)
            if not torch.isfinite(images).all(): raise ValueError('Non-finite input images.')
            if training: optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if not torch.isfinite(logits).all(): raise ValueError('Non-finite model output.')
            loss = criterion(logits,targets)
            if not torch.isfinite(loss): raise ValueError('Non-finite loss.')
            if training:
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('Non-finite gradients.')
                optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            truth.extend(targets.detach().cpu().tolist())
            predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
    metrics = classification_metrics(truth,predictions,[0,1,2],['Negative','Neutral','Positive'])
    metrics['loss'] = total_loss/len(truth)
    return metrics


def improved(candidate, best):
    return best is None or (candidate['macro_f1'],-candidate['loss']) > (best['macro_f1'],-best['loss'])


def capture_rng(generator):
    state = np.random.get_state()
    return {'python':random.getstate(),'numpy':[state[0],state[1].tolist(),state[2],state[3],state[4]],
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'loader':generator.get_state()}


def restore_rng(state,generator):
    random.setstate(state['python'])
    n = state['numpy']; np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'].cpu()); generator.set_state(state['loader'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def checkpoint_payload(model, optimizer, config, *, run_id, epoch, best, bad_epochs, history, manifest_hash, generator, source_hashes):
    return {'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'config':config,
            'meta':{'architecture':'EmotionCNN','model_id':run_id,'model_version':weights_hash(model),
                    'label_mapping':dict(CLASS_TO_ID),'input_size':48,'channels':1,'scale':255,
                    'threshold':None,'threshold_source':'not_applicable_argmax'},
            'epoch':epoch,'best':best,'bad_epochs':bad_epochs,'history':history,
            'manifest_sha256':manifest_hash,'rng':capture_rng(generator),'source_sha256':source_hashes}


def weights_hash(model):
    digest = hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path,payload):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temporary); temporary.replace(path)


def load_checkpoint(path,device='cpu'):
    payload = torch.load(path,map_location=device,weights_only=True)
    config = payload['config']
    model,optimizer = build_model_optimizer(config,device)
    model.load_state_dict(payload['state_dict']); optimizer.load_state_dict(payload['optimizer'])
    if payload['meta']['label_mapping'] != CLASS_TO_ID or weights_hash(model) != payload['meta']['model_version']:
        raise ValueError('Checkpoint labels or weight hash mismatch.')
    return model,optimizer,payload


def source_hashes(root):
    paths=['src/facial_attendance/emotion/model.py','src/facial_attendance/emotion/training.py',
           'src/facial_attendance/emotion/data_loader.py','src/facial_attendance/emotion/tensor_data.py',
           'scripts/train_emotion.py','configs/emotion.toml']
    # Canonical LF hashes identify source content across Windows/Linux checkouts.
    return {p:hashlib.sha256((root/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in paths}
