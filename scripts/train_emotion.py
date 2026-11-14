"""Train the agreed emotion baseline, or run a real-data preflight only."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from facial_attendance.emotion.training import (load_training_config,seed_everything,build_model_optimizer,
    build_data,run_epoch,improved,checkpoint_payload,save_checkpoint,load_checkpoint,source_hashes,restore_rng)


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def git(*args):
    try:return subprocess.check_output(['git',*args],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/emotion-training.toml')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--output',type=Path,help='Checkpoint directory; must be new except for resume.')
    p.add_argument('--resume',type=Path,help='Trusted project epoch checkpoint; requires the same output directory.')
    p.add_argument('--smoke-test',action='store_true',help='One train batch, one validation batch, save/reload; no full training/test evaluation.')
    a=p.parse_args()
    if a.threads<1:p.error('--threads must be positive.')
    if a.smoke_test and a.resume:p.error('Smoke tests cannot resume training.')
    if a.resume and not a.output:p.error('Resume requires the original --output directory.')
    device='cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device)
    if device=='cuda' and not torch.cuda.is_available():p.error('CUDA is not available in this environment.')
    torch.set_num_threads(a.threads)
    config=load_training_config((ROOT/a.config).resolve())
    seed_everything(config['training']['seed'])
    sources=source_hashes(ROOT)
    layout,loaders=build_data(config,ROOT)
    commit=git('rev-parse','HEAD')
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+f"-emotion-cnn-{(commit or 'unknown')[:7]}-s{config['training']['seed']}"
    output=(ROOT/a.output).resolve() if a.output else ROOT/'artifacts/checkpoints'/run_id
    if a.resume:
        if a.resume.resolve().parent!=output:p.error('Resume checkpoint must be in the original output directory.')
        model,optimizer,old=load_checkpoint(a.resume,device)
        if old['config']!=config or old['manifest_sha256']!=layout.manifest_sha256 or old['source_sha256']!=sources:
            p.error('Resume configuration, dataset or source code differs from the saved run.')
        if old['bad_epochs']>=config['training']['patience']:p.error('This run already reached early stopping.')
        epoch_start=old['epoch']+1; best=old['best'];bad=old['bad_epochs'];history=old['history'];run_id=old['meta']['model_id']
        if not (output/'best.pt').is_file():p.error('Original best.pt is required for resume.')
        if any((output/f'epoch-{e:03d}.pt').exists() for e in range(epoch_start,config['training']['max_epochs']+1)):
            p.error('Later checkpoints exist; resume the latest completed epoch instead.')
        restore_rng(old['rng'],loaders['train'].generator)
    else:
        output.mkdir(parents=True,exist_ok=False)
        model,optimizer=build_model_optimizer(config,device)
        epoch_start=1;best=None;bad=0;history=[]
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    common={'run_id':run_id,'config':config,'dataset_manifest_sha256':layout.manifest_sha256,
            'counts':layout.counts,'python':platform.python_version(),'torch':str(torch.__version__),
            'device':device,'device_name':torch.cuda.get_device_name() if device=='cuda' else platform.processor(),
            'threads':a.threads,'git_commit':commit,'git_dirty':bool(git('status','--porcelain')),
            'source_sha256':sources,'threshold':None,'threshold_source':'not_applicable_argmax',
            'resume_from':str(a.resume) if a.resume else None,
            'test_used':False,'training_data_role':'weights','validation_data_role':'checkpoint selection'}
    if a.smoke_test:
        before={k:v.detach().clone() for k,v in model.named_parameters()}
        train=run_epoch(model,loaders['train'],device,optimizer,max_batches=1)
        validation=run_epoch(model,loaders['val'],device,max_batches=1)
        changed=any(not torch.equal(before[k],v) for k,v in model.named_parameters())
        if not changed:raise RuntimeError('Optimizer did not update any trainable parameter.')
        payload=checkpoint_payload(model,optimizer,config,run_id=run_id,epoch=0,best=None,bad_epochs=0,
            history=[],manifest_hash=layout.manifest_sha256,generator=loaders['train'].generator,source_hashes=sources)
        save_checkpoint(output/'smoke.pt',payload)
        reloaded,_,_=load_checkpoint(output/'smoke.pt',device)
        images,targets=next(iter(loaders['val']));images=images.to(device)
        model.eval();reloaded.eval()
        with torch.no_grad():
            expected=model(images);actual=reloaded(images)
        if not torch.equal(expected,actual):raise RuntimeError('Checkpoint reload changed evaluation predictions.')
        report={**common,'status':'PASS','scope':'One real train batch and one validation batch, not trained-model performance.',
                'batch_shape':list(images.shape),'target_shape':list(targets.shape),'output_shape':list(actual.shape),
                'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                'optimizer_updated_weights':changed,'reload_predictions_identical':True,
                'train_batch':train,'validation_batch':validation,
                'checkpoint_sha256':hashlib.sha256((output/'smoke.pt').read_bytes()).hexdigest(),
                'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None}
        write_json(output/'smoke-report.json',report)
        print(f'PASS: batch 256 train/validation and checkpoint reload. Report: {output / "smoke-report.json"}')
        return
    report_dir=ROOT/'reports/runs'/run_id
    report_dir.mkdir(parents=True,exist_ok=bool(a.resume))
    # Save each completed epoch before updating the aggregate run manifest.
    for epoch in range(epoch_start,config['training']['max_epochs']+1):
        train=run_epoch(model,loaders['train'],device,optimizer)
        validation=run_epoch(model,loaders['val'],device)
        is_best=improved(validation,best)
        if is_best:best={**validation,'epoch':epoch};bad=0
        else:bad+=1
        history.append({'epoch':epoch,'train':train,'validation':validation})
        payload=checkpoint_payload(model,optimizer,config,run_id=run_id,epoch=epoch,best=best,bad_epochs=bad,
            history=history,manifest_hash=layout.manifest_sha256,generator=loaders['train'].generator,source_hashes=sources)
        path=output/f'epoch-{epoch:03d}.pt'
        save_checkpoint(path,payload)
        if is_best:
            temp=output/'best.pt.tmp';shutil.copyfile(path,temp);temp.replace(output/'best.pt')
        write_json(report_dir/'manifest.json',{**common,'history':history,'selected_epoch':best['epoch'],
            'selection_metric':'validation macro F1, then lower loss, then earlier epoch','selected_validation':best,
            'checkpoint_location':str(output/'best.pt'),
            'checkpoint_sha256':hashlib.sha256((output/'best.pt').read_bytes()).hexdigest(),
            'last_completed_epoch':epoch,'early_stopped':bad>=config['training']['patience'],
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None})
        print(f"Epoch {epoch}: train loss={train['loss']:.4f}; val loss={validation['loss']:.4f}; val macro F1={validation['macro_f1']:.4f}",flush=True)
        if bad>=config['training']['patience']:break
    print(f'Training finished. Selected checkpoint: {output / "best.pt"}. Test split was not evaluated.')


if __name__=='__main__':main()
