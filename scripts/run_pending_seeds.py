#!/usr/bin/env python3
"""Run the exported pending main-table jobs without the source server's queue state."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'ICLR')]
import main_seed_common as queue
from main_seed_common import smaller_batch

def resolve(value):
    if isinstance(value, dict):return {k:resolve(v) for k,v in value.items()}
    if isinstance(value, list):return [resolve(v) for v in value]
    if isinstance(value, str):return value.replace('${PROJECT_ROOT}', str(ROOT)).replace('${PYTHON_BIN}', sys.executable)
    return value

def configure(items, gpus):
    jobs=[]
    for item in items:
        j=queue.Job(**resolve(copy.deepcopy(item)))
        p=j.parameters
        if j.role != 'tis_weight_estimation':
            batch=len(j.gpus)*p['per_device_train_batch_size']*p['gradient_accumulation_steps']
            micro=p['per_device_train_batch_size']
            while micro>1 and batch%(len(gpus)*micro):micro-=1
            if batch%(len(gpus)*micro):raise ValueError(f'Cannot preserve global batch {batch} on {len(gpus)} GPUs: {j.key}')
            p['per_device_train_batch_size']=micro
            p['gradient_accumulation_steps']=batch//(len(gpus)*micro)
        j.gpus=gpus
        j.provenance['migration']='Restart unfinished experiment from original base weights; source partial checkpoints are not included.'
        build_command(j)
        jobs.append(j)
    return jobs

def build_command(j):
    target=[str(ROOT/'scripts/estimate_tis_weights.py')] if j.role=='tis_weight_estimation' else ['-m', 'src.training.sft_unified' if j.method=='sft' else 'src.training.preference_optimization']
    j.command=[sys.executable,'-u','-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(j.gpus)}','--max_restarts=0',*target]
    for k,v in j.parameters.items():j.command.extend(['--'+k,str(v)])

def selected_jobs(jobs, models, methods):
    lookup={j.key:j for j in jobs}
    keep={j.key for j in jobs if j.role=='main_result_training' and (not models or j.model_key in models) and (not methods or j.method in methods)}
    def add(key):
        if key not in lookup:raise ValueError(f'Dependency missing from exported plan: {key}')
        keep.add(key)
        for d in lookup[key].dependencies:add(d)
    for k in list(keep):add(k)
    return [j for j in jobs if j.key in keep]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',default='0,1,2,3')
    parser.add_argument('--models',nargs='*',choices=list(queue.MODELS))
    parser.add_argument('--methods',nargs='*',choices=list(queue.METHODS))
    parser.add_argument('--execute',action='store_true',help='Default is a dry run, without GPUs or model loading.')
    args=parser.parse_args()
    gpus=args.gpus.split(',')
    if not gpus or any(not g.isdigit() for g in gpus) or len(set(gpus))!=len(gpus):raise ValueError('Invalid GPU IDs')
    plan_path=ROOT/'migration/pending_main_jobs.json'
    jobs=selected_jobs(configure(json.loads(plan_path.read_text())['jobs'],gpus),args.models,args.methods)
    if not jobs:raise ValueError('No pending jobs match the selection')
    print('Selected final models:',sum(j.role=='main_result_training' for j in jobs),'total tasks:',len(jobs))
    for j in jobs:
        print(j.key, 'global_batch='+str(len(gpus)*j.parameters.get('per_device_train_batch_size',1)*j.parameters.get('gradient_accumulation_steps',1)))
        if not args.execute:print(shlex.join(j.command))
    if not args.execute:return
    # Check every external input before starting a long queue.
    for j in jobs:
        queue.check_model(j.parameters['model_id'])
        for key in ['dataset_name','deepspeed']:
            if key in j.parameters and not Path(j.parameters[key]).is_file():raise FileNotFoundError(j.parameters[key])
    suite=argparse.Namespace(project=ROOT)
    state=ROOT/'ICLR/experiments/migration_queue';state.mkdir(parents=True,exist_ok=True)
    completed={}
    with queue.exclusive_lock(state/'queue.lock'):
        for j in jobs:
            record_path=state/(hashlib.sha256(j.key.encode()).hexdigest()[:20]+'.json')
            # Record identity independently of runtime microbatch after an OOM retry.
            identity={'job':asdict(j),'input_plan_sha256':queue.sha256(str(plan_path)),
                      'sources':{str(p.relative_to(ROOT)):queue.sha256(str(p)) for p in queue.source_files(suite,j)},
                      'data':{k:queue.sha256(j.parameters[k]) for k in ['dataset_name','deepspeed'] if k in j.parameters}}
            if record_path.exists():
                old=queue.json_read(record_path)
                if old['identity']!=identity:raise ValueError(f'Configuration changed; choose a fresh output: {j.key}')
                j.parameters.update(old['runtime_parameters']);build_command(j)
            elif Path(j.output).exists() and any(Path(j.output).iterdir()):
                raise ValueError(f'Untracked output exists: {j.output}; preserve it and choose a new destination')
            while True:
                queue.json_write(record_path,{'identity':identity,'runtime_parameters':j.parameters})
                if queue.check_completed(j):
                    completed[j.key]=True;print('[complete]',j.key,flush=True);break
                if any(d not in completed for d in j.dependencies):raise ValueError(f'Unfinished dependency: {j.key}')
                output=Path(j.output);output.mkdir(parents=True,exist_ok=True)
                log=output/('estimate.log' if j.role=='tis_weight_estimation' else 'train.log')
                env={**os.environ,'CUDA_VISIBLE_DEVICES':args.gpus,'PYTHONPATH':str(ROOT)+os.pathsep+str(ROOT/'ICLR'),
                     'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
                     'OMP_NUM_THREADS':'1','PYTORCH_ALLOC_CONF':'expandable_segments:True'}
                queue.json_write(state/'status.json',{'job':j.key,'completed_tasks':len(completed),'total_tasks':len(jobs),'state':'training'})
                try:
                    queue.stream_command(j.command,env,ROOT,log)
                except Exception:
                    text=log.read_text(errors='replace')[-262144:] if log.exists() else ''
                    if not re.search(r'CUDA out of memory|CUDA error:\s*out of memory|OutOfMemoryError',text,re.I):raise
                    key='batch_size' if j.role=='tis_weight_estimation' else 'per_device_train_batch_size'
                    micro=j.parameters[key]
                    if micro<=1:raise
                    if j.role=='tis_weight_estimation':j.parameters[key]=micro//2
                    else:
                        j.parameters[key],j.parameters['gradient_accumulation_steps']=smaller_batch(micro,j.parameters['gradient_accumulation_steps'])
                    archived=output.with_name(output.name+f'.oom-{time.time_ns()}');output.rename(archived)
                    build_command(j);print('[OOM retry]',j.key,j.parameters[key],flush=True);continue
                if not queue.check_completed(j):raise RuntimeError(f'No validated final artifact: {j.output}')
            queue.json_write(state/'status.json',{'job':j.key,'completed_tasks':len(completed),'total_tasks':len(jobs),'state':'complete' if len(completed)==len(jobs) else 'between_jobs'})

if __name__=='__main__':main()
