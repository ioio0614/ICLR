#!/usr/bin/env python3
"""CPU-only checks for the remaining-seeds package."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ICLR')]

def main():
    files=[]
    for name in ['ICLR','src','scripts']:
        files.extend(p for p in (ROOT/name).rglob('*.py') if 'experiments' not in p.parts and 'evaluation' not in p.parts)
    for p in files:ast.parse(p.read_text(),filename=str(p))
    subprocess.run(['bash','-n',str(ROOT/'env.sh')],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/restore_data.py'),'--verify-only'],check=True)
    from reviewer20_eval_common import inputs
    from eval_common import ARENA_BASELINE_SOURCE,read_jsonl
    a,b,c=inputs();assert (len(a),len(b),len(c))==(805,500,80)
    assert {r['uid'] for r in b}=={r['uid'] for r in read_jsonl(ARENA_BASELINE_SOURCE)}
    assert all(len(r['turns'])==2 for r in c)
    spec=importlib.util.spec_from_file_location('remaining_queue',ROOT/'scripts/run_pending_seeds.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    items=json.loads((ROOT/'migration/pending_main_jobs.json').read_text())['jobs']
    assert len(items)==55
    for count in [4,8]:
        jobs=m.configure(items,[str(i) for i in range(count)])
        assert len(m.selected_jobs(jobs,None,None))==55
        assert sum(j.role=='main_result_training' for j in jobs)==40
        assert len(m.selected_jobs(jobs,None,['tis_dpo']))==20
        for old,new in zip(items,jobs):
            if old['role']=='tis_weight_estimation':continue
            p=old['parameters'];q=new.parameters
            batch=len(old['gpus'])*p['per_device_train_batch_size']*p['gradient_accumulation_steps']
            assert batch==count*q['per_device_train_batch_size']*q['gradient_accumulation_steps']
            micro,acc=q['per_device_train_batch_size'],q['gradient_accumulation_steps']
            while micro>1:
                micro,acc=m.smaller_batch(micro,acc);assert count*micro*acc==batch
    print('Verified:',len(files),'Python files; 40 remaining models; TIS dependencies; 4/8 GPU batches; benchmarks 805/500/80x2.')

if __name__=='__main__':main()
