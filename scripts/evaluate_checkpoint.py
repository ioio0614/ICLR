#!/usr/bin/env python3
"""Evaluate any supplied checkpoint with the existing three-benchmark protocol."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ICLR')]
from eval_common import ModelSpec

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--alias',required=True)
    p.add_argument('--family',choices=['qwen2.5','llama3','mistral'],required=True)
    p.add_argument('--base',action='store_true',help='Raw pretrained base rather than instruct model')
    p.add_argument('--stage',choices=['generate','judge','summary'],required=True)
    p.add_argument('--output',type=Path,default=ROOT/'ICLR/evaluation/portable')
    p.add_argument('--endpoint',default='http://127.0.0.1:8065/v1')
    p.add_argument('--limit',type=int)
    a,extra=p.parse_known_args()
    if not a.alias or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in a.alias):raise ValueError('Alias must be a simple filename-safe name')
    from main_seed_common import check_model
    check_model(a.checkpoint)
    spec=ModelSpec(a.alias,a.alias,a.checkpoint.resolve(),'custom','custom',1.,None,model_family=a.family,is_instruct=not a.base)
    import reviewer20_eval_worker as worker
    worker.models=lambda:[spec]
    # Freeze checkpoint identity and parameters so cached outputs cannot silently mix models.
    from reviewer20_eval_common import atomic_json,file_inventory
    import json
    root=a.output.resolve()/('smoke_'+str(a.limit) if a.limit else 'full')/a.alias
    identity={'checkpoint':str(spec.path),'files':file_inventory(spec.path),'family':a.family,'base':a.base,'limit':a.limit}
    marker=root/'portable_manifests'/f'{a.alias}.json'
    if marker.exists() and json.loads(marker.read_text())!=identity:raise ValueError('Alias already belongs to a different checkpoint; use a new alias/output')
    atomic_json(marker,identity)
    sys.argv=[sys.argv[0],a.stage,'--root',str(root),'--aliases',a.alias,'--endpoint',a.endpoint,*extra]
    if a.limit:sys.argv+=['--limit',str(a.limit)]
    worker.main()

if __name__=='__main__':main()
