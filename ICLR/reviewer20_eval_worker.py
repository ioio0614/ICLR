#!/usr/bin/env python3
"""Isolated generation or judging worker using the existing benchmark protocol."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import threading

from reviewer20_eval_common import models, inputs, output_paths, answer_rows, atomic_json, ALPACA, stamp


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['generate', 'judge', 'summary'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--aliases', required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--seqs', type=int, default=256)
    p.add_argument('--tokens', type=int, default=32768)
    p.add_argument('--context', type=int, default=8192)
    p.add_argument('--parallel', type=int, default=1024)
    p.add_argument('--endpoint', default='http://127.0.0.1:8065/v1')
    return p


def preserve_judge_errors(judge, root):
    """Keep failed API attempts and removed cache records outside scoring files."""
    lock = threading.Lock()
    original_request, original_compact = judge.openai_request, judge.compact_jsonl
    audit = root / 'audit'
    audit.mkdir(parents=True, exist_ok=True)
    def request(messages, args, max_tokens, **kwargs):
        raw, error = original_request(messages, args, max_tokens, **kwargs)
        if error:
            with lock, (audit / 'request_errors.jsonl').open('a') as f:
                f.write(json.dumps({'time': stamp(), 'error': error, 'max_tokens': max_tokens,
                                    'prompt_sha256': hashlib.sha256(json.dumps(messages).encode()).hexdigest()}) + '\n')
        return raw, error
    def compact(path, rows):
        if path.exists():
            retained = {json.dumps(r, sort_keys=True) for r in rows}
            removed = [r for r in judge.read_jsonl(path) if json.dumps(r, sort_keys=True) not in retained]
            if removed:
                name = hashlib.sha256(str(path).encode()).hexdigest()[:16]
                with (audit / f'removed_cache_{name}.jsonl').open('a') as f:
                    for row in removed:
                        f.write(json.dumps({'source': str(path), 'record': row}, ensure_ascii=False) + '\n')
        return original_compact(path, rows)
    judge.openai_request, judge.compact_jsonl = request, compact


def judge_args(a):
    return argparse.Namespace(limit=a.limit, overwrite_judgments=False, judge_id='qwen2.5-14b-local',
        judge_model='qwen-local-judge', judge_api_key='EMPTY', judge_api_base=a.endpoint,
        judge_timeout=900.0, judge_retries=4, judge_completion_passes=5,
        judge_parallel=a.parallel, alpaca_judge_parallel=a.parallel, judge_parallel_models=8,
        alpaca_eval_bin=str(ALPACA), arena_judge_max_tokens=2048, mt_judge_max_tokens=256,
        bootstrap_samples=1000, experiment_family='reviewer20')


def main():
    a = parser().parse_args()
    aliases = a.aliases.split(',')
    lookup = {s.alias: s for s in models()}
    specs = [lookup[name] for name in aliases]
    if len(set(aliases)) != len(aliases):
        raise ValueError('Duplicate aliases')
    a.root.mkdir(parents=True, exist_ok=True)
    if a.stage == 'generate':
        if len(specs) != 1 or not 0 <= a.shard < a.shards:
            raise ValueError('One model and a valid shard required')
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        import generate_eval_outputs as generator
        selected = tuple(group[a.shard::a.shards] for group in inputs(a.limit))
        if answer_rows(a.root, specs[0], selected) is not None:
            print('[skip] exact answer IDs already complete', flush=True)
            return
        settings = argparse.Namespace(limit=a.limit, experiment_family='reviewer20', overwrite=False,
            tensor_parallel_size=1, gpu_memory_utilization=.85, max_model_len=a.context,
            max_num_seqs=a.seqs, max_num_batched_tokens=a.tokens, max_new_tokens=1024)
        generator.generate_one(specs[0], settings, root=a.root, inputs=selected)
        if answer_rows(a.root, specs[0], selected) is None:
            raise RuntimeError('Generation did not produce exact expected question IDs')
        return
    import judge_eval_outputs as judge
    for spec in specs:
        if answer_rows(a.root, spec, inputs(a.limit)) is None:
            raise RuntimeError(f'Invalid candidate answers: {spec.alias}')
    settings = judge_args(a)
    preserve_judge_errors(judge, a.root)
    if a.stage == 'judge':
        for name, runner in (('alpaca', judge.run_alpaca), ('arena', judge.run_arena), ('mt', judge.run_mt)):
            atomic_json(a.root / 'judge_stage.json', {'stage': name, 'started_at_utc': stamp(), 'pid': os.getpid()})
            judge.run_model_stage(name, runner, specs, settings, a.root)
    judge.validate_evaluation_complete(specs, settings, a.root)
    # Validate IDs as well as counts before publishing the final table.
    arena_ids = {r['uid'] for r in inputs(a.limit)[1]}
    mt_ids = {(r['question_id'], t) for r in inputs(a.limit)[2] for t in (1, 2)}
    for spec in specs:
        arena = judge.read_jsonl(a.root / 'arena_hard_v0.1/model_judgment' / settings.judge_id / f'{spec.alias}.jsonl')
        mt = judge.read_jsonl(a.root / 'mt_bench/model_judgment' / settings.judge_id / f'{spec.alias}.jsonl')
        if {r['uid'] for r in arena if judge.valid_arena_record(r)} != arena_ids:
            raise RuntimeError(f'Wrong Arena IDs: {spec.alias}')
        if {(r['question_id'], r['turn']) for r in mt if r.get('status') == 'ok'} != mt_ids:
            raise RuntimeError(f'Wrong MT IDs: {spec.alias}')
    csv_path = judge.write_summary(specs, settings, a.root)
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    fields = ['model', 'alpaca_lc', 'alpaca_wr', 'arena_wr', 'mt_overall']
    for row in rows:
        for key in fields[1:]:
            score = float(row[key])
            lo, hi = (1, 10) if key == 'mt_overall' else (0, 100)
            if not math.isfinite(score) or not lo <= score <= hi:
                raise RuntimeError(f'Invalid final metric {key}: {row}')
    with (a.root / 'summary/four_metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(a.root / 'judging_complete.json', {'models': aliases, 'time': stamp()})


if __name__ == '__main__':
    main()
