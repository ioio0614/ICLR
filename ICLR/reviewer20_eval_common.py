"""Benchmark validation utilities; no historical experiment discovery."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from eval_common import ALPACA_SOURCE, ARENA_QUESTION_SOURCE, MT_QUESTION_SOURCE, atomic_write_text, read_jsonl
ALPACA=Path(os.environ.get('ALPACA_EVAL_BIN') or shutil.which('alpaca_eval') or 'alpaca_eval')

def stamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    atomic_write_text(Path(path), json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(limit=None):
    alpaca = json.loads(ALPACA_SOURCE.read_text())
    arena = read_jsonl(ARENA_QUESTION_SOURCE)
    mt = [dict(question_id=r['question_id'], category=r['category'],
               turns=r.get('prompt', r.get('turns'))) for r in read_jsonl(MT_QUESTION_SOURCE)]
    if limit:
        return alpaca[:limit], arena[:limit], mt[:limit]
    return alpaca, arena, mt


def file_inventory(path):
    return {p.name: {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
            for p in sorted(Path(path).iterdir()) if p.is_file()}


def output_paths(root, alias):
    return (root / 'alpaca_outputs' / f'{alias}.json',
            root / 'arena_hard_v0.1/model_answer' / f'{alias}.jsonl',
            root / 'mt_bench/model_answer' / f'{alias}.jsonl')


def answer_rows(root, spec, expected_inputs):
    """Exact question identity and structure checks, including shard boundaries."""
    paths = output_paths(root, spec.alias)
    if not all(p.is_file() for p in paths):
        return None
    try:
        rows = [json.loads(paths[0].read_text()), read_jsonl(paths[1]), read_jsonl(paths[2])]
        for group, expected in zip(rows, expected_inputs):
            if len(group) != len(expected):
                return None
        if any(r.get('instruction') != e['instruction'] or not isinstance(r.get('output'), str)
               or r.get('generator') != spec.alias for r, e in zip(rows[0], expected_inputs[0])):
            return None
        if any(r.get('uid') != e['uid'] or r.get('model_id') != spec.alias
               or not isinstance(r['choices'][0]['message']['content'], str)
               for r, e in zip(rows[1], expected_inputs[1])):
            return None
        if any(r.get('question_id') != e['question_id'] or r.get('model_id') != spec.alias
               or len(r['choices'][0]['turns']) != 2 or not all(isinstance(t, str) for t in r['choices'][0]['turns'])
               for r, e in zip(rows[2], expected_inputs[2])):
            return None
        return rows
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def models():
    raise RuntimeError('Use scripts/evaluate_checkpoint.py with an explicit checkpoint')
