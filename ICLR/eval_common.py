"""Shared paths and records for explicit-checkpoint benchmark evaluation."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import re
import tempfile
from typing import Any

PROJECT_ROOT=Path(__file__).resolve().parents[1]
ICLR_ROOT=PROJECT_ROOT/'ICLR'
ALPACA_SOURCE=PROJECT_ROOT/'alpaca_eval/AI-ModelScope/alpaca_eval/alpaca_eval_gpt4_baseline.json'
ARENA_QUESTION_SOURCE=PROJECT_ROOT/'arena-hard-auto/data/arena-hard-v0.1/question.jsonl'
ARENA_BASELINE_SOURCE=PROJECT_ROOT/'arena-hard-auto/data/arena-hard-v0.1/model_answer/base.jsonl'
MT_QUESTION_SOURCE=PROJECT_ROOT/'mt_bench/raw/question.jsonl'

@dataclass(frozen=True)
class ModelSpec:
    alias: str
    run_name: str
    path: Path
    chosen_region: str
    rejected_region: str
    fraction: float
    attention_heads: int | None
    rest_weight: float | None = None
    decomposition_mode: str | None = None
    gate_fn: str | None = None
    gate_scope: str | None = None
    model_size: str | None = None
    method: str | None = None
    tokenizer_path: Path | None = None
    model_family: str | None = None
    is_instruct: bool | None = None
    beta: float | None = None


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_jsonl_map(path: Path, key: str) -> dict[Any, dict[str, Any]]:
    return {row[key]: row for row in read_jsonl(path)}


def select_models(
    specs: list[ModelSpec], model: str | None, model_filter: str
) -> list[ModelSpec]:
    if model:
        selected = [spec for spec in specs if spec.alias == model]
        if not selected:
            raise ValueError(f"Unknown model alias: {model}")
        return selected
    if model_filter:
        needles = [part.strip().lower() for part in model_filter.split(",") if part.strip()]
        selected = [
            spec
            for spec in specs
            if any(
                needle in spec.alias.lower() or needle in spec.run_name.lower()
                for needle in needles
            )
        ]
        if not selected:
            raise ValueError(f"No model matched --model-filter={model_filter!r}")
        return selected
    return specs


def atomic_write_text(path: Path, text: str) -> None:
    """Publish a complete shared file when independent workers write concurrently."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_manifest(specs: list[ModelSpec], root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "model_manifest.json"
    rows = [
        {
            "alias": spec.alias,
            "run_name": spec.run_name,
            "model_path": str(spec.path),
            "chosen_region": spec.chosen_region,
            "rejected_region": spec.rejected_region,
            "selection_fraction": spec.fraction,
            "rest_weight": spec.rest_weight,
            "decomposition_mode": spec.decomposition_mode,
            "gate_fn": spec.gate_fn,
            "gate_scope": spec.gate_scope,
            "model_size": spec.model_size,
            "method": spec.method,
            "tokenizer_path": str(spec.tokenizer_path) if spec.tokenizer_path else None,
            "model_family": spec.model_family,
            "is_instruct": spec.is_instruct,
            "beta": spec.beta,
        }
        for spec in specs
    ]
    atomic_write_text(path, json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    return path


def discover_models(*args, **kwargs):
    raise RuntimeError('Use scripts/evaluate_checkpoint.py with an explicit checkpoint')

def evaluation_root(limit, experiment_family=None):
    root=ICLR_ROOT/'evaluation/main_seeds'
    return root/f'smoke_{limit}' if limit else root/'full'
