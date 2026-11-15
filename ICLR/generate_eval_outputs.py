#!/usr/bin/env python3
"""Generate AlpacaEval 2, Arena-Hard v0.1 and MT-Bench answers."""

from __future__ import annotations

import argparse
from collections import deque
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.model_utils import detect_model_config, resolve_end_token_id

from eval_common import (
    ALPACA_SOURCE,
    ARENA_QUESTION_SOURCE,
    MT_QUESTION_SOURCE,
    ModelSpec,
    discover_models,
    evaluation_root,
    read_jsonl,
    select_models,
    write_manifest,
)


TEMPERATURE_BY_CATEGORY = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}


def load_inputs(limit: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    alpaca = json.loads(ALPACA_SOURCE.read_text(encoding="utf-8"))
    arena = read_jsonl(ARENA_QUESTION_SOURCE)
    mt_raw = read_jsonl(MT_QUESTION_SOURCE)
    mt = []
    for question in mt_raw:
        turns = question.get("prompt", question.get("turns"))
        if not isinstance(turns, list) or len(turns) != 2:
            raise ValueError(f"MT-Bench question {question.get('question_id')} is not two-turn")
        mt.append(
            {
                "question_id": question["question_id"],
                "category": question["category"],
                "turns": turns,
            }
        )
    if limit:
        return alpaca[:limit], arena[:limit], mt[:limit]
    return alpaca, arena, mt


def atomic_json(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def complete_json(path: Path, expected: int) -> bool:
    if not path.exists():
        return False
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    # An empty completion is a valid model outcome (for example, an immediate
    # EOS from a base model). Re-generating the entire dataset cannot reliably
    # eliminate those outcomes and makes completed runs restart forever.
    return len(rows) == expected and all("output" in row for row in rows)


def complete_jsonl(path: Path, expected: int, two_turn: bool = False) -> bool:
    if not path.exists():
        return False
    try:
        rows = read_jsonl(path)
    except (json.JSONDecodeError, OSError):
        return False
    if len(rows) != expected:
        return False
    if two_turn:
        return all(
            len(row.get("choices", [{}])[0].get("turns", [])) == 2
            for row in rows
        )
    return all(
        "content" in row.get("choices", [{}])[0].get("message", {})
        for row in rows
    )


def prompt_metadata(spec: ModelSpec) -> tuple[str, bool]:
    if spec.model_family is not None and spec.is_instruct is not None:
        return spec.model_family, spec.is_instruct
    return detect_model_config(str(spec.path))


def format_messages(
    tokenizer: Any, spec: ModelSpec, messages: list[dict[str, str]]
) -> str:
    family, is_instruct = prompt_metadata(spec)
    if is_instruct:
        formatted = list(messages)
        if family == "qwen2.5" and (
            not formatted or formatted[0].get("role") != "system"
        ):
            formatted = [
                {"role": "system", "content": "You are a helpful assistant."},
                *formatted,
            ]
        return tokenizer.apply_chat_template(
            formatted, tokenize=False, add_generation_prompt=True
        )

    # Base-model preference runs were trained from raw prompt continuations.
    # Preserve that convention for turn 1; for MT-Bench turn 2, append the
    # previous answer and follow-up with newlines without inventing chat tokens.
    return "\n".join(str(message["content"]).strip() for message in messages)


def generate_group(
    llm: Any,
    prompts: list[str],
    indices: list[int],
    temperature: float,
    max_tokens: int,
    repetition_penalty: float,
    stop_token_ids: list[int] | None,
) -> dict[int, str]:
    from vllm import SamplingParams

    if not indices:
        return {}
    params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        stop_token_ids=stop_token_ids,
        seed=0,
    )
    outputs = llm.generate([prompts[i] for i in indices], params, use_tqdm=True)
    return {
        i: output.outputs[0].text.strip() if output.outputs else ""
        for i, output in zip(indices, outputs)
    }


def compatible_tp(spec: ModelSpec, requested: int) -> int:
    heads = spec.attention_heads
    if not heads or heads % requested == 0:
        return requested
    for candidate in range(requested - 1, 0, -1):
        if heads % candidate == 0:
            print(f"[{spec.alias}] heads={heads} incompatible with TP={requested}; using TP={candidate}")
            return candidate
    return 1


def generate_one(spec: ModelSpec, args: argparse.Namespace, *, root=None, inputs=None) -> None:
    root = root if root is not None else evaluation_root(args.limit, args.experiment_family)
    alpaca, arena, mt = inputs if inputs is not None else load_inputs(args.limit)
    alpaca_path = root / "alpaca_outputs" / f"{spec.alias}.json"
    arena_path = root / "arena_hard_v0.1/model_answer" / f"{spec.alias}.jsonl"
    mt_path = root / "mt_bench/model_answer" / f"{spec.alias}.jsonl"

    need_alpaca = args.overwrite or not complete_json(alpaca_path, len(alpaca))
    need_arena = args.overwrite or not complete_jsonl(arena_path, len(arena))
    need_mt = args.overwrite or not complete_jsonl(mt_path, len(mt), two_turn=True)
    if not any((need_alpaca, need_arena, need_mt)):
        print(f"[skip] {spec.alias}: all answer files are complete")
        return

    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer_path = spec.tokenizer_path or spec.path
    family, is_instruct = prompt_metadata(spec)
    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True, "use_fast": True}
    if family == "mistral":
        tokenizer_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), **tokenizer_kwargs)
    stop_token_id = resolve_end_token_id(tokenizer, family, is_instruct)
    stop_token_ids = [stop_token_id] if stop_token_id is not None else None
    tp = compatible_tp(spec, args.tensor_parallel_size)
    print(f"[load] {spec.alias} <- {spec.path} (TP={tp})")
    scheduling = {}
    if getattr(args, "max_num_batched_tokens", None) is not None:
        scheduling["max_num_batched_tokens"] = args.max_num_batched_tokens
        scheduling["enable_chunked_prefill"] = True
    llm = LLM(
        model=str(spec.path),
        tokenizer=str(tokenizer_path),
        tensor_parallel_size=tp,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        disable_log_stats=True,
        **scheduling,
    )

    try:
        if need_alpaca:
            prompts = [
                format_messages(tokenizer, spec, [{"role": "user", "content": item["instruction"].strip() or "Empty prompt"}])
                for item in alpaca
            ]
            generated = generate_group(
                llm, prompts, list(range(len(prompts))), 0.0, args.max_new_tokens, 1.05,
                stop_token_ids,
            )
            rows = [
                {
                    "instruction": item["instruction"],
                    "output": generated[i],
                    "generator": spec.alias,
                }
                for i, item in enumerate(alpaca)
            ]
            atomic_json(alpaca_path, rows)
            print(f"[alpaca] wrote {len(rows)} rows to {alpaca_path}")

        if need_arena:
            prompts = [
                format_messages(tokenizer, spec, [{"role": "user", "content": item["prompt"].strip() or "Empty prompt"}])
                for item in arena
            ]
            generated = generate_group(
                llm, prompts, list(range(len(prompts))), 0.0, args.max_new_tokens, 1.05,
                stop_token_ids,
            )
            rows = [
                {
                    "uid": item["uid"],
                    "model_id": spec.alias,
                    "choices": [{"message": {"content": generated[i]}}],
                }
                for i, item in enumerate(arena)
            ]
            atomic_jsonl(arena_path, rows)
            print(f"[arena] wrote {len(rows)} rows to {arena_path}")

        if need_mt:
            first_prompts = [
                format_messages(tokenizer, spec, [{"role": "user", "content": item["turns"][0]}])
                for item in mt
            ]
            first_answers = [""] * len(mt)
            temperatures = sorted(
                {TEMPERATURE_BY_CATEGORY.get(item["category"], 0.7) for item in mt}
            )
            for temperature in temperatures:
                indices = [
                    i
                    for i, item in enumerate(mt)
                    if TEMPERATURE_BY_CATEGORY.get(item["category"], 0.7) == temperature
                ]
                first_answers_map = generate_group(
                    llm, first_prompts, indices, temperature, args.max_new_tokens, 1.0,
                    stop_token_ids,
                )
                for i, text in first_answers_map.items():
                    first_answers[i] = text

            second_prompts = [
                format_messages(
                    tokenizer, spec,
                    [
                        {"role": "user", "content": item["turns"][0]},
                        {"role": "assistant", "content": first_answers[i]},
                        {"role": "user", "content": item["turns"][1]},
                    ],
                )
                for i, item in enumerate(mt)
            ]
            second_answers = [""] * len(mt)
            for temperature in temperatures:
                indices = [
                    i
                    for i, item in enumerate(mt)
                    if TEMPERATURE_BY_CATEGORY.get(item["category"], 0.7) == temperature
                ]
                second_answers_map = generate_group(
                    llm, second_prompts, indices, temperature, args.max_new_tokens, 1.0,
                    stop_token_ids,
                )
                for i, text in second_answers_map.items():
                    second_answers[i] = text

            rows = [
                {
                    "question_id": item["question_id"],
                    "answer_id": f"{spec.alias}-{item['question_id']}",
                    "model_id": spec.alias,
                    "choices": [{"index": 0, "turns": [first_answers[i], second_answers[i]]}],
                    "tstamp": time.time(),
                }
                for i, item in enumerate(mt)
            ]
            atomic_jsonl(mt_path, rows)
            print(f"[mt-bench] wrote {len(rows)} two-turn rows to {mt_path}")
    finally:
        del llm
        gc.collect()
        try:
            import torch
            from vllm.distributed.parallel_state import destroy_model_parallel

            destroy_model_parallel()
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[cleanup warning] {type(exc).__name__}: {exc}")


