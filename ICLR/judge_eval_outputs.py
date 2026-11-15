#!/usr/bin/env python3
"""Judge ICLR entropy-pair outputs and aggregate all requested metrics."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from statistics import mean
from typing import Any

from tqdm import tqdm
import yaml

from eval_common import (
    ALPACA_SOURCE,
    ARENA_BASELINE_SOURCE,
    ARENA_QUESTION_SOURCE,
    ICLR_ROOT,
    MT_QUESTION_SOURCE,
    ModelSpec,
    atomic_write_text,
    discover_models,
    evaluation_root,
    read_jsonl,
    read_jsonl_map,
    safe_name,
    select_models,
    write_manifest,
)


ARENA_SYSTEM_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]"."""

ARENA_USER_TEMPLATE = """<|User Prompt|>
{question}

<|The Start of Assistant A's Answer|>
{answer_a}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_b}
<|The End of Assistant B's Answer|>"""

LABEL_PATTERN = re.compile(r"\[\[([AB<>=]+)\]\]|\[([AB<>=]+)\]")
LABEL_TO_SCORE = {
    "A>B": [1.0],
    "A>>B": [1.0, 1.0, 1.0],
    "A=B": [0.5],
    "A<<B": [0.0, 0.0, 0.0],
    "A<B": [0.0],
    "B>A": [0.0],
    "B>>A": [0.0, 0.0, 0.0],
    "B=A": [0.5],
    "B<<A": [1.0, 1.0, 1.0],
    "B<A": [1.0],
}

ALPACA_SHARED_ASSET_LOCK = threading.Lock()


def clear_cached_judgments(
    specs: list[ModelSpec], args: argparse.Namespace, root: Path
) -> None:
    """Remove only derived judge artifacts before a forced full re-evaluation."""
    arena_dir = root / "arena_hard_v0.1/model_judgment" / safe_name(args.judge_id)
    mt_dir = root / "mt_bench/model_judgment" / safe_name(args.judge_id)
    removed = 0
    for spec in specs:
        for directory in (
            root / "alpaca_results" / spec.alias,
            root / "alpaca_metric_weights" / spec.alias,
        ):
            if directory.exists():
                shutil.rmtree(directory)
                removed += 1
        annotator = root / "alpaca_annotators" / spec.alias
        if annotator.exists():
            for cache_path in annotator.glob("annotations_*.json"):
                cache_path.unlink()
                removed += 1
        for path in (
            arena_dir / f"{spec.alias}.jsonl",
            mt_dir / f"{spec.alias}.jsonl",
        ):
            if path.exists():
                path.unlink()
                removed += 1
    summary_dir = root / "summary"
    if summary_dir.exists():
        shutil.rmtree(summary_dir)
        removed += 1
    print(f"[overwrite] cleared {removed} cached judgment artifacts")


def load_mt_questions(limit: int | None) -> list[dict[str, Any]]:
    rows = []
    for question in read_jsonl(MT_QUESTION_SOURCE):
        turns = question.get("prompt", question.get("turns"))
        rows.append(
            {
                "question_id": question["question_id"],
                "category": question["category"],
                "turns": turns,
            }
        )
    return rows[:limit] if limit else rows


def openai_request(
    messages: list[dict[str, str]],
    args: argparse.Namespace,
    max_tokens: int,
    structured_choices: list[str] | None = None,
) -> tuple[str | None, str | None]:
    from openai import OpenAI

    client = OpenAI(
        api_key=args.judge_api_key,
        base_url=args.judge_api_base,
        timeout=args.judge_timeout,
        max_retries=0,
    )
    error = None
    for attempt in range(1, args.judge_retries + 1):
        try:
            request_kwargs: dict[str, Any] = {}
            if structured_choices:
                # vLLM 0.15 supports choice-constrained decoding through its
                # OpenAI-compatible ``structured_outputs`` extension.  This
                # prevents embedded answer instructions (for example,
                # "output SOLVED") from defeating label-only judge retries.
                request_kwargs["extra_body"] = {
                    "structured_outputs": {"choice": structured_choices}
                }
            response = client.chat.completions.create(
                model=args.judge_model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                **request_kwargs,
            )
            return response.choices[0].message.content or "", None
        except Exception as exc:  # OpenAI-compatible servers expose several exception classes.
            error = f"{type(exc).__name__}: {exc}"
            if attempt < args.judge_retries:
                time.sleep(min(2 * attempt, 10))
    return None, error


def prepare_alpaca_reference(root: Path, limit: int | None) -> Path:
    source = json.loads(ALPACA_SOURCE.read_text(encoding="utf-8"))
    if limit:
        source = source[:limit]
    rows = [
        {
            "instruction": item["instruction"],
            "output": item["output"],
            "generator": "gpt4_1106_preview",
        }
        for item in source
    ]
    path = root / "references/alpaca_eval2_gpt4_1106_preview.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    return path


def prepare_alpaca_hf_cache(assets_dir: Path) -> Path:
    """Expose AlpacaEval's local metric data through the HF cache layout."""
    source = assets_dir / "df_gamed.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Missing AlpacaEval metric asset: {source}")

    # AlpacaEval 0.6.6 ignores ALPACA_EVAL_ASSETS and always calls
    # hf_hub_download("tatsu-lab/alpaca_eval", "df_gamed.csv"). Materialize a
    # tiny local snapshot so that call also works when HF_HUB_OFFLINE is set.
    hub_cache = ICLR_ROOT.parent / ".cache/huggingface/alpaca_eval_hub"
    repository = hub_cache / "datasets--tatsu-lab--alpaca_eval"
    snapshot_name = "local-assets"
    snapshot = repository / "snapshots" / snapshot_name
    snapshot.mkdir(parents=True, exist_ok=True)
    atomic_write_text(snapshot / source.name, source.read_text(encoding="utf-8"))
    refs = repository / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    atomic_write_text(refs / "main", snapshot_name)
    return hub_cache


def run_alpaca(spec: ModelSpec, args: argparse.Namespace, root: Path) -> None:
    model_outputs = root / "alpaca_outputs" / f"{spec.alias}.json"
    if not model_outputs.exists():
        raise FileNotFoundError(f"Missing AlpacaEval answers: {model_outputs}")
    output_dir = root / "alpaca_results" / spec.alias
    leaderboard = output_dir / "leaderboard.csv"
    expected = len(json.loads(ALPACA_SOURCE.read_text(encoding="utf-8")))
    if args.limit:
        expected = min(expected, args.limit)

    def result_count() -> int:
        if not leaderboard.exists():
            return 0
        with leaderboard.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return 0
        raw = rows[-1].get("n_total", rows[-1].get("n", 0))
        try:
            return int(float(raw or 0))
        except (TypeError, ValueError):
            return 0

    existing_count = result_count()
    if existing_count == expected and not args.overwrite_judgments:
        print(f"[alpaca skip] {spec.alias}: complete ({existing_count}/{expected})")
        return
    annotator = root / "alpaca_annotators" / spec.alias
    metric_weights = root / "alpaca_metric_weights" / spec.alias
    if args.overwrite_judgments:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if metric_weights.exists():
            shutil.rmtree(metric_weights)
        if annotator.exists():
            for cache_path in annotator.glob("annotations_*.json"):
                cache_path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    with ALPACA_SHARED_ASSET_LOCK:
        reference = prepare_alpaca_reference(root, args.limit)
    template_dir = ICLR_ROOT / "configs/alpaca_qwen25_14b"
    annotator.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template_dir / "prompt.txt", annotator / "prompt.txt")
    config = yaml.safe_load((template_dir / "configs.yaml").read_text(encoding="utf-8"))
    config["qwen25_14b_judge"]["prompt_template"] = f"{spec.alias}/prompt.txt"
    config["qwen25_14b_judge"]["completions_kwargs"]["num_procs"] = (
        args.alpaca_judge_parallel
    )
    (annotator / "configs.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    def remove_failed_cached_annotations() -> int:
        """Drop only failed cache rows so AlpacaEval retries those examples."""
        removed = 0
        for cache_path in annotator.glob("annotations_*.json"):
            rows = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                continue
            valid = [row for row in rows if row.get("preference") in {1, 1.5, 2}]
            removed += len(rows) - len(valid)
            if len(valid) != len(rows):
                if args.experiment_family == "reviewer20":
                    audit = root / "audit"
                    audit.mkdir(parents=True, exist_ok=True)
                    with (audit / f"alpaca_failed_{spec.alias}.jsonl").open("a", encoding="utf-8") as handle:
                        for row in rows:
                            if row.get("preference") not in {1, 1.5, 2}:
                                handle.write(json.dumps({"source": str(cache_path), "record": row}, ensure_ascii=False) + "\n")
                temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
                temporary.write_text(
                    json.dumps(valid, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                temporary.replace(cache_path)
        return removed

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = args.judge_api_key
    env["OPENAI_API_BASE"] = args.judge_api_base
    env["OPENAI_BASE_URL"] = args.judge_api_base
    assets_dir = Path(
        env.setdefault("ALPACA_EVAL_ASSETS", str(ICLR_ROOT.parent / "alpaca_assets"))
    )
    with ALPACA_SHARED_ASSET_LOCK:
        env["HF_HUB_CACHE"] = str(prepare_alpaca_hf_cache(assets_dir))
    command = [
        args.alpaca_eval_bin,
        "evaluate",
        "--model_outputs", str(model_outputs),
        "--reference_outputs", str(reference),
        "--annotators_config", str(annotator),
        "--output_path", str(output_dir),
        "--precomputed_leaderboard", "None",
        "--is_cache_leaderboard", "False",
        "--metric_kwargs",
        json.dumps({"save_weights_dir": str(metric_weights)}),
    ]
    for completion_pass in range(1, args.judge_completion_passes + 1):
        current_count = result_count()
        if current_count == expected:
            print(f"[alpaca complete] {spec.alias}: {current_count}/{expected}")
            return
        removed = remove_failed_cached_annotations()
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if metric_weights.exists():
            shutil.rmtree(metric_weights)
        print(
            f"[alpaca] {spec.alias}: pass={completion_pass}, "
            f"previous={current_count}/{expected}, retry_cache_rows={removed}"
        )
        print("[alpaca] " + " ".join(command))
        # This AlpacaEval installation resolves its local length-control assets
        # from ``alpaca_assets/`` relative to the process working directory.
        subprocess.run(command, check=True, cwd=str(ICLR_ROOT.parent), env=env)

    final_count = result_count()
    if final_count != expected:
        raise RuntimeError(
            f"Incomplete AlpacaEval judgments for {spec.alias}: "
            f"{final_count}/{expected} after {args.judge_completion_passes} passes"
        )
    print(f"[alpaca complete] {spec.alias}: {final_count}/{expected}")


def extract_arena_label(text: str | None) -> str | None:
    if not text:
        return None
    matches = LABEL_PATTERN.findall(text.upper())
    labels = [(left or right) for left, right in matches if (left or right)]
    return labels[-1] if labels else None


def arena_game(
    question: str,
    answer_a: str,
    answer_b: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    user = ARENA_USER_TEMPLATE.format(
        question=question, answer_a=answer_a, answer_b=answer_b
    )
    messages = [
        {"role": "system", "content": ARENA_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    raw, error = openai_request(messages, args, args.arena_judge_max_tokens)
    score = extract_arena_label(raw)
    repair_raw = None
    repair_error = None
    if score is None:
        # A few long Arena-Hard judgments reach max_tokens before emitting the
        # final label. Retrying the same deterministic prompt repeats that
        # truncation forever, so recover only the missing verdict with a short
        # label-only request over the original question and answers.
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "Compare Assistant A and Assistant B. Return exactly one "
                    "label and no explanation: [[A>>B]], [[A>B]], [[A=B]], "
                    "[[B>A]], or [[B>>A]]."
                ),
            },
            {"role": "user", "content": user},
        ]
        repair_raw, repair_error = openai_request(
            repair_messages,
            args,
            16,
            structured_choices=[
                "[[A>>B]]",
                "[[A>B]]",
                "[[A=B]]",
                "[[B>A]]",
                "[[B>>A]]",
            ],
        )
        score = extract_arena_label(repair_raw)
    return {
        "score": score,
        "judgment": {
            "answer": raw,
            "error": error,
            "repair_answer": repair_raw,
            "repair_error": repair_error,
        },
        "prompt": messages,
    }


def arena_task(
    question: dict[str, Any],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    spec: ModelSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    base_text = baseline["choices"][0]["message"]["content"]
    candidate_text = candidate["choices"][0]["message"]["content"]
    return {
        "uid": question["uid"],
        "category": question["category"],
        "judge": args.judge_id,
        "model": spec.alias,
        "baseline": baseline["model_id"],
        "games": [
            arena_game(question["prompt"], base_text, candidate_text, args),
            arena_game(question["prompt"], candidate_text, base_text, args),
        ],
    }


def valid_arena_record(record: dict[str, Any]) -> bool:
    games = record.get("games", [])
    return len(games) == 2 and all(game and game.get("score") in LABEL_TO_SCORE for game in games)


def compact_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def compact_arena_judgments(path: Path, question_order: list[str]) -> None:
    valid: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if valid_arena_record(row):
            valid[row["uid"]] = row
    compact_jsonl(path, [valid[uid] for uid in question_order if uid in valid])


def run_arena(spec: ModelSpec, args: argparse.Namespace, root: Path) -> None:
    questions = read_jsonl(ARENA_QUESTION_SOURCE)
    if args.limit:
        questions = questions[:args.limit]
    baselines = read_jsonl_map(ARENA_BASELINE_SOURCE, "uid")
    answer_path = root / "arena_hard_v0.1/model_answer" / f"{spec.alias}.jsonl"
    candidates = read_jsonl_map(answer_path, "uid")
    expected = {question["uid"] for question in questions}
    if set(candidates) != expected:
        raise RuntimeError(f"Incomplete Arena-Hard answers: {answer_path}")
    output_path = root / "arena_hard_v0.1/model_judgment" / safe_name(args.judge_id) / f"{spec.alias}.jsonl"
    if args.overwrite_judgments and output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    question_order = [question["uid"] for question in questions]
    for completion_pass in range(1, args.judge_completion_passes + 1):
        existing_rows = read_jsonl(output_path) if output_path.exists() else []
        existing = {row["uid"] for row in existing_rows if valid_arena_record(row)}
        pending = [question for question in questions if question["uid"] not in existing]
        if not pending:
            compact_arena_judgments(output_path, question_order)
            print(f"[arena complete] {spec.alias}: {len(existing)} judgments")
            return
        print(
            f"[arena] {spec.alias}: pass={completion_pass}, "
            f"pending={len(pending)}, parallel={args.judge_parallel}"
        )
        with output_path.open("a", encoding="utf-8") as handle:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.judge_parallel) as pool:
                futures = [
                    pool.submit(
                        arena_task,
                        question,
                        baselines[question["uid"]],
                        candidates[question["uid"]],
                        spec,
                        args,
                    )
                    for question in pending
                ]
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc=f"arena {spec.alias} pass {completion_pass}",
                ):
                    handle.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
                    handle.flush()
        compact_arena_judgments(output_path, question_order)
    final_rows = read_jsonl(output_path) if output_path.exists() else []
    final_valid = {row["uid"] for row in final_rows if valid_arena_record(row)}
    if len(final_valid) == len(questions):
        compact_arena_judgments(output_path, question_order)
        print(f"[arena complete] {spec.alias}: {len(final_valid)} judgments")
        return
    raise RuntimeError(
        f"Incomplete Arena-Hard judgments for {spec.alias}: "
        f"{len(final_valid)}/{len(questions)} after {args.judge_completion_passes} passes"
    )


def mt_messages(
    question: dict[str, Any], answer: dict[str, Any], turn: int
) -> list[dict[str, str]]:
    system = (
        "You are an impartial evaluator for a conversational assistant. "
        "Score only the assistant response for correctness, relevance, helpfulness, "
        "clarity, and self-contained reasoning. Do not reward verbosity. "
        "Use an integer score from 1 (very poor) to 10 (excellent)."
    )
    turns = answer["choices"][0]["turns"]
    if turn == 1:
        content = (
            "Evaluate this first-turn response.\n\n"
            f"[User]\n{question['turns'][0]}\n\n"
            f"[Assistant]\n{turns[0]}\n\n"
            'Return only valid JSON: {"score": <integer from 1 to 10>}.'
        )
    else:
        content = (
            "Evaluate the assistant's second-turn response using the whole conversation.\n\n"
            f"[User turn 1]\n{question['turns'][0]}\n\n"
            f"[Assistant turn 1]\n{turns[0]}\n\n"
            f"[User turn 2]\n{question['turns'][1]}\n\n"
            f"[Assistant turn 2]\n{turns[1]}\n\n"
            'Return only valid JSON: {"score": <integer from 1 to 10>}.'
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def extract_mt_score(text: str | None) -> int | None:
    if not text:
        return None
    standalone = re.fullmatch(r"\s*(10|[1-9])(?:\.0+)?\s*", text)
    if standalone:
        return int(standalone.group(1))
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    for candidate in ([match.group(0)] if match else []) + [text]:
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                continue
            value = parsed.get("score")
            score = int(value)
            if 1 <= score <= 10:
                return score
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    match = re.search(r'"?score"?\s*[:=]\s*(10|[1-9])(?!\d)', text, flags=re.I)
    return int(match.group(1)) if match else None


def mt_task(
    question: dict[str, Any], answer: dict[str, Any], turn: int, args: argparse.Namespace
) -> dict[str, Any]:
    raw, error = openai_request(
        mt_messages(question, answer, turn), args, args.mt_judge_max_tokens
    )
    score = extract_mt_score(raw)
    repair_raw = None
    repair_error = None
    if score is None:
        source_messages = mt_messages(question, answer, turn)
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "You are grading the quoted assistant response, not answering "
                    "the quoted task. Output exactly one integer from 1 to 10 and "
                    "nothing else."
                ),
            },
            {"role": "user", "content": source_messages[-1]["content"]},
        ]
        repair_raw, repair_error = openai_request(
            repair_messages,
            args,
            4,
            structured_choices=[str(value) for value in range(1, 11)],
        )
        score = extract_mt_score(repair_raw)
    return {
        "question_id": question["question_id"],
        "category": question["category"],
        "turn": turn,
        "score": score,
        "status": "ok" if score is not None else ("api_error" if error else "parse_error"),
        "judge_model": args.judge_id,
        "raw_judgment": raw,
        "error": error,
        "repair_judgment": repair_raw,
        "repair_error": repair_error,
    }


def run_mt(spec: ModelSpec, args: argparse.Namespace, root: Path) -> None:
    questions = load_mt_questions(args.limit)
    answer_path = root / "mt_bench/model_answer" / f"{spec.alias}.jsonl"
    answers = read_jsonl_map(answer_path, "question_id")
    expected = {question["question_id"] for question in questions}
    if set(answers) != expected:
        raise RuntimeError(f"Incomplete MT-Bench answers: {answer_path}")
    output_path = root / "mt_bench/model_judgment" / safe_name(args.judge_id) / f"{spec.alias}.jsonl"
    if args.overwrite_judgments and output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_keys = [
        (question["question_id"], turn) for question in questions for turn in (1, 2)
    ]
    for completion_pass in range(1, args.judge_completion_passes + 1):
        existing_rows = read_jsonl(output_path) if output_path.exists() else []
        valid = {
            (row["question_id"], row["turn"]): row
            for row in existing_rows
            if row.get("status") == "ok"
            and isinstance(row.get("score"), int)
            and 1 <= row["score"] <= 10
        }
        tasks = [
            (question, turn)
            for question in questions
            for turn in (1, 2)
            if (question["question_id"], turn) not in valid
        ]
        if not tasks:
            compact_jsonl(output_path, [valid[key] for key in ordered_keys])
            print(f"[mt complete] {spec.alias}: {len(valid)} scores")
            return
        print(
            f"[mt] {spec.alias}: pass={completion_pass}, "
            f"pending={len(tasks)}, parallel={args.judge_parallel}"
        )
        with output_path.open("a", encoding="utf-8") as handle:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.judge_parallel) as pool:
                futures = [
                    pool.submit(
                        mt_task,
                        question,
                        answers[question["question_id"]],
                        turn,
                        args,
                    )
                    for question, turn in tasks
                ]
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc=f"mt {spec.alias} pass {completion_pass}",
                ):
                    handle.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
                    handle.flush()
        compact_jsonl(
            output_path,
            [
                row
                for row in read_jsonl(output_path)
                if row.get("status") == "ok"
                and isinstance(row.get("score"), int)
                and 1 <= row["score"] <= 10
            ],
        )
    final_rows = read_jsonl(output_path) if output_path.exists() else []
    final_valid = {
        (row["question_id"], row["turn"]): row
        for row in final_rows
        if row.get("status") == "ok"
        and isinstance(row.get("score"), int)
        and 1 <= row["score"] <= 10
    }
    if len(final_valid) == len(ordered_keys):
        compact_jsonl(output_path, [final_valid[key] for key in ordered_keys])
        print(f"[mt complete] {spec.alias}: {len(final_valid)} scores")
        return
    raise RuntimeError(
        f"Incomplete MT-Bench judgments for {spec.alias}: "
        f"{len(final_valid)}/{len(ordered_keys)} after {args.judge_completion_passes} passes"
    )


def bootstrap_ci(scores: list[float], n_boot: int, seed: int = 42) -> tuple[float, float, float]:
    if not scores:
        return math.nan, math.nan, math.nan
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [scores[rng.randrange(len(scores))] for _ in scores]
        means.append(mean(sample))
    means.sort()
    return mean(scores), means[int(0.05 * (n_boot - 1))], means[int(0.95 * (n_boot - 1))]


def arena_scores(records: list[dict[str, Any]]) -> list[float]:
    scores: list[float] = []
    for record in records:
        if not valid_arena_record(record):
            continue
        first, second = record["games"]
        scores.extend(LABEL_TO_SCORE[second["score"]])
        scores.extend(1.0 - value for value in LABEL_TO_SCORE[first["score"]])
    return scores


def alpaca_metrics(spec: ModelSpec, root: Path) -> tuple[str, str, str]:
    path = root / "alpaca_results" / spec.alias / "leaderboard.csv"
    if not path.exists():
        return "N/A", "N/A", "0"
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return "N/A", "N/A", "0"
    row = rows[-1]
    return (
        row.get("length_controlled_winrate", "N/A"),
        row.get("win_rate", "N/A"),
        row.get("n_total", row.get("n", "805")),
    )


def validate_evaluation_complete(
    specs: list[ModelSpec], args: argparse.Namespace, root: Path
) -> None:
    alpaca_expected = len(json.loads(ALPACA_SOURCE.read_text(encoding="utf-8")))
    arena_expected = len(read_jsonl(ARENA_QUESTION_SOURCE))
    mt_expected = 2 * len(load_mt_questions(args.limit))
    if args.limit:
        alpaca_expected = min(alpaca_expected, args.limit)
        arena_expected = min(arena_expected, args.limit)

    incomplete: list[str] = []
    for spec in specs:
        _, _, alpaca_n = alpaca_metrics(spec, root)
        try:
            alpaca_complete = int(float(alpaca_n)) == alpaca_expected
        except (TypeError, ValueError):
            alpaca_complete = False

        arena_path = (
            root
            / "arena_hard_v0.1/model_judgment"
            / safe_name(args.judge_id)
            / f"{spec.alias}.jsonl"
        )
        arena_records = read_jsonl(arena_path) if arena_path.exists() else []
        arena_valid = {
            row.get("uid") for row in arena_records if valid_arena_record(row)
        }

        mt_path = (
            root
            / "mt_bench/model_judgment"
            / safe_name(args.judge_id)
            / f"{spec.alias}.jsonl"
        )
        mt_records = read_jsonl(mt_path) if mt_path.exists() else []
        mt_valid = {
            (row.get("question_id"), row.get("turn"))
            for row in mt_records
            if row.get("status") == "ok"
            and isinstance(row.get("score"), int)
            and 1 <= row["score"] <= 10
        }
        if not (
            alpaca_complete
            and len(arena_valid) == arena_expected
            and len(mt_valid) == mt_expected
        ):
            incomplete.append(
                f"{spec.alias}: alpaca={alpaca_n}/{alpaca_expected}, "
                f"arena={len(arena_valid)}/{arena_expected}, "
                f"mt={len(mt_valid)}/{mt_expected}"
            )
    if incomplete:
        raise RuntimeError(
            "Refusing to write final metrics; incomplete evaluations:\n"
            + "\n".join(incomplete)
        )
    print(
        f"[validation] complete: models={len(specs)}, "
        f"alpaca={alpaca_expected}, arena={arena_expected}, mt={mt_expected} each"
    )


def write_summary(specs: list[ModelSpec], args: argparse.Namespace, root: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        lc, alpaca_wr, alpaca_n = alpaca_metrics(spec, root)
        arena_path = root / "arena_hard_v0.1/model_judgment" / safe_name(args.judge_id) / f"{spec.alias}.jsonl"
        arena_records = read_jsonl(arena_path) if arena_path.exists() else []
        scores = arena_scores(arena_records)
        arena_mean, arena_lo, arena_hi = bootstrap_ci(scores, args.bootstrap_samples)
        mt_path = root / "mt_bench/model_judgment" / safe_name(args.judge_id) / f"{spec.alias}.jsonl"
        mt_records = read_jsonl(mt_path) if mt_path.exists() else []
        turn1 = [row["score"] for row in mt_records if row.get("status") == "ok" and row.get("turn") == 1]
        turn2 = [row["score"] for row in mt_records if row.get("status") == "ok" and row.get("turn") == 2]
        both = turn1 + turn2
        rows.append(
            {
                "model": spec.alias,
                "run_name": spec.run_name,
                "chosen_region": spec.chosen_region,
                "rejected_region": spec.rejected_region,
                "selection_fraction": f"{spec.fraction:.2f}",
                "rest_weight": (
                    f"{spec.rest_weight:.6g}" if spec.rest_weight is not None else "N/A"
                ),
                "decomposition_mode": spec.decomposition_mode or "N/A",
                "gate_fn": spec.gate_fn or "N/A",
                "gate_scope": spec.gate_scope or "N/A",
                "model_size": spec.model_size or "N/A",
                "method": spec.method or "N/A",
                "beta": f"{spec.beta:g}" if spec.beta is not None else "N/A",
                "alpaca_lc": lc,
                "alpaca_wr": alpaca_wr,
                "alpaca_n": alpaca_n,
                "arena_wr": f"{arena_mean * 100:.3f}" if scores else "N/A",
                "arena_ci_low": f"{arena_lo * 100:.3f}" if scores else "N/A",
                "arena_ci_high": f"{arena_hi * 100:.3f}" if scores else "N/A",
                "arena_n_scores": len(scores),
                "mt_turn1": f"{mean(turn1):.3f}" if turn1 else "N/A",
                "mt_turn2": f"{mean(turn2):.3f}" if turn2 else "N/A",
                "mt_overall": f"{mean(both):.3f}" if both else "N/A",
                "mt_n_scores": len(both),
                "judge": args.judge_id,
            }
        )
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_stem = {
        "reviewer20": "reviewer20_metrics",
        "pair": "entropy_pair_metrics",
        "pair_40_90": "entropy_pair_40_90_metrics",
        "band_30_50": "entropy_band_30_50_metrics",
        "band_50_90": "entropy_band_50_90_metrics",
        "mixed_band_00_90": "mixed_entropy_band_00_90_metrics",
        "reference_beta": "reference_low20_beta_metrics",
        "policy_overlap": "policy_low20_overlap_metrics",
        "residual": "residual_weight_metrics",
        "decomposition": "gradient_decomposition_metrics",
        "preserve": "preserve_low20_metrics",
        "gate": "policy_low_gate_metrics",
        "low20_gate": "policy_low20_gate_metrics",
        "gate_combined": "policy_gate_metrics",
        "tdpo_tisdpo": "tdpo_tisdpo_metrics",
        "llama_mistral_preference": "llama_mistral_preference_metrics",
        "qwen25_7b_gating_gamma": "gating_gamma_ablation_metrics",
        "mistral_7b_instruct_gating_gamma": "gating_gamma_ablation_metrics",
        "llama31_8b_instruct_gating_gamma": "gating_gamma_ablation_metrics",
    }[args.experiment_family]
    csv_path = summary_dir / f"{summary_stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = summary_dir / f"{summary_stem}.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text_path = summary_dir / "final_metrics.txt"
    text_columns = [
        "model",
        "beta",
        "alpaca_lc",
        "alpaca_wr",
        "alpaca_n",
        "arena_wr",
        "arena_ci_low",
        "arena_ci_high",
        "arena_n_scores",
        "mt_turn1",
        "mt_turn2",
        "mt_overall",
        "mt_n_scores",
        "judge",
    ]
    text_lines = [
        "Complete evaluation metrics",
        f"Experiment: {args.experiment_family}",
        "\t".join(text_columns),
    ]
    text_lines.extend(
        "\t".join(str(row[column]) for column in text_columns) for row in rows
    )
    text_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    print(f"[summary] {csv_path}")
    print(f"[summary] {text_path}")
    for row in rows:
        print(
            f"{row['model']} | Alpaca LC={row['alpaca_lc']} WR={row['alpaca_wr']} | "
            f"Arena={row['arena_wr']} | MT={row['mt_overall']}"
        )
    return csv_path


def run_model_stage(
    stage: str,
    runner: Any,
    specs: list[ModelSpec],
    args: argparse.Namespace,
    root: Path,
) -> None:
    if not specs:
        return
    workers = min(args.judge_parallel_models, len(specs))
    if workers < 1:
        raise ValueError("--judge-parallel-models must be at least 1")

    stage_args = argparse.Namespace(**vars(args))
    if workers > 1:
        # The CLI concurrency values are global budgets. Divide them between
        # concurrent model jobs so model-level parallelism does not multiply
        # the load on the shared vLLM judge endpoint.
        stage_args.judge_parallel = max(1, math.ceil(args.judge_parallel / workers))
        stage_args.alpaca_judge_parallel = max(
            1, math.ceil(args.alpaca_judge_parallel / workers)
        )
    print(
        f"[{stage}] model_parallel={workers}, "
        f"request_parallel_per_model={stage_args.judge_parallel}, "
        f"alpaca_parallel_per_model={stage_args.alpaca_judge_parallel}"
    )

    def run_one(index: int, spec: ModelSpec) -> None:
        print(f"[{stage}] model {index}/{len(specs)}: {spec.alias}")
        runner(spec, stage_args, root)

    if workers == 1:
        for index, spec in enumerate(specs, 1):
            run_one(index, spec)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_one, index, spec)
            for index, spec in enumerate(specs, 1)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


