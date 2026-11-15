#!/usr/bin/env python3
"""Estimate TIS-DPO token weights from forward and reverse DPO adapters."""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import AutoModelForCausalLM

from src.common.model_utils import (
    detect_model_config,
    dpo_collator,
    load_dpo_dataset,
    load_tokenizer,
    resolve_end_token_id,
)


FORWARD_ADAPTER_NAME = "tis_forward_dpo"
REVERSE_ADAPTER_NAME = "tis_reverse_dpo"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--forward_adapter", required=True)
    parser.add_argument("--reverse_adapter", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--round_digits", type=int, default=2)
    parser.add_argument("--distributed_timeout_minutes", type=int, default=120)
    return parser.parse_args()


def target_token_logps(model, batch, side):
    output = model(
        input_ids=batch[f"{side}_input_ids"],
        attention_mask=batch[f"{side}_attention_mask"],
        use_cache=False,
    )
    logits = output.logits[:, :-1, :]
    labels = batch[f"{side}_labels"][:, 1:].clone()
    mask = labels.ne(-100)
    gather_ids = labels.masked_fill(~mask, 0).unsqueeze(-1)
    target_logits = torch.gather(logits, -1, gather_ids).squeeze(-1).float()
    log_normalizer = torch.logsumexp(logits.float(), dim=-1)
    return (target_logits - log_normalizer) * mask.float(), mask


def extract_response_weights(logp_difference, mask, digits):
    all_weights = []
    for row_difference, row_mask in zip(logp_difference, mask):
        values = row_difference[row_mask].float().cpu().tolist()
        # The local pipeline appends EOS; TIS estimates response-text weights only.
        values = values[:-1] if values else []
        all_weights.append([round(float(value), digits) for value in values])
    return all_weights


def token_balanced_assignments(dataset, world_size):
    """Assign examples greedily so each rank receives a similar token count."""
    token_counts = [
        len(example["chosen_input_ids"]) + len(example["rejected_input_ids"])
        for example in dataset
    ]
    assignments = [[] for _ in range(world_size)]
    token_loads = [0] * world_size
    example_counts = [0] * world_size
    for index in sorted(range(len(dataset)), key=lambda item: (-token_counts[item], item)):
        rank = min(
            range(world_size),
            key=lambda worker: (token_loads[worker], example_counts[worker], worker),
        )
        assignments[rank].append(index)
        token_loads[rank] += token_counts[index]
        example_counts[rank] += 1
    for indices in assignments:
        indices.sort()
    return assignments, token_loads


def recover_records(paths, allowed_indices):
    """Recover valid records for one rank from current and legacy shard files."""
    recovered = {}
    allowed_indices = set(allowed_indices)
    visited = set()
    for path in paths:
        path = Path(path)
        if path in visited or not path.is_file():
            continue
        visited.add(path)
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    record = json.loads(line)
                    index = record["index"]
                    record["chosen_weight"]
                    record["rejected_weight"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    print(f"[weights] ignoring invalid record {path}:{line_number}")
                    continue
                if index in allowed_indices and index not in recovered:
                    recovered[index] = record
    return recovered


def merge_shards(source_path, shard_paths, output_path, expected_count, owner_by_index):
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    readers = [path.open("r", encoding="utf-8") for path in shard_paths]
    try:
        with Path(source_path).open("r", encoding="utf-8") as source, temporary_path.open(
            "w", encoding="utf-8"
        ) as destination:
            count = 0
            for index, source_line in enumerate(source):
                rank = owner_by_index[index]
                weight_line = readers[rank].readline()
                if not weight_line:
                    raise RuntimeError(f"Weight shard {rank} ended before example {index}")
                record = json.loads(weight_line)
                if record["index"] != index:
                    raise RuntimeError(
                        f"Weight order mismatch: expected {index}, got {record['index']}"
                    )
                example = json.loads(source_line)
                example["chosen_weight"] = record["chosen_weight"]
                example["rejected_weight"] = record["rejected_weight"]
                destination.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
        if count != expected_count:
            raise RuntimeError(f"Merged {count} examples, expected {expected_count}")
        for rank, reader in enumerate(readers):
            if reader.readline():
                raise RuntimeError(f"Weight shard {rank} contains extra records")
        os.replace(temporary_path, output_path)
    finally:
        for reader in readers:
            reader.close()
        if temporary_path.exists():
            temporary_path.unlink()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive")
    if args.distributed_timeout_minutes < 1:
        raise ValueError("--distributed_timeout_minutes must be positive")
    for name in ("model_id", "dataset_name", "forward_adapter", "reverse_adapter"):
        value = getattr(args, name)
        if not Path(value).exists():
            raise FileNotFoundError(f"--{name} does not exist: {value}")

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    output_dir = Path(args.output_dir).resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    model_family, is_instruct = detect_model_config(args.model_id)
    tokenizer = load_tokenizer(args.model_id, model_family)
    end_token_id = resolve_end_token_id(tokenizer, model_family, is_instruct)

    # Rank zero creates/reuses the dataset cache before concurrent readers enter.
    if rank == 0:
        dataset = load_dpo_dataset(
            args.dataset_name,
            tokenizer,
            model_family,
            is_instruct,
            end_token_id,
            max_len=args.dataset_max_length,
        )
    if distributed:
        dist.barrier()
    if rank != 0:
        dataset = load_dpo_dataset(
            args.dataset_name,
            tokenizer,
            model_family,
            is_instruct,
            end_token_id,
            max_len=args.dataset_max_length,
        )

    assignments, token_loads = token_balanced_assignments(dataset, world_size)
    indices = assignments[rank]
    owner_by_index = [0] * len(dataset)
    for worker, worker_indices in enumerate(assignments):
        for index in worker_indices:
            owner_by_index[index] = worker

    # Keep balanced shards separate so legacy modulo shards can be reused safely.
    shard_path = output_dir / f"weights_balanced_rank{rank:02d}.jsonl"
    shard_tmp = shard_path.with_suffix(".jsonl.tmp")
    legacy_paths = sorted(output_dir.glob("weights_rank*.jsonl"))
    legacy_paths += sorted(output_dir.glob("weights_rank*.jsonl.tmp"))
    recovered = recover_records([shard_path, shard_tmp, *legacy_paths], indices)
    missing_indices = [index for index in indices if index not in recovered]
    print(
        f"[weights] rank {rank}: recovered {len(recovered)}/{len(indices)} examples; "
        f"assigned_tokens={token_loads[rank]}; remaining={len(missing_indices)}"
    )

    model = None
    if missing_indices:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            base_model,
            args.forward_adapter,
            adapter_name=FORWARD_ADAPTER_NAME,
            is_trainable=False,
        )
        model.load_adapter(
            args.reverse_adapter,
            adapter_name=REVERSE_ADAPTER_NAME,
            is_trainable=False,
        )
        model.to(device)
        model.eval()

    recovered_tmp = shard_path.with_suffix(".jsonl.recovered.tmp")
    with recovered_tmp.open("w", encoding="utf-8") as output_file:
        for index in indices:
            if index in recovered:
                output_file.write(json.dumps(recovered[index], ensure_ascii=False) + "\n")
        output_file.flush()
    os.replace(recovered_tmp, shard_tmp)

    with shard_tmp.open("a", encoding="utf-8") as output_file, torch.inference_mode():
        for start in range(0, len(missing_indices), args.batch_size):
            batch_indices = missing_indices[start : start + args.batch_size]
            features = [dataset[index] for index in batch_indices]
            batch = {
                key: value.to(device)
                for key, value in dpo_collator(features, tokenizer.pad_token_id).items()
            }
            model.set_adapter(FORWARD_ADAPTER_NAME)
            forward_chosen, chosen_mask = target_token_logps(model, batch, "chosen")
            forward_rejected, rejected_mask = target_token_logps(model, batch, "rejected")
            model.set_adapter(REVERSE_ADAPTER_NAME)
            reverse_chosen, _ = target_token_logps(model, batch, "chosen")
            reverse_rejected, _ = target_token_logps(model, batch, "rejected")
            chosen_weights = extract_response_weights(
                reverse_chosen - forward_chosen, chosen_mask, args.round_digits
            )
            rejected_weights = extract_response_weights(
                reverse_rejected - forward_rejected, rejected_mask, args.round_digits
            )
            for index, chosen_weight, rejected_weight in zip(
                batch_indices, chosen_weights, rejected_weights
            ):
                record = {
                    "index": index,
                    "chosen_weight": chosen_weight,
                    "rejected_weight": rejected_weight,
                }
                recovered[index] = record
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            completed = len(recovered)
            should_report = (
                start == 0
                or start + args.batch_size >= len(missing_indices)
                or start % (args.batch_size * 50) == 0
            )
            if should_report:
                output_file.flush()
            if rank == 0 and should_report:
                print(
                    f"[weights] rank 0: {completed}/"
                    f"{len(indices)} local examples"
                )

    if len(recovered) != len(indices):
        raise RuntimeError(
            f"Rank {rank} produced {len(recovered)} records, expected {len(indices)}"
        )
    sorted_tmp = shard_path.with_suffix(".jsonl.sorted.tmp")
    with sorted_tmp.open("w", encoding="utf-8") as output_file:
        for index in indices:
            output_file.write(json.dumps(recovered[index], ensure_ascii=False) + "\n")
    os.replace(sorted_tmp, shard_path)
    shard_tmp.unlink(missing_ok=True)

    if distributed:
        dist.barrier()
    if rank == 0:
        shard_paths = [
            output_dir / f"weights_balanced_rank{worker:02d}.jsonl"
            for worker in range(world_size)
        ]
        weighted_dataset_path = output_dir / "train.weighted.jsonl"
        merge_shards(
            args.dataset_name,
            shard_paths,
            weighted_dataset_path,
            len(dataset),
            owner_by_index,
        )
        metadata = {
            "estimator": "log_p_reverse(token)-log_p_forward(token)",
            "chosen_transform_during_training": "negate_then_rank_based",
            "rejected_transform_during_training": "rank_based",
            "round_digits": args.round_digits,
            "examples": len(dataset),
            "world_size": world_size,
            "shard_strategy": "greedy_chosen_plus_rejected_token_balance",
            "estimated_tokens_per_rank": token_loads,
            "distributed_timeout_minutes": args.distributed_timeout_minutes,
            "model_id": str(Path(args.model_id).resolve()),
            "forward_adapter": str(Path(args.forward_adapter).resolve()),
            "reverse_adapter": str(Path(args.reverse_adapter).resolve()),
            "source_dataset": str(Path(args.dataset_name).resolve()),
            "weighted_dataset": str(weighted_dataset_path),
        }
        (output_dir / "weight_estimation_config.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")
        print(f"Saved weighted dataset to {weighted_dataset_path}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
