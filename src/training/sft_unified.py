#!/usr/bin/env python3
"""Distributed LoRA SFT with a merged-model, restart-safe output contract.

The chosen UltraFeedback response is used as the supervised target. Prompt
and response tokenization deliberately reuses the preference-training path so
that SFT, DPO and entropy-selected DPO differ in objective rather than format.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from peft import PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed

from src.common.model_utils import (
    build_chat_prompt,
    detect_model_config,
    encode_response,
    get_lora_config,
    load_tokenizer,
    pad_1d,
    resolve_end_token_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--deepspeed", required=True)
    parser.add_argument("--dataset_max_length", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--keep_temp_lora", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("model_id", "dataset_name", "deepspeed"):
        if not Path(getattr(args, name)).exists():
            raise FileNotFoundError(f"--{name} does not exist: {getattr(args, name)}")
    if args.dataset_max_length < 2:
        raise ValueError("--dataset_max_length must be at least 2")
    if args.per_device_train_batch_size < 1:
        raise ValueError("--per_device_train_batch_size must be positive")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be positive")


def write_run_config(args: argparse.Namespace, output_dir: Path) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    config = vars(args).copy()
    config.update(
        {
            "method": "sft",
            "supervised_target": "chosen",
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "save_format": "standalone_merged_model",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    adapter_path = output_dir / "temp_lora"
    final_model_path = output_dir / "model"
    write_run_config(args, output_dir)

    model_family, is_instruct = detect_model_config(args.model_id)
    tokenizer = load_tokenizer(args.model_id, model_family)
    end_token_id = resolve_end_token_id(tokenizer, model_family, is_instruct)
    raw_dataset = load_dataset("json", data_files=args.dataset_name, split="train")

    def tokenize(example):
        prompt = str(example["prompt"]).strip() or "Empty prompt"
        prompt_text = build_chat_prompt(
            prompt,
            tokenizer,
            model_family,
            is_instruct,
            messages=example.get("messages"),
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids, attention_mask, labels = encode_response(
            str(example["chosen"]),
            tokenizer,
            prompt_ids,
            end_token_id,
            model_family,
            is_instruct,
            max_len=args.dataset_max_length,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    dataset = raw_dataset.map(
        tokenize,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing chosen responses for SFT",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = get_peft_model(model, get_lora_config())
    model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=str(adapter_path),
        logging_dir=str(output_dir / "logs"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        report_to="tensorboard",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        save_strategy="no",
        deepspeed=args.deepspeed,
        seed=args.seed,
        data_seed=args.seed,
        run_name=args.experiment_name,
        ddp_find_unused_parameters=False,
    )
    # Trainer needs the ordinary CausalLM field names. Keep this explicit to
    # avoid relying on a Seq2Seq collator that might alter masked labels.
    def sft_collator(features):
        return {
            "input_ids": pad_1d(
                [item["input_ids"] for item in features],
                tokenizer.pad_token_id,
                "right",
            ),
            "attention_mask": pad_1d(
                [item["attention_mask"] for item in features], 0, "right"
            ),
            "labels": pad_1d(
                [item["labels"] for item in features], -100, "right"
            ),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=sft_collator,
    )
    if trainer.is_world_process_zero():
        print(
            f"Training {args.experiment_name}: SFT on chosen responses, "
            f"model={args.model_id}, world_size={trainer.args.world_size}"
        )
    trainer.train()
    trainer.save_model(str(adapter_path))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(adapter_path)
    trainer.accelerator.wait_for_everyone()

    if trainer.is_world_process_zero():
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        merged_model = PeftModel.from_pretrained(
            base_model, adapter_path
        ).merge_and_unload()
        merged_model.save_pretrained(final_model_path, safe_serialization=True)
        tokenizer.save_pretrained(final_model_path)
        if not args.keep_temp_lora:
            shutil.rmtree(adapter_path)
        (output_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")
        print(f"Saved merged SFT model to {final_model_path}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
