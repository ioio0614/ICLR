#!/usr/bin/env python3
"""Unified DPO/IPO/CPO/SimPO/KTO/TDPO/TIS-DPO/ConfPO training.

The baseline objectives are consolidated from the former KBS trainer, while
TDPO/TIS-DPO and TIS weight handling are consolidated from the former ICLR
trainer.  The public launcher is ``scripts/run_llama_mistral_preference.sh``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, set_seed
from trl import DPOConfig, DPOTrainer

from src.common.model_utils import (
    detect_model_config,
    dpo_collator,
    get_lora_config,
    load_dpo_dataset,
    load_tokenizer,
    resolve_end_token_id,
    tokenize_dpo_example,
)
from src.training.entropy_logging import EntropyHistoryCallback


PUBLIC_METHODS = ("dpo", "ipo", "cpo", "simpo", "kto", "tdpo", "tis_dpo", "confpo")
INTERNAL_METHODS = ("contrast_forward", "contrast_reverse")
ALL_METHODS = PUBLIC_METHODS + INTERNAL_METHODS


def _rank_based(values, negate=False, min_scale=0.7, max_scale=1.3):
    """Apply the TIS-DPO per-response rank transform, stable for ties."""
    values = [-float(value) for value in values] if negate else [float(value) for value in values]
    count = len(values)
    if count == 0:
        return []
    if count == 1:
        return [1.0]
    order = sorted(range(count), key=lambda index: values[index])
    transformed = [0.0] * count
    for rank, index in enumerate(order):
        transformed[index] = min_scale + rank / (count - 1) * (max_scale - min_scale)
    return transformed


def _attach_full_weights(example, tokenized, min_scale, max_scale):
    result = dict(tokenized)
    for side, negate in (("chosen", True), ("rejected", False)):
        transformed = _rank_based(
            example[f"{side}_weight"],
            negate=negate,
            min_scale=min_scale,
            max_scale=max_scale,
        )
        labels = result[f"{side}_labels"]
        valid_positions = [index for index, label in enumerate(labels) if label != -100]
        expected = max(len(valid_positions) - 1, 0)  # EOS always has zero TIS weight.
        if len(transformed) != expected:
            raise ValueError(
                f"{side} weight/token mismatch: got {len(transformed)}, expected {expected}"
            )
        full_weights = [0.0] * len(labels)
        for position, value in zip(valid_positions[:-1], transformed):
            full_weights[position] = value
        result[f"{side}_weight"] = full_weights
    return result


def load_tis_dataset(
    weighted_dataset_path,
    tokenizer,
    model_family,
    is_instruct,
    end_token_id,
    max_len,
    min_scale,
    max_scale,
):
    dataset = load_dataset("json", data_files=weighted_dataset_path, split="train")

    def tokenize(example):
        tokenized = tokenize_dpo_example(
            example,
            tokenizer,
            model_family,
            is_instruct,
            end_token_id,
            max_len=max_len,
        )
        return _attach_full_weights(example, tokenized, min_scale, max_scale)

    return dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        desc="Tokenizing weighted TIS-DPO dataset",
    )


def weighted_dpo_collator(features, pad_token_id):
    batch = dpo_collator(features, pad_token_id)
    for side in ("chosen", "rejected"):
        key = f"{side}_weight"
        max_len = max(len(feature[key]) for feature in features)
        batch[key] = torch.tensor(
            [feature[key] + [0.0] * (max_len - len(feature[key])) for feature in features],
            dtype=torch.float32,
        )
    return batch


def reverse_preferences(dataset):
    """Swap tokenized chosen/rejected fields for the reverse TIS contrast model."""

    def swap(example):
        output = {}
        for suffix in ("input_ids", "attention_mask", "labels"):
            output[f"chosen_{suffix}"] = example[f"rejected_{suffix}"]
            output[f"rejected_{suffix}"] = example[f"chosen_{suffix}"]
        return output

    return dataset.map(swap, desc="Reversing preference labels")


class UnifiedPreferenceTrainer(DPOTrainer):
    """One trainer with explicit implementations of all supported objectives."""

    def __init__(
        self,
        *args,
        method,
        cpo_alpha=1.0,
        simpo_gamma=0.5,
        confpo_temperature=1.0,
        tdpo_alpha=0.5,
        tdpo_version=2,
        vocab_chunk_size=16384,
        **kwargs,
    ):
        if method not in ALL_METHODS:
            raise ValueError(f"Unsupported method: {method}")
        if tdpo_version not in {1, 2}:
            raise ValueError("tdpo_version must be 1 or 2")
        self.method = method
        self.cpo_alpha = float(cpo_alpha)
        self.simpo_gamma = float(simpo_gamma)
        self.confpo_temperature = float(confpo_temperature)
        self.tdpo_alpha = float(tdpo_alpha)
        self.tdpo_version = int(tdpo_version)
        self.vocab_chunk_size = int(vocab_chunk_size)
        super().__init__(*args, **kwargs)

    def _prepare_dataset(self, dataset, processing_class, args, stage):
        return dataset

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, metrics = self.get_batch_loss_metrics(model, inputs)
        if self.is_in_train:
            # Reduce each micro-batch metric across workers, then let DPOTrainer
            # average the stored values over the gradient-accumulation window.
            # The base Trainer already logs the optimizer-step loss, so do not
            # overwrite it with a separately accumulated copy.
            reduced_metrics = {}
            for name, value in metrics.items():
                if name == "loss":
                    continue
                if isinstance(value, torch.Tensor):
                    value = self.accelerator.reduce(
                        value.detach().float(), reduction="mean"
                    ).item()
                reduced_metrics[name] = value
            self.store_metrics(reduced_metrics, train_eval="train")
        outputs = {"loss": loss.detach(), **metrics}
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _token_logps_from_logits(logits, labels):
        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:]
        logps = F.log_softmax(shifted_logits, dim=-1)
        gather_ids = shifted_labels.clamp_min(0).unsqueeze(-1)
        token_logps = torch.gather(logps, dim=-1, index=gather_ids).squeeze(-1)
        mask = shifted_labels.ne(-100)
        token_logps = token_logps * mask.to(token_logps.dtype)
        return F.pad(token_logps, (1, 0), value=0.0), labels.ne(-100)

    def _forward_token_logps(self, model, input_ids, attention_mask, labels):
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return self._token_logps_from_logits(output.logits, labels)

    @staticmethod
    def _token_entropy_from_logits(logits, labels, vocab_chunk_size):
        """Compute response-token entropy in float32 without a full fp32 vocab copy."""
        shifted_logits = logits[:, :-1, :].detach()
        shifted_labels = labels[:, 1:]
        mask = shifted_labels.ne(-100)

        # H(p) = log Z - E_p[logit].  Chunking over the vocabulary keeps the
        # extra float32 working set bounded for Qwen's large vocabulary.
        max_logits = shifted_logits.max(dim=-1).values.float()
        exp_sum = torch.zeros_like(max_logits)
        exp_logit_sum = torch.zeros_like(max_logits)
        vocab_size = shifted_logits.size(-1)
        for start in range(0, vocab_size, vocab_chunk_size):
            end = min(start + vocab_chunk_size, vocab_size)
            chunk = shifted_logits[..., start:end].float()
            shifted_exp = torch.exp(chunk - max_logits.unsqueeze(-1))
            exp_sum.add_(shifted_exp.sum(dim=-1))
            exp_logit_sum.add_((shifted_exp * chunk).sum(dim=-1))

        entropy = max_logits + exp_sum.log() - exp_logit_sum / exp_sum
        entropy = entropy.clamp_min_(0.0) * mask.float()
        return F.pad(entropy, (1, 0), value=0.0), labels.ne(-100)

    def _forward_token_logps_and_entropy(self, model, input_ids, attention_mask, labels):
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        token_logps, mask = self._token_logps_from_logits(output.logits, labels)
        with torch.no_grad():
            entropy, _ = self._token_entropy_from_logits(
                output.logits, labels, self.vocab_chunk_size
            )
        return token_logps, entropy, mask

    def _reference_token_logps(self, model, batch):
        with torch.no_grad():
            reference_context = model.disable_adapter() if self.ref_model is None else nullcontext()
            with reference_context:
                reference_model = model if self.ref_model is None else self.ref_model
                chosen, _ = self._forward_token_logps(
                    reference_model,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["chosen_labels"],
                )
                rejected, _ = self._forward_token_logps(
                    reference_model,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch["rejected_labels"],
                )
        return chosen, rejected

    @staticmethod
    def _sum_and_count(token_logps, mask):
        mask_f = mask.to(token_logps.dtype)
        return (token_logps * mask_f).sum(-1), mask_f.sum(-1).clamp_min(1.0)

    def _policy_pair(self, model, batch):
        chosen, chosen_mask = self._forward_token_logps(
            model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_labels"],
        )
        rejected, rejected_mask = self._forward_token_logps(
            model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_labels"],
        )
        return chosen, rejected, chosen_mask, rejected_mask

    def _policy_pair_with_entropy(self, model, batch):
        chosen, chosen_entropy, chosen_mask = self._forward_token_logps_and_entropy(
            model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_labels"],
        )
        rejected, rejected_entropy, rejected_mask = self._forward_token_logps_and_entropy(
            model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_labels"],
        )
        return (
            chosen,
            rejected,
            chosen_entropy,
            rejected_entropy,
            chosen_mask,
            rejected_mask,
        )

    @staticmethod
    def _masked_mean(values, mask):
        mask_f = mask.to(values.dtype)
        return (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)

    def _standard_dpo(self, model, batch):
        beta = self.args.beta
        (
            chosen,
            rejected,
            chosen_entropy,
            rejected_entropy,
            chosen_mask,
            rejected_mask,
        ) = self._policy_pair_with_entropy(model, batch)
        ref_chosen, ref_rejected = self._reference_token_logps(model, batch)
        chosen_sum, _ = self._sum_and_count(chosen - ref_chosen, chosen_mask)
        rejected_sum, _ = self._sum_and_count(rejected - ref_rejected, rejected_mask)
        margin = chosen_sum - rejected_sum
        loss = -F.logsigmoid(beta * margin).mean()
        return loss, {
            "loss": loss.detach(),
            "dpo/chosen_logratio": chosen_sum.mean(),
            "dpo/rejected_logratio": rejected_sum.mean(),
            "dpo/margin": margin.mean(),
            "dpo/accuracy": (margin > 0).float().mean(),
            "entropy/chosen_mean": self._masked_mean(chosen_entropy, chosen_mask),
            "entropy/rejected_mean": self._masked_mean(rejected_entropy, rejected_mask),
        }

    def _ipo(self, model, batch):
        chosen, rejected, chosen_mask, rejected_mask = self._policy_pair(model, batch)
        ref_chosen, ref_rejected = self._reference_token_logps(model, batch)
        chosen_sum, chosen_count = self._sum_and_count(chosen - ref_chosen, chosen_mask)
        rejected_sum, rejected_count = self._sum_and_count(
            rejected - ref_rejected, rejected_mask
        )
        # IPO uses length-normalized response log-ratios. Without this
        # normalization, longer responses dominate both the margin and gradient.
        chosen_mean = chosen_sum / chosen_count
        rejected_mean = rejected_sum / rejected_count
        margin = chosen_mean - rejected_mean
        loss = ((margin - 1.0 / (2.0 * self.args.beta)) ** 2).mean()
        return loss, {
            "loss": loss.detach(),
            "ipo/margin": margin.mean(),
            "ipo/accuracy": (margin > 0).float().mean(),
        }

    def _cpo(self, model, batch):
        chosen, rejected, chosen_mask, rejected_mask = self._policy_pair(model, batch)
        chosen_sum, chosen_count = self._sum_and_count(chosen, chosen_mask)
        rejected_sum, rejected_count = self._sum_and_count(rejected, rejected_mask)
        chosen_mean = chosen_sum / chosen_count
        rejected_mean = rejected_sum / rejected_count
        preference_loss = -F.logsigmoid(self.args.beta * (chosen_mean - rejected_mean)).mean()
        nll_loss = -chosen_mean.mean()
        loss = preference_loss + self.cpo_alpha * nll_loss
        return loss, {
            "loss": loss.detach(),
            "cpo/preference_loss": preference_loss.detach(),
            "cpo/nll_loss": nll_loss.detach(),
            "cpo/margin": (chosen_mean - rejected_mean).mean(),
        }

    def _simpo(self, model, batch):
        chosen, rejected, chosen_mask, rejected_mask = self._policy_pair(model, batch)
        chosen_sum, chosen_count = self._sum_and_count(chosen, chosen_mask)
        rejected_sum, rejected_count = self._sum_and_count(rejected, rejected_mask)
        chosen_reward = self.args.beta * chosen_sum / chosen_count
        rejected_reward = self.args.beta * rejected_sum / rejected_count
        margin = chosen_reward - rejected_reward
        loss = -F.logsigmoid(margin - self.simpo_gamma).mean()
        return loss, {
            "loss": loss.detach(),
            "simpo/chosen_reward": chosen_reward.mean(),
            "simpo/rejected_reward": rejected_reward.mean(),
            "simpo/margin": margin.mean(),
        }

    def _kto(self, model, batch):
        """Paired KTO with the same batch-level KL approximation used in KBS."""
        chosen, rejected, chosen_mask, rejected_mask = self._policy_pair(model, batch)
        ref_chosen, ref_rejected = self._reference_token_logps(model, batch)
        chosen_ratio, _ = self._sum_and_count(chosen - ref_chosen, chosen_mask)
        rejected_ratio, _ = self._sum_and_count(rejected - ref_rejected, rejected_mask)
        kl = chosen_ratio.detach().mean()
        chosen_loss = (1.0 - torch.sigmoid(self.args.beta * (chosen_ratio - kl))).mean()
        rejected_loss = (1.0 - torch.sigmoid(-self.args.beta * (rejected_ratio - kl))).mean()
        loss = 0.5 * (chosen_loss + rejected_loss)
        return loss, {
            "loss": loss.detach(),
            "kto/chosen_loss": chosen_loss.detach(),
            "kto/rejected_loss": rejected_loss.detach(),
            "kto/kl_approx": kl.detach(),
        }

    def _confpo_weights(self, token_logps, mask):
        mask_f = mask.to(token_logps.dtype)
        with torch.no_grad():
            confidence = torch.exp(token_logps.detach().clamp_max(0.0))
            uncertainty = 1.0 - confidence
            raw = torch.exp(uncertainty / self.confpo_temperature) * mask_f
            mean_weight = raw.sum(-1, keepdim=True) / mask_f.sum(-1, keepdim=True).clamp_min(1.0)
            weights = raw / (mean_weight + 1e-8)
        return weights * mask_f

    def _confpo(self, model, batch):
        chosen, rejected, chosen_mask, rejected_mask = self._policy_pair(model, batch)
        ref_chosen, ref_rejected = self._reference_token_logps(model, batch)
        chosen_weights = self._confpo_weights(chosen, chosen_mask)
        rejected_weights = self._confpo_weights(rejected, rejected_mask)
        chosen_weighted = ((chosen - ref_chosen) * chosen_weights).sum(-1)
        rejected_weighted = ((rejected - ref_rejected) * rejected_weights).sum(-1)
        margin = chosen_weighted - rejected_weighted
        loss = -F.logsigmoid(self.args.beta * margin).mean()
        valid_weights = torch.cat(
            [chosen_weights[chosen_mask.bool()], rejected_weights[rejected_mask.bool()]]
        )
        return loss, {
            "loss": loss.detach(),
            "confpo/margin": margin.mean(),
            "confpo/weight_mean": valid_weights.mean(),
        }

    def _response_stats_from_logits(
        self,
        policy_logits,
        reference_logits,
        labels,
        weights=None,
        kl_direction="reference_policy",
    ):
        shifted_policy = policy_logits[:, :-1, :]
        shifted_reference = reference_logits[:, :-1, :]
        shifted_labels = labels[:, 1:].clone()
        mask = shifted_labels.ne(-100)
        gather_ids = shifted_labels.masked_fill(~mask, 0).unsqueeze(-1)
        policy_log_z = torch.logsumexp(shifted_policy.float(), dim=-1)
        with torch.no_grad():
            reference_log_z = torch.logsumexp(shifted_reference.float(), dim=-1)
        policy_target = torch.gather(shifted_policy, -1, gather_ids).squeeze(-1).float() - policy_log_z
        reference_target = (
            torch.gather(shifted_reference, -1, gather_ids).squeeze(-1).float() - reference_log_z
        )

        per_position_kl = torch.zeros_like(policy_log_z)
        vocab_size = shifted_policy.size(-1)
        for start in range(0, vocab_size, self.vocab_chunk_size):
            end = min(start + self.vocab_chunk_size, vocab_size)
            policy_logp = shifted_policy[..., start:end].float() - policy_log_z.unsqueeze(-1)
            reference_logp = shifted_reference[..., start:end].float() - reference_log_z.unsqueeze(-1)
            if kl_direction == "reference_policy":
                contribution = reference_logp.exp() * (reference_logp - policy_logp)
            elif kl_direction == "policy_reference":
                contribution = policy_logp.exp() * (policy_logp - reference_logp)
            else:
                raise ValueError(f"Unknown KL direction: {kl_direction}")
            per_position_kl = per_position_kl + contribution.sum(-1)

        mask_f = mask.float()
        effective_weights = mask_f if weights is None else weights[:, 1:].float() * mask_f
        log_ratio = ((policy_target - reference_target) * effective_weights).sum(-1)
        position_kl = (per_position_kl * effective_weights).sum(-1)
        policy_logp = (policy_target * effective_weights).sum(-1)
        return log_ratio, position_kl, policy_logp, mask

    def _token_level_side(self, model, batch, side):
        input_ids = batch[f"{side}_input_ids"]
        attention_mask = batch[f"{side}_attention_mask"]
        labels = batch[f"{side}_labels"]
        policy_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        with torch.no_grad():
            reference_context = model.disable_adapter() if self.ref_model is None else nullcontext()
            with reference_context:
                reference_model = model if self.ref_model is None else self.ref_model
                reference_logits = reference_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
        is_tdpo = self.method == "tdpo"
        return self._response_stats_from_logits(
            policy_logits,
            reference_logits,
            labels,
            weights=None if is_tdpo else batch[f"{side}_weight"],
            kl_direction="reference_policy" if is_tdpo else "policy_reference",
        )

    def _token_level_objective(self, model, batch):
        chosen_margin, chosen_kl, chosen_logp, chosen_mask = self._token_level_side(
            model, batch, "chosen"
        )
        rejected_margin, rejected_kl, rejected_logp, rejected_mask = self._token_level_side(
            model, batch, "rejected"
        )
        preference_margin = chosen_margin - rejected_margin
        if self.method == "tdpo":
            if self.tdpo_version == 1:
                objective_logits = preference_margin - (rejected_kl - chosen_kl)
            else:
                objective_logits = preference_margin - self.tdpo_alpha * (
                    rejected_kl - chosen_kl.detach()
                )
            chosen_reward = chosen_margin + chosen_kl
            rejected_reward = rejected_margin + rejected_kl
        else:
            objective_logits = preference_margin - self.tdpo_alpha * (chosen_kl - rejected_kl)
            chosen_reward = chosen_margin - chosen_kl
            rejected_reward = rejected_margin - rejected_kl
        loss = -F.logsigmoid(self.args.beta * objective_logits).mean()
        metrics = {
            "loss": loss.detach(),
            f"{self.method}/chosen_reward": chosen_reward.mean(),
            f"{self.method}/rejected_reward": rejected_reward.mean(),
            f"{self.method}/margin": (chosen_reward - rejected_reward).mean(),
            f"{self.method}/chosen_kl": chosen_kl.mean(),
            f"{self.method}/rejected_kl": rejected_kl.mean(),
            f"{self.method}/chosen_logp": chosen_logp.mean(),
            f"{self.method}/rejected_logp": rejected_logp.mean(),
            f"{self.method}/chosen_tokens": chosen_mask.float().sum(-1).mean(),
            f"{self.method}/rejected_tokens": rejected_mask.float().sum(-1).mean(),
        }
        return loss, metrics

    def get_batch_loss_metrics(self, model, batch):
        if self.method == "dpo" or self.method in INTERNAL_METHODS:
            return self._standard_dpo(model, batch)
        if self.method == "ipo":
            return self._ipo(model, batch)
        if self.method == "cpo":
            return self._cpo(model, batch)
        if self.method == "simpo":
            return self._simpo(model, batch)
        if self.method == "kto":
            return self._kto(model, batch)
        if self.method == "confpo":
            return self._confpo(model, batch)
        return self._token_level_objective(model, batch)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=ALL_METHODS, required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weighted_dataset")
    parser.add_argument("--deepspeed", required=True)
    parser.add_argument("--dataset_max_length", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--cpo_alpha", type=float, default=1.0)
    parser.add_argument("--simpo_gamma", type=float, default=0.5)
    parser.add_argument("--confpo_temperature", type=float, default=1.0)
    parser.add_argument("--tdpo_alpha", type=float, default=0.5)
    parser.add_argument("--tdpo_version", type=int, choices=(1, 2), default=2)
    parser.add_argument("--tis_min_scale", type=float, default=0.7)
    parser.add_argument("--tis_max_scale", type=float, default=1.3)
    parser.add_argument("--vocab_chunk_size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--keep_temp_lora", action="store_true")
    return parser.parse_args()


def validate_args(args):
    for name in ("model_id", "dataset_name", "deepspeed"):
        if not Path(getattr(args, name)).exists():
            raise FileNotFoundError(f"--{name} does not exist: {getattr(args, name)}")
    if args.method == "tis_dpo":
        if not args.weighted_dataset or not Path(args.weighted_dataset).is_file():
            raise FileNotFoundError("--weighted_dataset is required and must exist for tis_dpo")
    if args.dataset_max_length > args.max_length:
        raise ValueError("--dataset_max_length cannot exceed --max_length")
    if args.per_device_train_batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("batch size and gradient accumulation must be positive")
    if args.beta <= 0 or args.confpo_temperature <= 0:
        raise ValueError("beta and confpo_temperature must be positive")
    if args.tis_min_scale <= 0 or args.tis_max_scale < args.tis_min_scale:
        raise ValueError("invalid TIS rank scale interval")
    if args.vocab_chunk_size < 1:
        raise ValueError("vocab_chunk_size must be positive")


def write_run_config(args, output_dir):
    if int(os.environ.get("RANK", "0")) != 0:
        return
    config = vars(args).copy()
    config.update(
        {
            "public_method": args.method in PUBLIC_METHODS,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "save_format": (
                "lora_adapter_only" if args.method in INTERNAL_METHODS else "standalone_merged_model"
            ),
            "kto_note": "paired KTO with batch-level KL approximation" if args.method == "kto" else None,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    adapter_path = output_dir / ("adapter" if args.method in INTERNAL_METHODS else "temp_lora")
    final_model_path = output_dir / "model"
    write_run_config(args, output_dir)

    model_family, is_instruct = detect_model_config(args.model_id)
    tokenizer = load_tokenizer(args.model_id, model_family)
    end_token_id = resolve_end_token_id(tokenizer, model_family, is_instruct)
    if args.method == "tis_dpo":
        dataset = load_tis_dataset(
            args.weighted_dataset,
            tokenizer,
            model_family,
            is_instruct,
            end_token_id,
            args.dataset_max_length,
            args.tis_min_scale,
            args.tis_max_scale,
        )
        collator = lambda features: weighted_dpo_collator(features, tokenizer.pad_token_id)
    else:
        dataset = load_dpo_dataset(
            args.dataset_name,
            tokenizer,
            model_family,
            is_instruct,
            end_token_id,
            max_len=args.dataset_max_length,
        )
        if args.method == "contrast_reverse":
            dataset = reverse_preferences(dataset)
        collator = lambda features: dpo_collator(features, tokenizer.pad_token_id)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = get_peft_model(model, get_lora_config())
    model.enable_input_require_grads()

    training_args = DPOConfig(
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
        beta=args.beta,
        max_length=args.max_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        save_strategy="no",
        deepspeed=args.deepspeed,
        seed=args.seed,
        data_seed=args.seed,
        run_name=args.experiment_name,
    )
    trainer = UnifiedPreferenceTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        method=args.method,
        cpo_alpha=args.cpo_alpha,
        simpo_gamma=args.simpo_gamma,
        confpo_temperature=args.confpo_temperature,
        tdpo_alpha=args.tdpo_alpha,
        tdpo_version=args.tdpo_version,
        vocab_chunk_size=args.vocab_chunk_size,
    )
    trainer.add_callback(EntropyHistoryCallback(output_dir / "entropy_history.csv"))
    if trainer.is_world_process_zero():
        print(
            f"Training {args.experiment_name}: method={args.method}, "
            f"model={args.model_id}, world_size={trainer.args.world_size}"
        )

    trainer.train()
    trainer.save_model(str(adapter_path))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(adapter_path)
    trainer.accelerator.wait_for_everyone()

    if trainer.is_world_process_zero():
        if args.method in INTERNAL_METHODS:
            (output_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")
            print(f"Saved TIS contrast adapter to {adapter_path}")
        else:
            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True,
            )
            merged_model = PeftModel.from_pretrained(base_model, adapter_path).merge_and_unload()
            merged_model.save_pretrained(final_model_path, safe_serialization=True)
            tokenizer.save_pretrained(final_model_path)
            if not args.keep_temp_lora:
                shutil.rmtree(adapter_path)
            (output_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")
            print(f"Saved merged model to {final_model_path}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
