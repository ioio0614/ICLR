"""Utilities for persisting per-step chosen/rejected policy entropy."""

from __future__ import annotations

import csv
from pathlib import Path

from transformers import TrainerCallback


class EntropyHistoryCallback(TrainerCallback):
    """Append optimizer-step entropy metrics to a CSV file on rank zero."""

    fieldnames = ("step", "epoch", "chosen_entropy", "rejected_entropy")

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.fieldnames).writeheader()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        chosen = logs.get("entropy/chosen_mean")
        rejected = logs.get("entropy/rejected_mean")
        if chosen is None or rejected is None:
            return
        row = {
            "step": state.global_step,
            "epoch": logs.get("epoch", state.epoch),
            "chosen_entropy": float(chosen),
            "rejected_entropy": float(rejected),
        }
        with self.output_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.fieldnames).writerow(row)
