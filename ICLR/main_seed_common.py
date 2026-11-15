"""Only the helpers required by the unfinished main-table seed queue."""
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

MODELS = ('llama_base','llama_instruct','mistral_instruct')
METHODS = ('dpo','ipo','cpo','kto','simpo','tdpo','tis_dpo','confpo','sft')

@dataclass
class Job:
    key: str
    stage: str
    role: str
    model_key: str
    method: str
    seed: int
    gpus: list[str]
    parameters: dict
    output: str
    command: list[str]
    provenance: dict
    dependencies: list[str]
    environment: dict


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@lru_cache(maxsize=None)
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_model(path):
    path = Path(path)
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Missing model config: {path}")
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (path / name).is_file():
            shards = set(json_read(path / name)["weight_map"].values())
            missing = [shard for shard in shards if not (path / shard).is_file()]
            if not shards or missing:
                raise FileNotFoundError(f"Missing model shards in {path}: {missing}")
            return
    if not (path / "model.safetensors").is_file() and not (path / "pytorch_model.bin").is_file():
        raise FileNotFoundError(f"Missing model weights: {path}")


def check_completed(job):
    path = Path(job.output)
    if not (path / "_SUCCESS").is_file():
        return False
    if job.role == "tis_weight_estimation":
        if not (path / "train.weighted.jsonl").is_file() or not (path / "train.weighted.jsonl").stat().st_size:
            raise FileNotFoundError(f"Completed TIS job has no weighted data: {path}")
    elif job.role == "tis_auxiliary_training":
        adapter = path / "adapter"
        if not (adapter / "adapter_config.json").is_file() or not any(
                (adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
            raise FileNotFoundError(f"Completed TIS job has no adapter: {path}")
    else:
        check_model(path / "model")
    return True


@contextmanager
def exclusive_lock(path, blocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise RuntimeError(f"Another launcher holds {path}") from error
        yield


def stream_command(command, env, cwd, log_path):
    """Preserve nonzero exits and stop the entire torchrun process group on interruption."""
    with Path(log_path).open("a", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True) as process:
            try:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    log.flush()
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise

def smaller_batch(micro, accumulation):
    if micro <= 1:
        raise RuntimeError('CUDA OOM persists at per-device batch 1; automatic reduction exhausted')
    total = micro * accumulation
    reduced = micro // 2
    # Odd batches (e.g. 3 -> 1) require more than doubling accumulation.
    while total % reduced:
        reduced -= 1
    return reduced, total // reduced


def source_files(suite, job):
    root=suite.project
    result=[Path(__file__).resolve(),root/'scripts/run_pending_seeds.py',root/'src/common/model_utils.py']
    if job.role=='tis_weight_estimation':return result+[root/'scripts/estimate_tis_weights.py']
    return result+[root/'src/training'/('sft_unified.py' if job.method=='sft' else 'preference_optimization.py'),root/'src/training/entropy_logging.py']
