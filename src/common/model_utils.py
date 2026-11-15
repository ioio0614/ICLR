import copy
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import LoraConfig


def detect_model_config(model_id: str):
    name = model_id.lower().strip()
    family = None
    if "qwen" in name:
        family = "qwen2.5"
    elif "llama" in name:
        family = "llama3"
    elif "mistral" in name:
        family = "mistral"
    else:
        raise ValueError(f"不支持的模型类型: {model_id}，仅支持 qwen2.5 / llama3 / mistral")

    is_instruct = "instruct" in name or "chat" in name
    return family, is_instruct


def load_tokenizer(model_id: str, model_family: str):
    if model_family == "mistral":
        tokenizer = AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer


def resolve_end_token_id(tokenizer, model_family: str, is_instruct: bool):
    if is_instruct and model_family == "llama3":
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None and eot_id != tokenizer.unk_token_id:
            return eot_id
    return tokenizer.eos_token_id


def build_chat_prompt(
    prompt_str: str,
    tokenizer,
    model_family: str,
    is_instruct: bool,
    messages=None,
):
    if not is_instruct:
        return prompt_str

    if messages:
        chat_messages = [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in messages
        ]
        if model_family == "qwen2.5" and chat_messages[0]["role"] != "system":
            chat_messages.insert(
                0, {"role": "system", "content": "You are a helpful assistant."}
            )
        return tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )

    if model_family == "llama3":
        messages = [{"role": "user", "content": prompt_str}]
    elif model_family == "mistral":
        messages = [{"role": "user", "content": prompt_str}]
    elif model_family == "qwen2.5":
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_str},
        ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode_response(response_content: str, tokenizer, prompt_ids, end_token_id,
                    model_family: str, is_instruct: bool, max_len: int = 1024):
    if is_instruct and model_family == "mistral":
        response_text = " " + response_content.lstrip()
    else:
        response_text = response_content.lstrip()

    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    input_ids = prompt_ids + response_ids + [end_token_id]

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len - 1] + [end_token_id]

    labels = copy.deepcopy(input_ids)
    mask_len = min(len(prompt_ids), len(input_ids))
    labels[:mask_len] = [-100] * mask_len

    return input_ids, [1] * len(input_ids), labels


def tokenize_dpo_example(example, tokenizer, model_family: str, is_instruct: bool,
                         end_token_id, max_len: int = 1024):
    prompt_str = str(example["prompt"]).strip() or "Empty prompt"
    prompt_text = build_chat_prompt(
        prompt_str,
        tokenizer,
        model_family,
        is_instruct,
        messages=example.get("messages"),
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    c_ids, c_mask, c_labels = encode_response(
        example["chosen"], tokenizer, prompt_ids, end_token_id, model_family, is_instruct, max_len
    )
    r_ids, r_mask, r_labels = encode_response(
        example["rejected"], tokenizer, prompt_ids, end_token_id, model_family, is_instruct, max_len
    )

    return {
        "chosen_input_ids": c_ids,
        "chosen_attention_mask": c_mask,
        "chosen_labels": c_labels,
        "rejected_input_ids": r_ids,
        "rejected_attention_mask": r_mask,
        "rejected_labels": r_labels,
    }


def load_dpo_dataset(dataset_path, tokenizer, model_family, is_instruct, end_token_id, max_len=1024):
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def _tokenize(ex):
        return tokenize_dpo_example(ex, tokenizer, model_family, is_instruct, end_token_id, max_len)

    dataset = dataset.map(_tokenize, remove_columns=dataset.column_names, desc="Tokenizing dataset")
    return dataset


def pad_1d(seqs, pad_value, pad_side):
    max_len = max(len(s) for s in seqs)
    if pad_side == "right":
        return torch.tensor([s + [pad_value] * (max_len - len(s)) for s in seqs])
    else:
        return torch.tensor([[pad_value] * (max_len - len(s)) + s for s in seqs])


def dpo_collator(features, pad_token_id):
    batch = {}
    for k, v in [
        ("chosen_input_ids", pad_token_id),
        ("chosen_attention_mask", 0),
        ("chosen_labels", -100),
        ("rejected_input_ids", pad_token_id),
        ("rejected_attention_mask", 0),
        ("rejected_labels", -100),
    ]:
        batch[k] = pad_1d([f[k] for f in features], v, "right")
    return batch


def get_lora_config():
    return LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
