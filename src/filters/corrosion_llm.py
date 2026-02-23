import json
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

# Optional deps for token counting / validation
try:
    import tiktoken
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False

# OpenAI SDK
from openai import OpenAI


# -----------------------------
# Prompts (kept same as yours)
# -----------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are a chemist assistant with expertise in material compatibility and chemical properties. "
    "Your task is to determine if a given chemical is corrosive to metals based on its SMILES representation."
)

def default_user_prompt(smiles: str) -> str:
    return (
        f"Given the SMILES representation '{smiles}', determine if this compound is corrosive to metals. "
        "Respond strictly with '0' for non-corrosive and '1' for corrosive, without any additional text."
    )


# -----------------------------
# 1) Build JSONL from CSV
# -----------------------------
def csv_to_jsonl_for_finetune(
    train_csv_path: str,
    jsonl_out_path: str,
    smiles_col: str = "smiles",
    label_col: str = "corrosion_status",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_prompt_fn=default_user_prompt,
) -> None:
    """
    Converts a training CSV to OpenAI chat fine-tune JSONL format.
    Keeps your message structure (system -> user -> assistant).
    """
    train_df = pd.read_csv(train_csv_path)

    with open(jsonl_out_path, "w", encoding="utf-8") as jsonl_file:
        for _, row in train_df.iterrows():
            smiles = str(row[smiles_col])
            corrosion_status = str(row[label_col])  # keep as string ("0" or "1")

            entry = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_fn(smiles)},
                    {"role": "assistant", "content": corrosion_status},
                ]
            }
            jsonl_file.write(json.dumps(entry) + "\n")


# -----------------------------
# 2) Load + validate JSONL (format checks)
# -----------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def validate_chat_jsonl_format(dataset: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Lightweight version of your format checks.
    Returns dict of errors (empty => OK).
    """
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    return dict(format_errors)


# -----------------------------
# 3) Token stats (optional; needs tiktoken)
# -----------------------------
def token_stats(dataset: List[Dict[str, Any]], model_encoding: str = "cl100k_base") -> Dict[str, Any]:
    """
    Replicates your token counting approach.
    Requires tiktoken installed. If not installed, raises RuntimeError.
    """
    if not _HAS_TIKTOKEN:
        raise RuntimeError("tiktoken is not installed. Install it or skip token_stats().")

    encoding = tiktoken.get_encoding(model_encoding)

    def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def num_assistant_tokens_from_messages(messages):
        num_tokens = 0
        for message in messages:
            if message["role"] == "assistant":
                num_tokens += len(encoding.encode(message["content"]))
        return num_tokens

    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1

        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

    def dist(values):
        return {
            "min": int(np.min(values)),
            "max": int(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p10": float(np.quantile(values, 0.1)),
            "p90": float(np.quantile(values, 0.9)),
        }

    return {
        "num_examples": len(dataset),
        "missing_system": int(n_missing_system),
        "missing_user": int(n_missing_user),
        "num_messages_per_example": dist(n_messages),
        "num_total_tokens_per_example": dist(convo_lens),
        "num_assistant_tokens_per_example": dist(assistant_message_lens),
        "num_over_16385": int(sum(l > 16385 for l in convo_lens)),
        "billing_tokens_estimate_1_epoch": int(sum(min(16385, l) for l in convo_lens)),
    }


# -----------------------------
# 4) Upload file + create fine-tune job
# -----------------------------
def upload_training_file(api_key: str, jsonl_path: str) -> str:
    """
    Upload JSONL to OpenAI for fine-tuning. Returns file_id.
    """
    client = OpenAI(api_key=api_key)
    resp = client.files.create(file=open(jsonl_path, "rb"), purpose="fine-tune")
    return resp.id


def create_finetune_job(api_key: str, training_file_id: str, base_model: str) -> str:
    """
    Starts fine-tuning job. Returns fine_tuning_job_id.
    """
    client = OpenAI(api_key=api_key)
    job = client.fine_tuning.jobs.create(training_file=training_file_id, model=base_model)
    return job.id


def get_finetune_job(api_key: str, fine_tuning_job_id: str) -> Dict[str, Any]:
    """
    Retrieve fine-tune job info.
    """
    client = OpenAI(api_key=api_key)
    job = client.fine_tuning.jobs.retrieve(fine_tuning_job_id)
    # job is an object; return as dict-like via model_dump() if available
    return job.model_dump() if hasattr(job, "model_dump") else dict(job)


# -----------------------------
# 5) Predict on a dataframe / CSV using your fine-tuned model
# -----------------------------
def predict_corrosion_for_smiles_list(
    api_key: str,
    model_name: str,
    smiles_list: List[str],
    system_prompt: str = "You are a chemist assistant.",
    user_template: str = "Is {smiles} corrosive to metals? Respond only with 0 for non-corrosive and 1 for corrosive.",
    temperature: float = 0.0,
) -> List[str]:
    """
    Runs chat completions for each SMILES and returns raw string outputs (as you do now).
    Keeps your inference prompt style.
    """
    client = OpenAI(api_key=api_key)

    preds = []
    for smiles in smiles_list:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_template.format(smiles=smiles)},
            ],
            temperature=temperature,
        )
        preds.append(completion.choices[0].message.content)
    return preds


def predict_corrosion_for_csv(
    api_key: str,
    model_name: str,
    csv_path: str,
    smiles_col: str = "smiles",
    out_csv_path: Optional[str] = None,
    **predict_kwargs,
) -> pd.DataFrame:
    """
    Loads csv, predicts, returns df with new column 'corr_pred_raw'.
    If out_csv_path provided, saves it.
    """
    df = pd.read_csv(csv_path)
    smiles_list = df[smiles_col].astype(str).tolist()
    preds = predict_corrosion_for_smiles_list(
        api_key=api_key,
        model_name=model_name,
        smiles_list=smiles_list,
        **predict_kwargs,
    )
    df["corr_pred_raw"] = preds

    if out_csv_path is not None:
        df.to_csv(out_csv_path, index=False)

    return df