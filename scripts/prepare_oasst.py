# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Implementation derived from https://github.com/tloen/alpaca-lora"""

import json
import os
import sys
import re
from pathlib import Path
from typing import Optional

import requests
import torch
from torch.utils.data import random_split
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer


def prepare(
    destination_path: Path = Path("data/oasst-top1"),
    checkpoint_dir: Path = Path("/teamspace/s3_connections/tinyllama-checkpoints/export/step-00357500-3.0T"),
    # dataset_name: str = "timdettmers/openassistant-guanaco",
    dataset_name: str = "OpenAssistant/oasst_top1_2023-08-25",
    max_seq_length: Optional[int] = None,
) -> None:
    """TODO
    """
    from datasets import load_dataset  # huggingface datasets


    if max_seq_length is None:
        with open(checkpoint_dir / "lit_config.json", "r", encoding="utf-8") as file:
            config = json.load(file)
            max_seq_length = config["block_size"]

    destination_path.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = Tokenizer(checkpoint_dir)

    dataset = load_dataset(dataset_name, num_proc=(os.cpu_count() // 2))

    train_set = dataset["train"]
    test_set = dataset["test"]

    print(f"train has {len(train_set):,} samples")
    print(f"test has {len(test_set):,} samples")

    # print(train_set[2]["text"])
    sample = prepare_sample(
        example=train_set[2],
        tokenizer=tokenizer,
        max_length=max_seq_length,
    )
    print(sample)

    # print("Processing train split ...")
    # train_set = [
    #     prepare_sample(
    #         example=sample,
    #         tokenizer=tokenizer,
    #         max_length=max_seq_length,
    #     )
    #     for sample in tqdm(train_set)
    # ]
    # torch.save(train_set, destination_path / "train.pt")

    # print("Processing test split ...")
    # test_set = [
    #     prepare_sample(
    #         example=sample,
    #         tokenizer=tokenizer,
    #         max_length=max_seq_length,
    #     )
    #     for sample in tqdm(test_set)
    # ]
    # torch.save(test_set, destination_path / "test.pt")


def download_if_missing(file_path: Path, file_url: str) -> None:
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists() and file_path.stat().st_size > 0:
        return
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(requests.get(file_url).text)


def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int) -> dict:
    """Processes a single sample.

    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    text = example["text"]
    start_marker = "<|im_start|>"
    end_marker = "<|im_end|>"
    regex = f'{re.escape(start_marker)}user\n(.*){re.escape(end_marker)}\n{re.escape(start_marker)}assistant\n(.*){re.escape(end_marker)}'
    pattern = re.compile(regex, re.DOTALL)

    match = pattern.search(text)
    example = {
        "instruction": match.group(1),
        "output": match.group(2),
    }

    # assert start_marker not in example["instruction"], f"{example['instruction']}"
    # assert end_marker not in example["instruction"]
    # assert start_marker not in example["output"]
    # assert end_marker not in example["output"]

    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)
    encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, bos=False, eos=True, max_length=max_length)
    labels = encoded_full_prompt_and_response.clone()

    return {**example, "input_ids": encoded_full_prompt_and_response, "labels": labels}


def generate_prompt(example: dict) -> str:
    """Generates a standardized message to prompt the model with an instruction."""
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
