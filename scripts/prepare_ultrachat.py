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
    destination_path: Path = Path("data/ultrachat2"),
    checkpoint_dir: Path = Path("/teamspace/s3_connections/tinyllama-checkpoints/export/step-00357500-3.0T"),
    dataset_name: str = "stingning/ultrachat",
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

    split_dataset = dataset["train"].train_test_split(test_size=100, seed=42, shuffle=True)

    train_set = split_dataset["train"]
    test_set = split_dataset["test"]

    print(f"train has {len(train_set):,} samples")
    print(f"test has {len(test_set):,} samples")

    print("Processing test split ...")
    test_set_out = []
    for sample in tqdm(test_set):
        messages = sample["data"]
        for user_message, assistant_message in zip(messages[0::2], messages[1::2]):
            sample = {
                "instruction": user_message,
                "output": assistant_message,
            }
            test_set_out.append(prepare_sample(sample, tokenizer, max_length=max_seq_length))
            break

    # print(test_set_out[2]["labels"].shape)
    torch.save(test_set_out, destination_path / "test.pt")


    # print("Processing train split ...")
    # train_set_out = []
    # for sample in tqdm(train_set):
    #     messages = sample["data"]
    #     for user_message, assistant_message in zip(messages[0::2], messages[1::2]):
    #         sample = {
    #             "instruction": user_message,
    #             "output": assistant_message,
    #         }
    #         train_set_out.append(prepare_sample(sample, tokenizer, max_length=max_seq_length))
    # torch.save(train_set_out, destination_path / "train.pt")



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
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    print(full_prompt_and_response)
    # encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)
    encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, bos=False, eos=True, max_length=max_length)
    labels = encoded_full_prompt_and_response.clone()
    labels[:len(full_prompt)] = -1

    return {**example, "input_ids": encoded_full_prompt_and_response, "labels": labels}


def generate_prompt(example: dict) -> str:
    """Generates a standardized message to prompt the model with an instruction."""
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response: "
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
