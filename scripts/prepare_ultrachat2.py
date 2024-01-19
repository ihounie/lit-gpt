# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
import os
import sys
from pathlib import Path
from typing import Union, Optional

import numpy as np
from tqdm import tqdm
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Tokenizer


def prepare(
    destination_path: Path = Path("data/ultrachat3"),
    checkpoint_dir: Path = Path("checkpoints/lit-tiny-llama/lit-tiny-llama-3.0T"),
    dataset_name: str = "stingning/ultrachat",
    max_seq_length: int = 2048,
) -> None:
    from datasets import load_dataset  # huggingface datasets

    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(checkpoint_dir)

    dataset = load_dataset(dataset_name, num_proc=(os.cpu_count() // 2))

    split_dataset = dataset["train"].train_test_split(test_size=100, seed=42, shuffle=True)

    train_set = split_dataset["train"]
    test_set = split_dataset["test"]

    print(f"train has {len(train_set):,} samples")
    print(f"test has {len(test_set):,} samples")

    def process(example):
        messages = example["data"]
        # assert len(messages) % 2 == 0
        # for user_message, assistant_message in zip(messages[0::2], messages[1::2]):
        sample = {
            "instruction": messages[0],
            "output": messages[1],
        }
        return prepare_sample(sample, tokenizer, max_length=max_seq_length)

    # tokenize the dataset
    tokenized = split_dataset.map(process, remove_columns=["id", "data"], desc="tokenizing the splits", num_proc=(os.cpu_count() // 2))
    tokenized.set_format("pt", output_all_columns=True)

    for split, dset in tokenized.items():
        filename = destination_path / f"{split}.bin"
        # dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        all_data = list(dset)
        print(all_data[0])
        torch.save(all_data, filename)

    # concatenate all the ids in each dataset into one large file we can use for training
    # for split, dset in tokenized.items():
    #     arr_len = np.sum(dset["len"], dtype=np.uint64)
    #     filename = destination_path / f"{split}.bin"
    #     dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
    #     arr = np.memmap(str(filename), dtype=dtype, mode="w+", shape=(arr_len,))
    #     total_batches = 1024

    #     idx = 0
    #     for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
    #         # Batch together samples for faster write
    #         batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
    #         arr_batch = np.concatenate(batch["ids"])
    #         # Write into mmap
    #         arr[idx : idx + len(arr_batch)] = arr_batch
    #         idx += len(arr_batch)
    #     arr.flush()


def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int) -> dict:
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length, bos=True, eos=False)
    encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, bos=True, eos=True, max_length=max_length)
    labels = encoded_full_prompt_and_response.clone()
    labels[:len(encoded_full_prompt)] = -1

    return {"input_ids": encoded_full_prompt_and_response.short(), "labels": labels.short()}


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
