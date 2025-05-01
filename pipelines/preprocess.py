import os
import sys
from typing import Optional
from dataclasses import dataclass, field
import numpy as np
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi, add_collection_item
from transformers import HfArgumentParser
import tiktoken


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import CONFIGS

HUGGINGFACE_CONFIGS = CONFIGS.services.huggingface
HFAPI = HfApi(HUGGINGFACE_CONFIGS["token"])
DATASET_SPLIT_SEED = 42
DATASET_CONFIGS = CONFIGS.datasets.preprocess
DATASET_CACHE_DIR = CONFIGS.utils.cache["dataset_cache_dir"]
TOKENIZER = tiktoken.get_encoding("cl100k_base")


@dataclass
class ScriptArguments:
    prefix: Optional[str] = field(
        default=None,
        metadata={"help": "prefix for the datasets to preprocess, default is None."},
    )


# Download RLHF dataset and split according to configs
def load_and_split(dataset_name):
    # If the dataset has only one split, load and split it
    if len(DATASET_CONFIGS[dataset_name]["split"]) == 1:
        raw_dataset = load_dataset(
            DATASET_CONFIGS[dataset_name]["id"],
            name=DATASET_CONFIGS[dataset_name]["name"],
            split=DATASET_CONFIGS[dataset_name]["split"][0],
            cache_dir=DATASET_CACHE_DIR,
        )
        if DATASET_CONFIGS[dataset_name]["limit"]:
            raw_dataset = raw_dataset.select(
                range(DATASET_CONFIGS[dataset_name]["limit"])
            )
        splitted_dataset = raw_dataset.train_test_split(
            test_size=DATASET_CONFIGS[dataset_name]["ratio"],
            seed=DATASET_SPLIT_SEED,
        )
        raw_train_dataset, raw_eval_dataset = (
            splitted_dataset["train"],
            splitted_dataset["test"],
        )
    # If the dataset has two splits, load them separately
    elif len(DATASET_CONFIGS[dataset_name]["split"]) == 2:
        raw_train_dataset = load_dataset(
            DATASET_CONFIGS[dataset_name]["id"],
            name=DATASET_CONFIGS[dataset_name]["name"],
            split=DATASET_CONFIGS[dataset_name]["split"][0],
            cache_dir=DATASET_CACHE_DIR,
        )
        raw_eval_dataset = load_dataset(
            DATASET_CONFIGS[dataset_name]["id"],
            name=DATASET_CONFIGS[dataset_name]["name"],
            split=DATASET_CONFIGS[dataset_name]["split"][1],
            cache_dir=DATASET_CACHE_DIR,
        )
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    return raw_train_dataset, raw_eval_dataset


# Convert datasets to SFT & DPO format, using the preferred answer as target
def format_and_upload(dataset_name, raw_train_dataset, raw_eval_dataset):
    # PKU-Alignment/PKU-SafeRLHF*
    if dataset_name.startswith("P"):
        if dataset_name.startswith("PP"):
            preference_label = "better_response_id"
        elif dataset_name.startswith("PM"):
            preference_label = "safer_response_id"
        else:
            raise ValueError(f"Unsupported dataset {dataset_name}")
        # PKU-SafeRLHF follows `Human: <prompt>\n\nAssistant: <response_chosen>` format
        # Only consider the "safer"/harmfulness preference here, dataset also provide a "better"/helpfulness preference
        format_func = lambda sample: {
            "prompt": f'Human: {sample["prompt"]}\n\nAssistant: ',
            "chosen": (
                sample["response_0"]
                if sample[preference_label] == 0
                else sample["response_1"]
            ),
            "rejected": (
                sample["response_1"]
                if sample[preference_label] == 0
                else sample["response_0"]
            ),
        }

    # openbmb/UltraFeedback
    elif dataset_name.startswith("U"):
        # Only use samples where there are four completions
        # Filter out samples that are too long
        filter_func = lambda sample: (len(sample["completions"]) == 4) and (
            max(
                [
                    len(TOKENIZER.encode(sample["instruction"] + c["response"]))
                    for c in sample["completions"]
                ]
            )
            < 2048 - 256
        )
        raw_train_dataset = raw_train_dataset.filter(filter_func, num_proc=8)
        raw_eval_dataset = raw_eval_dataset.filter(filter_func, num_proc=8)
        format_func = lambda sample: {
            # UltraFeedback uses the same template as Mistral-Instruct-v0.2
            "prompt": f'[INST] {sample["instruction"]} [/INST] ',
            # Only use the overall score to determine the chosen and rejected
            "chosen": sample["completions"][
                np.argmax([c["overall_score"] for c in sample["completions"]])
            ]["response"],
            "rejected": sample["completions"][
                np.argmin([c["overall_score"] for c in sample["completions"]])
            ]["response"],
        }

    # Anthropic/hh-rlhf
    elif dataset_name.startswith("AH"):
        # Only use samples where the chose and rejected are different
        filter_func = lambda sample: sample["chosen"] != sample["rejected"]
        raw_train_dataset = raw_train_dataset.filter(filter_func, num_proc=8)
        raw_eval_dataset = raw_eval_dataset.filter(filter_func, num_proc=8)
        # Anthropic-hh follows `Human: <prompt>\n\nAssistant: <chosen>` format
        # Omit the first two newlines just for consistency with other datasets
        search_term = "\n\nAssistant:"
        format_func = lambda sample: {
            "prompt": sample["chosen"][
                2 : sample["chosen"].rfind(search_term) + len(search_term)
            ],
            "chosen": sample["chosen"][
                sample["chosen"].rfind(search_term) + len(search_term) :
            ],
            "rejected": sample["rejected"][
                sample["chosen"].rfind(search_term) + len(search_term) :
            ],
        }

    # openai/summarize_from_feedback
    elif dataset_name.startswith("OS"):
        # OpenAI-Summarize follows https://arxiv.org/pdf/2009.01325.pdf page 18
        format_func = lambda sample: {
            "prompt": f'SUBREDDIT: r/{sample["info"]["subreddit"]}\nTITLE: {sample["info"]["title"]}\nPOST: {sample["info"]["post"]}\nTL;DR:',
            "chosen": sample["summaries"][sample["choice"]]["text"],
            "rejected": sample["summaries"][1 - sample["choice"]]["text"],
        }

    # openai/webgpt_comparisons
    elif dataset_name.startswith("OW"):
        # Only use samples where the two scores are different
        filter_func = lambda sample: sample["score_0"] != sample["score_1"]
        raw_train_dataset = raw_train_dataset.filter(filter_func, num_proc=8)
        raw_eval_dataset = raw_eval_dataset.filter(filter_func, num_proc=8)
        # OpenAI-WebGPT follows `Human: <prompt>\n\nAssistant: <response>` format
        format_func = lambda sample: {
            "prompt": f'Human: {sample["question"]["full_text"]}\n\nAssistant: ',
            "chosen": (
                sample["answer_0"]
                if sample["score_0"] > sample["score_1"]
                else sample["answer_1"]
            ),
            "rejected": (
                sample["answer_1"]
                if sample["score_0"] > sample["score_1"]
                else sample["answer_0"]
            ),
        }

    # We only support listed datasets so far
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    # Map the format function to the datasets
    train_dataset = raw_train_dataset.map(format_func, num_proc=8)
    eval_dataset = raw_eval_dataset.map(format_func, num_proc=8)

    # Upload the formatted datasets to Hugging Face Hub
    DatasetDict(
        {
            "train": train_dataset,
            "eval": eval_dataset,
        }
    ).push_to_hub(HUGGINGFACE_CONFIGS["prefix"]["datasets"] + dataset_name)
    # Add the dataset to the collection
    add_collection_item(
        collection_slug=HUGGINGFACE_CONFIGS["collections"]["datasets"],
        item_id=HUGGINGFACE_CONFIGS["prefix"]["datasets"] + dataset_name,
        item_type="dataset",
        exists_ok=True,
    )


def display_dataset(dataset_name):
    dataset = load_dataset(
        f"{HUGGINGFACE_CONFIGS['prefix']['datasets']}{dataset_name}", cache_dir=DATASET_CACHE_DIR
    )
    for split_name in ["train", "eval"]:
        print(
            f"Dataset: {dataset_name}, Split: {split_name}, Size: {len(dataset[split_name])}"
        )


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    for dataset_name in DATASET_CONFIGS.keys():
        if script_args.prefix and not dataset_name.startswith(script_args.prefix):
            continue
        raw_train_dataset, raw_eval_dataset = load_and_split(dataset_name)
        format_and_upload(dataset_name, raw_train_dataset, raw_eval_dataset)
        print(f"Preprocessed and uploaded dataset: {dataset_name}")

    for dataset_name in DATASET_CONFIGS.keys():
        if script_args.prefix and not dataset_name.startswith(script_args.prefix):
            continue
        display_dataset(dataset_name)


if __name__ == "__main__":
    main()
