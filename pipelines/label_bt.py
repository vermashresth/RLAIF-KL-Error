import os
import sys
from dataclasses import dataclass, field
from tqdm.auto import tqdm
from huggingface_hub import HfApi
import torch
import numpy as np
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
)
from transformers import logging
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_dataset, DatasetDict


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import CONFIGS, sample_every_k_batched

# Try to import safe_rlhf.models, make it optional for testing
try:
    from safe_rlhf.models import AutoModelForScore
    HAS_SAFE_RLHF = True
except ImportError:
    HAS_SAFE_RLHF = False
    AutoModelForScore = None

HUGGINGFACE_CONFIGS = CONFIGS.services.huggingface
HFAPI = HfApi(HUGGINGFACE_CONFIGS["token"])
REWARD_CONFIGS = CONFIGS.evaluations.reward
CACHE_CONFIGS = CONFIGS.utils.cache

accelerator = Accelerator()
tqdm.pandas()
logging.set_verbosity_error()


@dataclass
class ScriptArguments:
    dataset: str = field(
        metadata={"help": "dataset name to load and label"},
    )
    tag: str = field(
        default="default",
        metadata={"help": "tag for the experiment"},
    )
    split: str = field(
        default="default",
        metadata={"help": "dataset split to evaluate"},
    )
    every_k: int = field(
        default=1,
        metadata={
            "help": "evaluate every k samples, if a fraction, evaluate each sample 1/k times"
        },
    )
    per_device_evalreward_batch_size: int = field(
        default=16,
        metadata={"help": "batch size per device"},
    )
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "maximum sequence length, sequences will be right padded (and possibly truncated)"
        },
    )
    padding: bool = field(
        default=True,
        metadata={"help": "use padding for tokenizer"},
    )
    truncation: bool = field(
        default=True,
        metadata={"help": "allow truncation for tokenizer"},
    )
    score_model_id: str = field(
        default="none",
        metadata={"help": "model id for scoring, do not change"},
    )
    model_cache_dir: str = field(
        default=CACHE_CONFIGS["model_cache_dir"],
        metadata={"help": "model cache directory"},
    )
    dataset_cache_dir: str = field(
        default=CACHE_CONFIGS["dataset_cache_dir"],
        metadata={"help": "dataset cache directory"},
    )
    process_all_splits: bool = field(
        default=True,
        metadata={"help": "process all splits in the dataset instead of just the specified split"},
    )
    seed: int = field(
        default=42,
        metadata={"help": "random seed for Bernoulli sampling"},
    )
    save_as_default: bool = field(
        default=False,
        metadata={"help": "the bt_probs and scores don't have the suffix {suffix}"},
    )


# Load dataset and remove unnecessary columns
def load_generated_dataset(script_args):
    response_dataset = load_dataset(
        HUGGINGFACE_CONFIGS["prefix"]["evaluations"] + script_args.dataset,
        name=script_args.tag,
        cache_dir=script_args.dataset_cache_dir,
    )

    # Get the specified split
    if script_args.split in response_dataset:
        eval_dataset = response_dataset[script_args.split]
    else:
        print(f"Warning: Split '{script_args.split}' not found. Available splits: {list(response_dataset.keys())}")
        # Try to use 'default' split, if not available use the first available split
        if "default" in response_dataset:
            eval_dataset = response_dataset["default"]
        else:
            first_split = list(response_dataset.keys())[0]
            print(f"Using first available split: {first_split}")
            eval_dataset = response_dataset[first_split]
    
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    return eval_dataset

def sanitize_model_name(model_id):
    """Convert model ID to a safe column name suffix"""
    model_name = model_id.split("/")[-1]
    # Replace problematic characters with underscores
    model_name = model_name.replace("-", "_").replace(".", "_")
    # Truncate if too long
    if len(model_name) > 20:
        model_name = model_name[:20]
    return model_name

# Load model and tokenizer for inference
def load_score_model(script_args):
    print(f"Loading score model: {script_args.score_model_id}")
    
    # Generic model loading - try different model classes in order of preference
    model = None
    error_messages = []
    
    # Common model loading arguments
    common_args = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": accelerator.local_process_index},
        "use_cache": True,
        "cache_dir": script_args.model_cache_dir,
    }
    
    # Try AutoModelForScore first (for PKU-Alignment models)
    if HAS_SAFE_RLHF:
        try:
            model = AutoModelForScore.from_pretrained(
                script_args.score_model_id,
                trust_remote_code=False,
                **common_args
            )
            model._model_type = "score"
        except Exception as e:
            error_messages.append(f"AutoModelForScore failed: {str(e)}")
    
    # Try AutoModelForSequenceClassification
    if model is None:
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                script_args.score_model_id,
                **common_args
            )
            model._model_type = "sequence_classification"
        except Exception as e:
            error_messages.append(f"AutoModelForSequenceClassification failed: {str(e)}")
    
    # Try AutoModel with trust_remote_code=True (for custom models like openbmb)
    if model is None:
        try:
            model = AutoModel.from_pretrained(
                script_args.score_model_id,
                trust_remote_code=True,
                **common_args
            )
            model._model_type = "custom"
        except Exception as e:
            error_messages.append(f"AutoModel with trust_remote_code failed: {str(e)}")
    
    # Try AutoModel without trust_remote_code
    if model is None:
        try:
            model = AutoModel.from_pretrained(
                script_args.score_model_id,
                **common_args
            )
            model._model_type = "auto"
        except Exception as e:
            error_messages.append(f"AutoModel failed: {str(e)}")
    
    if model is None:
        raise ValueError(f"Failed to load model {script_args.score_model_id}. Errors encountered:\n" + "\n".join(error_messages))
    
    print(f"Successfully loaded model using {model._model_type} model class")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.score_model_id,
        model_max_length=script_args.model_max_length,
        padding_side="left",
        use_fast=True,
        cache_dir=script_args.model_cache_dir,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


# Evaluate reward of responses
def evaluate_reward(model, tokenizer, response_dataset, script_args):
    model_name = sanitize_model_name(script_args.score_model_id)
    suffix = f"_{model_name}" if not script_args.save_as_default else ""

    generator, num_iters = sample_every_k_batched(
        response_dataset,
        script_args.every_k,
        batch_size=script_args.per_device_evalreward_batch_size,
    )
    # Progress bar only on main process
    pbar = tqdm(total=num_iters, disable=not accelerator.is_local_main_process)
    all_indices_and_samples = list(generator)
    with accelerator.split_between_processes(
        all_indices_and_samples
    ) as process_indices_and_samples:
        # Inference in batch
        result = {}
        for indices, samples in process_indices_and_samples:
            with torch.no_grad():
                prompts = [sample["prompt"] for sample in samples]
                chosens = [sample["chosen"] for sample in samples]
                rejecteds = [sample["rejected"] for sample in samples]

                chosen_texts = [p + c for p, c in zip(prompts, chosens)]
                rejected_texts = [p + r for p, r in zip(prompts, rejecteds)]

                chosen_inputs = tokenizer(
                    chosen_texts,
                    max_length=script_args.model_max_length,
                    truncation=script_args.truncation,
                    padding=script_args.padding,
                    return_tensors="pt",
                ).to(model.device)
                rejected_inputs = tokenizer(
                    rejected_texts,
                    max_length=script_args.model_max_length,
                    truncation=script_args.truncation,
                    padding=script_args.padding,
                    return_tensors="pt",
                ).to(model.device)

                chosen_outputs = model(**chosen_inputs)
                rejected_outputs = model(**rejected_inputs)

                # Generic score extraction based on model type
                if hasattr(model, '_model_type'):
                    if model._model_type == "score":
                        chosen_scores = chosen_outputs.end_scores.cpu().flatten().tolist()
                        rejected_scores = rejected_outputs.end_scores.cpu().flatten().tolist()
                    elif model._model_type == "sequence_classification":
                        chosen_scores = chosen_outputs.logits.cpu().flatten().tolist()
                        rejected_scores = rejected_outputs.logits.cpu().flatten().tolist()
                    elif model._model_type in ["custom", "auto"]:
                        chosen_scores = chosen_outputs.cpu().flatten().tolist()
                        rejected_scores = rejected_outputs.cpu().flatten().tolist()
                    else:
                        raise RuntimeError(f"Unknown model type: {model._model_type}")
                else:
                    # Fallback: try to extract scores from common output attributes
                    if hasattr(chosen_outputs, 'end_scores'):
                        chosen_scores = chosen_outputs.end_scores.cpu().flatten().tolist()
                        rejected_scores = rejected_outputs.end_scores.cpu().flatten().tolist()
                    elif hasattr(chosen_outputs, 'logits'):
                        chosen_scores = chosen_outputs.logits.cpu().flatten().tolist()
                        rejected_scores = rejected_outputs.logits.cpu().flatten().tolist()
                    elif torch.is_tensor(chosen_outputs):
                        chosen_scores = chosen_outputs.cpu().flatten().tolist()
                        rejected_scores = rejected_outputs.cpu().flatten().tolist()
                    else:
                        raise RuntimeError(f"Unable to extract scores from model outputs. Output type: {type(chosen_outputs)}")

                bt_probs = [float(torch.sigmoid(torch.tensor(cs - rs))) for cs, rs in zip(chosen_scores, rejected_scores)]

                # Store results
                for i, (idx, cs, rs, prob) in enumerate(zip(indices, chosen_scores, rejected_scores, bt_probs)):
                    if idx not in result:
                        result[idx] = {
                            f"chosen_score{suffix}": [],
                            f"rejected_score{suffix}": [],
                            f"bt_prob{suffix}": []
                        }
                    result[idx][f"chosen_score{suffix}"].append(cs)
                    result[idx][f"rejected_score{suffix}"].append(rs)
                    result[idx][f"bt_prob{suffix}"].append(prob)

            if accelerator.is_local_main_process:
                # Update progress bar
                pbar.update(accelerator.num_processes)
        # Transform to list of dicts, otherwise gather_object() will not collect correctly
        result = [result]

        gathered = gather_object(result)
        results = {}
        for d in gathered:
            for k, v in d.items():
                if k not in results:
                    results[k] = {
                        f"chosen_score{suffix}": [],
                        f"rejected_score{suffix}": [],
                        f"bt_prob{suffix}": []
                    }
                results[k][f"chosen_score{suffix}"].extend(v[f"chosen_score{suffix}"])
                results[k][f"rejected_score{suffix}"].extend(v[f"rejected_score{suffix}"])
                results[k][f"bt_prob{suffix}"].extend(v[f"bt_prob{suffix}"])

        def get_column(col):
            return [
                float(np.mean(results[idx][col])) if idx in results and results[idx][col] else None 
                for idx in range(len(response_dataset))
            ]

        for col in [
            f"chosen_score{suffix}",
            f"rejected_score{suffix}",
            f"bt_prob{suffix}"
        ]:
            if col in response_dataset.column_names:
                response_dataset = response_dataset.remove_columns(col)
            response_dataset = response_dataset.add_column(col, get_column(col))

        return response_dataset


def main():
    print('Start Label BT (Bradley-Terry Labeling) script!!')
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Set random seed for reproducible Bernoulli sampling
    np.random.seed(script_args.seed)

    # Model & Tokenizer
    model, tokenizer = load_score_model(script_args)
    model.eval()


    if script_args.process_all_splits:
        # Load the full DatasetDict (all splits)
        response_dataset_dict = load_dataset(
            HUGGINGFACE_CONFIGS["prefix"]["evaluations"] + script_args.dataset,
            name=script_args.tag,
            cache_dir=script_args.dataset_cache_dir,
        )

        # Evaluate each split and store results
        result_dict = {}
        for split_name, split_dataset in response_dataset_dict.items():
            print(f"Evaluating split: {split_name} (size: {len(split_dataset)})")
            result_dict[split_name] = evaluate_reward(model, tokenizer, split_dataset, script_args)

        # Save and push with all splits preserved
        DatasetDict(result_dict).push_to_hub(
            HUGGINGFACE_CONFIGS["prefix"]["evaluations"] + script_args.dataset,
            script_args.tag,
        )
    else:
       raise NotImplementedError("Processing only a single split is not implemented yet.")

    print('Finished Label BT script - dataset labeled with Bradley-Terry probabilities')


if __name__ == "__main__":
    main()