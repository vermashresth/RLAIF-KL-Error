from functools import partial
import os
import sys
from dataclasses import dataclass, field
from tqdm.auto import tqdm
import torch
from huggingface_hub import HfApi, add_collection_item
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from peft import PeftModel
from accelerate import Accelerator
from accelerate.utils import gather_object

from utils import CONFIGS, rmab_format_func


HUGGINGFACE_CONFIGS = CONFIGS.services.huggingface
HFAPI = HfApi(HUGGINGFACE_CONFIGS["token"])
MODEL_CONFIGS = CONFIGS.models.names
DATASET_CONFIGS = CONFIGS.datasets.preprocess
CACHE_CONFIGS = CONFIGS.utils.cache

accelerator = Accelerator()
tqdm.pandas()
transformers.logging.set_verbosity_error()


@dataclass
class ScriptArguments:
    run: str = field(
        metadata={"help": "run name to generate"},
    )
    tag: str = field(
        metadata={"help": "tag for the experiment"},
    )
    split: str = field(
        default="eval",
        metadata={"help": "split to evaluate"},
    )
    eval_limit: int = field(
        default=-1,
        metadata={"help": "limit the number of samples to evaluate"},
    )
    eval_dataset: str = field(
        default=None,
        metadata={"help": "name of the evaluation dataset, if None, uses the run name"},
    )
    upload_name: str = field(
        default=None,
        metadata={
            "help": "name to upload the dataset to HuggingFace, if None, uses run name"
        },
    )
    bf16: bool = field(default=True, metadata={"help": "use bfloat16"})
    fp16: bool = field(default=False, metadata={"help": "use float16"})
    model_max_length: int = field(
        default=2048,
        metadata={"help": "maximum sequence length for the model"},
    )
    per_device_generation_batch_size: int = field(
        default=4,
        metadata={"help": "batch size per device"},
    )
    # Use LoRA by default, full training not supported
    use_lora: bool = field(
        default=True, metadata={"help": "use LoRA by default, do not change"}
    )
    use_flash_attn: bool = field(default=False, metadata={"help": "use flash attention"})
    num_beams: int = field(
        default=1,
        metadata={"help": "number of beams for beam search"},
    )
    do_sample: bool = field(
        default=False,
        metadata={"help": "use sampling for generation"},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "temperature for sampling"},
    )
    use_contrastive_search: bool = field(
        # Use contrastive search to improve the quality of small models
        default=True,
        metadata={"help": "use contrastive search for generation"},
    )
    penalty_alpha: float = field(
        default=0.6,
        metadata={"help": "repetition penalty"},
    )
    top_k: int = field(
        default=4,
        metadata={"help": "top k tokens to sample from"},
    )
    model_cache_dir: str = field(
        default=CACHE_CONFIGS["model_cache_dir"],
        metadata={"help": "model cache directory"},
    )
    dataset_cache_dir: str = field(
        default=CACHE_CONFIGS["dataset_cache_dir"],
        metadata={"help": "dataset cache directory"},
    )
    max_new_tokens: int = field(
        default=None,
        metadata={"help": "maximum number of new tokens to generate"},
    )


# Load dataset and remove unnecessary columns
def load_and_format_dataset(script_args):
    if script_args.eval_dataset is None:
        dataset_name = script_args.run.split("_")[2]

        if dataset_name == "RMAB":
            dataset_name += "_" + script_args.run.split("_")[3]
    else:
        dataset_name = script_args.eval_dataset

    dataset = load_dataset(
        HUGGINGFACE_CONFIGS["prefix"]["datasets"] + dataset_name,
        cache_dir=script_args.dataset_cache_dir,
    )

    if dataset_name.startswith("RMAB"):
        # Format RMAB dataset
        dataset[script_args.split] = dataset[script_args.split].map(
            partial(rmab_format_func, prefix="fixed_"), num_proc=4, desc="Formatting RMAB dataset"
        )

    # select_columns = ["prompt", "chosen", "rejected"]
    # eval_dataset = dataset["eval"].map(
    #     lambda sample: {col: sample[col] for col in select_columns},
    #     remove_columns=[
    #         col for col in dataset["eval"].column_names if col not in select_columns
    #     ],
    #     num_proc=4,
    # )

    # return eval_dataset
    return dataset[script_args.split]


# Load model and tokenizer for inference
def load_and_config_model(script_args):
    model_name = MODEL_CONFIGS[script_args.run.split("_")[1]]
    print(f"Loading model: {model_name}")
    
    config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=script_args.model_cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False
    compute_dtype = (
        torch.float16
        if script_args.fp16
        else (torch.bfloat16 if script_args.bf16 else torch.float32)
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=compute_dtype,
        device_map = {"": Accelerator().local_process_index},
        # Update transformers package to >=4.38.0 and no need to use trust_remote_code
        trust_remote_code=True,
        # Use flash attention if specified
        attn_implementation="flash_attention_2" if script_args.use_flash_attn else None,
        use_cache=True,
        cache_dir=script_args.model_cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=script_args.model_max_length,
        padding_side="left",
        use_fast=True,
        cache_dir=script_args.model_cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Assume PEFT is used for DPO training for now
    if script_args.use_lora:
        peft_model_id = HUGGINGFACE_CONFIGS["prefix"]["models"] + script_args.run
        print('PEFT ID: ', peft_model_id)
        model = PeftModel.from_pretrained(
            base_model,
            peft_model_id,
            revision=script_args.tag,
            is_trainable=True,
            adapter_name="default",
            cache_dir=script_args.model_cache_dir,
        )

    return model, tokenizer


def generate_responses(model, tokenizer, eval_dataset, script_args):
    # Progress bar only on main process
    pbar = tqdm(total=len(eval_dataset), disable=not accelerator.is_local_main_process)
    # Split the list of prompts to each process, note it only works for list
    all_prompts = list(eval_dataset["fixed_prompt"] if "fixed_prompt" in eval_dataset.column_names else eval_dataset["prompt"])
    with accelerator.split_between_processes(all_prompts) as process_prompts:
        dataloader = DataLoader(
            process_prompts,
            batch_size=script_args.per_device_generation_batch_size,
            # This is required to maintain the order of the prompts
            shuffle=False,
        )
        result = {"response": []}
        with torch.no_grad():
            for prompts in dataloader:
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="longest",
                    truncation=False,
                    pad_to_multiple_of=8,
                    add_special_tokens=False,
                )
                generate_kwargs = {
                    "penalty_alpha": script_args.penalty_alpha,
                    "top_k": script_args.top_k,
                }
                if script_args.do_sample:
                    generate_kwargs.update(
                        {
                            "do_sample": True,
                            "temperature": script_args.temperature,
                        }
                    )
                else:
                    generate_kwargs.update({"do_sample": False})

                if script_args.max_new_tokens is not None:
                    generate_kwargs["max_new_tokens"] = script_args.max_new_tokens
                else:
                    generate_kwargs["max_length"] = script_args.model_max_length
               
                outputs = model.generate(
                    **inputs.to(model.device),
                    num_beams=script_args.num_beams,
                    pad_token_id=tokenizer.eos_token_id,
                    **generate_kwargs if script_args.use_contrastive_search else {},
                )
                outputs = outputs[:, inputs["input_ids"].shape[1]:]  # Remove input part
                responses = tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                result["response"].extend(responses)
                if accelerator.is_local_main_process:
                    # Simulate the actual update by times the number of processes
                    pbar.update(len(prompts) * accelerator.num_processes)
        # Transform to list of dicts, otherwise gather_object() will not collect correctly
        result = [result]
    gathered = gather_object(result)
    return gathered


def main(script_args: ScriptArguments):
    print('Start Generate.py!!')

    # Model & Tokenizer
    model, tokenizer = load_and_config_model(script_args)
    model.eval()

    # Dataset
    eval_dataset = load_and_format_dataset(script_args)
    if script_args.eval_limit > 0:
        eval_dataset = eval_dataset.select(range(script_args.eval_limit))

    # Generation
    results = generate_responses(model, tokenizer, eval_dataset, script_args)

    # Push to Hub
    if accelerator.is_local_main_process:
        response_dataset = eval_dataset.add_column(
            "response",
            [response for result in results for response in result["response"]],
        )
        # also add fixed prompt, fixed chosen fixed rejects if exists
        # if "fixed_prompt" in eval_dataset.column_names:
        #     response_dataset = response_dataset.add_column(
        #         "fixed_prompt",
        #         [prompt for prompt in eval_dataset["fixed_prompt"]],
        #     )
        # if "fixed_chosen" in eval_dataset.column_names:
        #     response_dataset = response_dataset.add_column(
        #         "fixed_chosen",
        #         [chosen for chosen in eval_dataset["fixed_chosen"]],
        #     )
        # if "fixed_rejected" in eval_dataset.column_names:
        #     response_dataset = response_dataset.add_column(
        #         "fixed_rejected",
        #         [rejected for rejected in eval_dataset["fixed_rejected"]],
        #     )
        target_run = script_args.run if script_args.upload_name is None else script_args.upload_name
        DatasetDict(
            {"default": response_dataset},
        ).push_to_hub(
            HUGGINGFACE_CONFIGS["prefix"]["evaluations"] + target_run,
            script_args.tag,
        )
        # add_collection_item(
        #     collection_slug=HUGGINGFACE_CONFIGS["collections"]["evaluations"],
        #     item_id=HUGGINGFACE_CONFIGS["prefix"]["evaluations"] + script_args.run,
        #     item_type="dataset",
        #     exists_ok=True,
        # )
    print('Finished Generate.py')


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    (script_args,) = parser.parse_args_into_dataclasses()

    main(script_args)
