import os
import sys
import shutil
import hashlib
import socket
from dataclasses import dataclass, field
from typing import Optional
from tqdm.auto import tqdm
import torch
from huggingface_hub import HfApi, add_collection_item
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from accelerate import Accelerator
from transformers.integrations import deepspeed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
from scipy.special import expit, logit

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import CONFIGS, format_run_name, wandb_init
from utils.utils import load_and_format_dataset, sanitize_model_name
from cdpo import SFTTrainer

HUGGINGFACE_CONFIGS = CONFIGS.services.huggingface
HFAPI = HfApi(HUGGINGFACE_CONFIGS["token"])
MODEL_CONFIGS = CONFIGS.models.names
DATASET_CONFIGS = CONFIGS.datasets.preprocess
LORA_MODULES = CONFIGS.models.lora_modules
CACHE_CONFIGS = CONFIGS.utils.cache
HASHCODE = hashlib.sha1(socket.gethostname().encode()).hexdigest()

accelerator = Accelerator()
tqdm.pandas()


@dataclass
class ScriptArguments:
    model: str = field(metadata={"help": "base model name"})
    dataset: str = field(
        metadata={"help": "dataset name"},
    )
    tag: str = field(
        metadata={"help": "tag for the experiment"},
    )
    # Use LoRA by default, full training not supported
    use_lora: bool = field(
        default=True, metadata={"help": "use LoRA by default, do not change"}
    )
    lora_r: int = field(default=64, metadata={"help": "LoRA r"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_bias: str = field(default="none", metadata={"help": "LoRA bias"})
    use_q_lora: bool = field(default=False, metadata={"help": "used QLoRA"})
    use_flash_attn: bool = field(default=False, metadata={"help": "use flash attention"})
    model_cache_dir: str = field(
        default=CACHE_CONFIGS["model_cache_dir"],
        metadata={"help": "model cache directory"},
    )
    dataset_cache_dir: str = field(
        default=CACHE_CONFIGS["dataset_cache_dir"],
        metadata={"help": "dataset cache directory"},
    )
    reward_model: Optional[str] = field(
        default=None, metadata={"help": "reward model name for resampling labels"}
    )
    noise_type: Optional[str] = field(
        default=None, metadata={"help": "type of noise to apply (e.g., label_switching, bt_prob_gauss)"}
    )
    noise_level: Optional[float] = field(
        default=None, metadata={"help": "level of noise to apply"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    bf16: bool = field(default=True, metadata={"help": "use bfloat16"})
    max_steps: int = field(
        default=-1, metadata={"help": "maximum number of training steps"}
    )
    evaluation_strategy: str = field(
        default="epoch", metadata={"help": "evaluation strategy"}
    )
    save_strategy: str = field(default="epoch", metadata={"help": "save strategy"})
    load_best_model_at_end: bool = field(
        default=False, metadata={"help": "load best model at end"}
    )
    metric_for_best_model: str = field(
        default="loss", metadata={"help": "metric for best model"}
    )
    optim: str = field(default="adamw_torch", metadata={"help": "optimizer to use"})
    weight_decay: float = field(
        default=0.01, metadata={"help": "weight decay for optimizer"}
    )
    adam_beta2: float = field(
        default=0.95, metadata={"help": "adam beta2 for optimizer"}
    )
    warmup_ratio: float = field(
        default=0.05, metadata={"help": "warmup ratio for optimizer"}
    )
    lr_scheduler_type: str = field(
        default="cosine", metadata={"help": "lr scheduler type"}
    )
    gradient_accumulation_steps: int = field(
        default=4, metadata={"help": "gradient accumulation steps"}
    )
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "use gradient checkpointing"}
    )
    logging_first_step: bool = field(default=True, metadata={"help": "log first step"})
    logging_steps: int = field(default=1, metadata={"help": "logging steps interval"})
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "maximum sequence length, sequences will be right padded (and possibly truncated)"
        },
    )
    report_to: str = field(default="wandb", metadata={"help": "report to"})
    hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "hub model id, do not specify manually"}
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "output dir, do not specify manually"},
    )


# Load model and tokenizer and configure ZeRO, LoRA for training
def load_and_config_model(script_args, training_args):
    config = AutoConfig.from_pretrained(
        MODEL_CONFIGS[script_args.model],
        cache_dir=script_args.model_cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False
    # device_map = None
    device_map = {"": Accelerator().local_process_index}
    if script_args.use_q_lora and deepspeed.is_deepspeed_zero3_enabled():
        raise RuntimeError("ZeRO3 is incompatible with QLoRA.")
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIGS[script_args.model],
        torch_dtype=compute_dtype,
        device_map=device_map,
        quantization_config=(
            # Use 4-bit quantization as default QLoRA
            BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            if script_args.use_lora and script_args.use_q_lora
            else None
        ),
        low_cpu_mem_usage=not deepspeed.is_deepspeed_zero3_enabled(),
        # Update transformers package to >=4.38.0 and no need to use trust_remote_code
        trust_remote_code=True,
        # Use flash attention if specified
        attn_implementation="flash_attention_2" if script_args.use_flash_attn else None,
        use_cache=False if training_args.gradient_checkpointing else True,
        cache_dir=script_args.model_cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS[script_args.model],
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        cache_dir=script_args.model_cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if script_args.use_lora:
        for model_prefix in LORA_MODULES.keys():
            if script_args.model.startswith(model_prefix):
                lora_target_modules = LORA_MODULES[model_prefix]
                break
        lora_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=script_args.lora_dropout,
            bias=script_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if script_args.use_q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        model = get_peft_model(model, lora_config)
        # Print peft trainable params
        model.print_trainable_parameters()
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    return model, tokenizer


# Train the model using SFTTrainer
def train(model, tokenizer, train_dataset, eval_dataset, script_args, training_args):
    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        # Packing is more efficient, set max_steps=-1 to use num_train_epochs
        packing=True,
        max_seq_length=training_args.model_max_length,
    )
    trainer.train()
    return trainer


def main(script_args: ScriptArguments, training_args: TrainingArguments):
    # Adjust configs
    run_name = format_run_name(
        pipeline="SFT",
        model=script_args.model,
        dataset=script_args.dataset,
        extra_params={
            "reward_model": sanitize_model_name(script_args.reward_model),
            "noise_type": script_args.noise_type,
            "noise_level": script_args.noise_level,
        },
    )
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.hub_model_id = HUGGINGFACE_CONFIGS["prefix"]["models"] + run_name
    training_args.output_dir = os.path.join(
        CACHE_CONFIGS["checkpoint_cache_dir"], run_name + "_" + HASHCODE
    )

    # Model & Tokenizer
    model, tokenizer = load_and_config_model(script_args, training_args)

    # Dataset
    train_dataset, eval_dataset = load_and_format_dataset(script_args, format_type="sft")

    # WandB setup
    if accelerator.is_local_main_process:
        wandb_init(run_name, script_args, training_args)

    # Training
    trainer = train(
        model, tokenizer, train_dataset, eval_dataset, script_args, training_args
    )

    if accelerator.is_local_main_process:
        # Push to Hub
        trainer.push_to_hub(revision=script_args.tag)
        # add_collection_item(
        #     collection_slug=HUGGINGFACE_CONFIGS["collections"]["models"],
        #     item_id=training_args.hub_model_id,
        #     item_type="model",
        #     exists_ok=True,
        # )
        # Remove checkpoint cache
        shutil.rmtree(
            os.path.join(
                CACHE_CONFIGS["checkpoint_cache_dir"], run_name + "_" + HASHCODE
            )
        )


if __name__ == "__main__":
    parser = HfArgumentParser(
        (
            ScriptArguments,
            TrainingArguments,
        )
    )
    (
        script_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()
    main(script_args, training_args)
