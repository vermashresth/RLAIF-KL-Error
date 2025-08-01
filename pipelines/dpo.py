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
    AutoModel,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from accelerate import Accelerator
from transformers.integrations import deepspeed
from peft import PeftModel
import numpy as np
from scipy.special import expit, logit
from safe_rlhf.models import AutoModelForScore
from transformers import AutoModelForSequenceClassification



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Generalized DPO training script
from cdpo import GeneralizedDPOTrainer
from utils import CONFIGS, format_run_name, wandb_init
from utils.utils import load_and_format_dataset, sanitize_model_name


HUGGINGFACE_CONFIGS = CONFIGS.services.huggingface
HFAPI = HfApi(HUGGINGFACE_CONFIGS["token"])
MODEL_CONFIGS = CONFIGS.models.names
LORA_MODULES = CONFIGS.models.lora_modules
REWARD_CONFIGS = CONFIGS.models.reward
DATASET_CONFIGS = CONFIGS.datasets.preprocess
CACHE_CONFIGS = CONFIGS.utils.cache
HASHCODE = hashlib.sha1(socket.gethostname().encode()).hexdigest()

accelerator = Accelerator()
tqdm.pandas()


@dataclass
class ScriptArguments:
    pipeline: str = field(
        metadata={"help": "pipeline variant name of DPO training"},
    )
    model: str = field(metadata={"help": "base model name"})
    dataset: str = field(
        metadata={"help": "dataset name"},
    )
    tag: str = field(
        metadata={"help": "tag for the experiment"},
    )
    sft_tag: Optional[str] = field(
        default=None, metadata={"help": "tag for the SFT model, if different from the DPO tag"}
    )
    beta: float = field(default=0.1, metadata={"help": "the beta parameter for STDPO"})
    label_smoothing: float = field(
        default=0.0, metadata={"help": "label smoothing parameter for training"}
    )
    r: float = field(
        default=0.0, metadata={"help": "sampling mixing probability for DDP"}
    )
    rho: float = field(default=0.0, metadata={"help": "gradient coefficient for DDP"})
    p: float = field(
        default=0.0, metadata={"help": "sampling mixing probability for DPP"}
    )
    pi: float = field(default=0.0, metadata={"help": "gradient coefficient for DPP"})
    g: float = field(
        default=0.0, metadata={"help": "sampling mixing probability for DPR"}
    )
    gamma: float = field(default=0.0, metadata={"help": "gradient coefficient for DPR"})
    # Use LoRA by default, full training not supported
    use_lora: bool = field(
        default=True, metadata={"help": "use LoRA by default, do not change"}
    )
    # LoRA and QLoRA parameters are not used, as we use the PEFT adapters for SFT training for now
    use_q_lora: bool = field(default=False, metadata={"help": "used QLoRA"})
    use_flash_attn: bool = field(default=False, metadata={"help": "use flash attention"})
    generate_during_eval: bool = field(
        default=False, metadata={"help": "generate during evaluation"}
    )
    generation_reuse_multiplier: int = field(
        default=1, metadata={"help": "generation reuse multiplier"}
    )
    generation_num_batches: int = field(
        default=1, metadata={"help": "number of generation batches"}
    )
    per_device_generation_batch_size: int = field(
        default=24, metadata={"help": "generation batch size per device"}
    )
    generation_temperature: float = field(
        default=0.9, metadata={"help": "generation temperature"}
    )
    per_device_evalreward_batch_size: int = field(
        default=16, metadata={"help": "reward evaluation batch size"}
    )
    model_cache_dir: str = field(
        default=CACHE_CONFIGS["model_cache_dir"],
        metadata={"help": "model cache directory"},
    )
    dataset_cache_dir: str = field(
        default=CACHE_CONFIGS["dataset_cache_dir"],
        metadata={"help": "dataset cache directory"},
    )
    dro: float = field(
        default=0.0, metadata={"help": "sampling mixing probability for DRO-DPR"}
    )
    omega: float = field(default=0.0, metadata={"help": "gradient coefficient for DRO-DPR"})
    beta_prime: float = field(default=1.0, metadata={"help": "beta_prime parameter for DR_DPO"})
    epsilon: float = field(default=0.1, metadata={"help": "epsilon parameter for RDPO"})
    loss_type: str = field(
        default="generalized_sigmoid_smooth_label",
        metadata={"help": "loss type used for training"},
    )
    noise_type: str = field(
        default=None, metadata={"help": "Type of noise to inject (e.g., bt_prob_gauss)"}
    )
    noise_level: float = field(
        default=0.0, metadata={"help": "Level of noise to inject"}
    )
    resample_model: Optional[str] = field(
        default=None, metadata={"help": "Reward model to use for bt_prob resampling"}
    )
    logit_clipping: Optional[float] = field(
        default=None, metadata={"help": "Logit clipping value"}
    )
    dro_divergence_type: str = field(
        default="chi_squared", metadata={"help": "Divergence type for DRO method: 'kl_div' or 'chi_squared'"}
    )
    nu: float = field(
        default=0.1, metadata={"help": "Parameter controlling the divergence constraint in DPO pro"}
    )
    push_dataset: bool = field(
        default=True, metadata={"help": "Whether to push the model to the Hub"}
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
    optim: str = field(default="rmsprop", metadata={"help": "optimizer to use"})
    warmup_steps: int = field(default=100, metadata={"help": "number of warmup steps"})
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "gradient accumulation steps"}
    )
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "use gradient checkpointing"}
    )
    logging_first_step: bool = field(default=True, metadata={"help": "log first step"})
    logging_steps: int = field(default=10, metadata={"help": "logging steps interval"})
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "maximum sequence length, sequences will be right padded (and possibly truncated)"
        },
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help": "remove unused columns, set to False when using DPODataCollatorWithPadding"
        },
    )
    report_to: str = field(default="wandb", metadata={"help": "report to"})
    hub_model_id: Optional[str] = field(
        default="none", metadata={"help": "hub model id, do not specify manually"}
    )
    output_dir: Optional[str] = field(
        default="none", metadata={"help": "run name for wandb"}
    )

# Load model and tokenizer and configure ZeRO, LoRA for training
def load_and_config_model(script_args, training_args):
    print('Loading model')
    base_model = MODEL_CONFIGS[script_args.model]
    print('SFT model name:', base_model)
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

    # base model kwargs incompatible with ZeRO3
    optional_base_model_kwargs = {}
    if not deepspeed.is_deepspeed_zero3_enabled():
        # add device map and low cpu mem true
        optional_base_model_kwargs = {
            "device_map": device_map,
            "low_cpu_mem_usage": True,
        }

    # Assume PEFT is used for SFT training for now
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIGS[script_args.model],
        torch_dtype=compute_dtype,
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
        # Update transformers package to >=4.38.0 and no need to use trust_remote_code
        trust_remote_code=True,
        # Use flash attention if specified
        attn_implementation="flash_attention_2" if script_args.use_flash_attn else None,
        use_cache=False if training_args.gradient_checkpointing else True,
        cache_dir=script_args.model_cache_dir,
        **optional_base_model_kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS[script_args.model],
        use_fast=True,
        use_cache=True,
        cache_dir=script_args.model_cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token
    # BOS token is not configured for Qwen1.5-* models, but is used by DPO padding strategy
    # See https://github.com/huggingface/trl/issues/1073
    if script_args.model.startswith("Q"):
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id

    # Assume PEFT is used for SFT training for now
    if script_args.use_lora:
        peft_model_id = HUGGINGFACE_CONFIGS["prefix"]["models"] + format_run_name(
            pipeline="SFT",
            model=script_args.model,
            dataset=script_args.dataset,
            extra_params={ # don't include noise to make evaluation of DPO more comparable
                "resample_model": sanitize_model_name(script_args.resample_model),
                # "noise_type": script_args.noise_type,
                # "noise_level": script_args.noise_level,
            },
        )
        print('PEFT model id:', peft_model_id)
        model = PeftModel.from_pretrained(
            base_model,
            peft_model_id,
            revision=script_args.tag if script_args.sft_tag is None else script_args.sft_tag,
            is_trainable=True,
            adapter_name="default",
            cache_dir=script_args.model_cache_dir,
        )
        # Load the adapter a second time, with a different name, which will be our reference model.
        model.load_adapter(
            peft_model_id,
            revision=script_args.tag if script_args.sft_tag is None else script_args.sft_tag,
            adapter_name="reference",
            cache_dir=script_args.model_cache_dir,
        )
        # Print peft trainable params
        model.print_trainable_parameters()
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    return model, tokenizer


# Train the model using SFTTrainer
def train(model, tokenizer, train_dataset, eval_dataset, script_args, training_args):
    # Parsing pipeline and setting
    if script_args.pipeline == "DPO":
        print('Running DPO!!!')
        assert script_args.r == 0 and script_args.rho == 0
        assert script_args.p == 0 and script_args.pi == 0
        assert script_args.g == 0 and script_args.gamma == 0
        assert script_args.dro == 0 and script_args.omega == 0
    elif script_args.pipeline == "DDP":
        print('Running DDP!!!')
        assert script_args.p == 0 and script_args.pi == 0
        assert script_args.g == 0 and script_args.gamma == 0
        assert script_args.dro == 0 and script_args.omega == 0
    elif script_args.pipeline == "DPP":
        print('Running DPP!!!')
        assert script_args.r == 0 and script_args.rho == 0
        assert script_args.g == 0 and script_args.gamma == 0
        assert script_args.dro == 0 and script_args.omega == 0
    elif script_args.pipeline == "DPR":
        print('Running DPR!!!')
        assert script_args.r == 0 and script_args.rho == 0
        assert script_args.p == 0 and script_args.pi == 0
        assert script_args.dro == 0 and script_args.omega == 0
    elif script_args.pipeline == "DRO_DPR":
        print('Running DRO-DPR!!!')
        assert script_args.r == 0 and script_args.rho == 0
        assert script_args.p == 0 and script_args.pi == 0
        assert script_args.g == 0 and script_args.gamma == 0
    elif script_args.pipeline == "DR_DPO":
        print('Running DR_DPO!!!')
        assert script_args.r == 0 and script_args.rho == 0
        assert script_args.p == 0 and script_args.pi == 0
        assert script_args.g == 0 and script_args.gamma == 0
        assert script_args.dro == 0 and script_args.omega == 0
    elif script_args.pipeline == "MIX":
        pass
    else:
        raise ValueError(f"Invalid pipeline name {script_args.pipeline}")
    assert 0 <= (script_args.r + script_args.p + script_args.g + script_args.dro) <= 1
    training_args.num_train_epochs /= 1 - script_args.r - script_args.p - script_args.g - script_args.dro
    
    # Training
    trainer = GeneralizedDPOTrainer(
        model,
        # Two adapters as the default and reference models
        None,
        model_adapter_name="default",
        ref_adapter_name="reference",
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=training_args.model_max_length,
        max_prompt_length=training_args.model_max_length // 2,
        max_target_length=training_args.model_max_length // 2,
        generate_during_eval=script_args.generate_during_eval,
        loss_type=script_args.loss_type,
        # New parameters
        generation_reuse_multiplier=script_args.generation_reuse_multiplier,
        generation_num_batches=script_args.generation_num_batches,
        per_device_generation_batch_size=script_args.per_device_generation_batch_size,
        generation_temperature=script_args.generation_temperature,
        reward_model_id=None,
        reward_model=None,
        reward_tokenizer=None,
        reward_model_reverse=None,
        per_device_evalreward_batch_size=script_args.per_device_evalreward_batch_size,
        r=script_args.r,
        rho=script_args.rho,
        p=script_args.p,
        pi=script_args.pi,
        g=script_args.g,
        gamma=script_args.gamma,
        dro=script_args.dro,
        omega=script_args.omega,
        beta_prime=script_args.beta_prime,
        epsilon=script_args.epsilon,
        label_smoothing=script_args.label_smoothing,
        dro_divergence_type=script_args.dro_divergence_type,
        dataset_num_proc=4,
        logit_clipping=script_args.logit_clipping,
        nu=script_args.nu,
    )

    reward_model, reward_tokenizer, reward_model_reverse = None, None, None
    for dataset_prefix in REWARD_CONFIGS.keys():
        if script_args.dataset.startswith(dataset_prefix):
            reward_model_id = REWARD_CONFIGS[dataset_prefix]["id"]
            reward_model_reverse = REWARD_CONFIGS[dataset_prefix]["reverse"]
            # Load reward model
            if script_args.g > 0 or script_args.dro > 0:
                if reward_model_id.startswith("PKU-Alignment"):
                    print('Loading PKU-Alignment RM')
                    reward_model = AutoModelForScore.from_pretrained(
                        reward_model_id,
                        torch_dtype=torch.bfloat16,
                        # Use auto device map since RMs are large
                        device_map={"": accelerator.local_process_index},
                        cache_dir=script_args.model_cache_dir,
                    )
                    reward_tokenizer = AutoTokenizer.from_pretrained(
                        reward_model_id,
                        model_max_length=training_args.model_max_length,
                        padding_side="left",
                        use_fast=True,
                        cache_dir=script_args.model_cache_dir,
                    )
                elif reward_model_id.startswith("OpenAssistant"):
                    print('Loading openassitanat RM')
                    reward_model = AutoModelForSequenceClassification.from_pretrained(
                        reward_model_id,
                        torch_dtype=torch.bfloat16,
                        device_map={"": accelerator.local_process_index},
                        cache_dir=script_args.model_cache_dir,
                    )
                    reward_tokenizer = AutoTokenizer.from_pretrained(
                        reward_model_id,
                        model_max_length=training_args.model_max_length,
                        padding_side="left",
                        use_fast=True,
                        cache_dir=script_args.model_cache_dir,
                    )
                elif reward_model_id.startswith("openbmb"):
                    print('Loading openbmb RM')
                    reward_model = AutoModel.from_pretrained(
                        reward_model_id,
                        torch_dtype=torch.bfloat16,
                        device_map={"": accelerator.local_process_index},
                        # This is needed as EurusRewardModel is not in transformers' model registry
                        trust_remote_code=True,
                        use_cache=True,
                        cache_dir=script_args.model_cache_dir,
                    )
                    reward_tokenizer = AutoTokenizer.from_pretrained(
                        reward_model_id,
                        model_max_length=training_args.model_max_length,
                        padding_side="left",
                        use_fast=True,
                        cache_dir=script_args.model_cache_dir,
                    )
                else:
                    raise ValueError("No reward model found for the dataset")
                reward_model = reward_model.cpu()
                reward_tokenizer.pad_token = reward_tokenizer.eos_token
            break
    trainer.reward_model_id=reward_model_id
    trainer.reward_model=reward_model
    trainer.reward_tokenizer=reward_tokenizer
    trainer.reward_model_reverse=reward_model_reverse

    train_dataset_size = len(trainer.train_dataset)
    per_device_train_batch_size = training_args.per_device_train_batch_size
    gradient_accumulation_steps = training_args.gradient_accumulation_steps
    num_train_epochs = training_args.num_train_epochs

    # Calculate the number of update steps per epoch
    steps_per_epoch = (train_dataset_size + per_device_train_batch_size - 1) // per_device_train_batch_size
    total_update_steps = steps_per_epoch // gradient_accumulation_steps * num_train_epochs

    print(f"Estimated steps per epoch: {steps_per_epoch}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Number of training epochs: {num_train_epochs}")
    print(f"Estimated total update steps: {total_update_steps}")


    trainer.train()
    print('Completed Training')
    return trainer


def main(script_args: ScriptArguments, training_args: TrainingArguments):

    # Adjust configs
    run = format_run_name(
        pipeline=script_args.pipeline,
        model=script_args.model,
        dataset=script_args.dataset,
        extra_params={
            "beta": script_args.beta,
            "r": script_args.r,
            "rho": script_args.rho,
            "p": script_args.p,
            "pi": script_args.pi,
            "g": script_args.g,
            "gamma": script_args.gamma,
            "dro": script_args.dro,
            "omega": script_args.omega,
            "beta_prime": script_args.beta_prime,
            "epsilon": script_args.epsilon,
            "loss_type": script_args.loss_type,
            "resample_model": sanitize_model_name(script_args.resample_model),
            "noise_type": script_args.noise_type,
            "noise_level": script_args.noise_level,
        },
    )
    training_args.hub_model_id = HUGGINGFACE_CONFIGS["prefix"]["models"] + run
    print(f"Hub model id: {training_args.hub_model_id}")
    training_args.output_dir = os.path.join(
        CACHE_CONFIGS["checkpoint_cache_dir"], run + "_" + HASHCODE
    )

    # Model & Tokenizer
    model, tokenizer = load_and_config_model(script_args, training_args)

    # Dataset
    train_dataset, eval_dataset = load_and_format_dataset(script_args, format_type="dpo")

    # WandB setup
    if accelerator.is_local_main_process:
        wandb_init(run, script_args, training_args)

    # Training
    trainer = train(
        model, tokenizer, train_dataset, eval_dataset, script_args, training_args
    )

    if accelerator.is_local_main_process:
        # Push to Hub
        if script_args.push_dataset:
            trainer.push_to_hub(revision=script_args.tag)
        # add_collection_item (
        #     collection_slug=HUGGINGFACE_CONFIGS["collections"]["models"],
        #     item_id=training_args.hub_model_id,
        #     item_type="model",
        #     exists_ok=True,
        # )
        # Remove checkpoint cache
        shutil.rmtree(
            os.path.join(CACHE_CONFIGS["checkpoint_cache_dir"], run + "_" + HASHCODE)
        )
    print('Completed main for dpo.py')

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
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    main(script_args, training_args)
