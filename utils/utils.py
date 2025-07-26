import re
import wandb
from dataclasses import asdict
import numpy as np
from scipy.special import logit, expit
from datasets import load_dataset
from huggingface_hub import HfApi
from typing import Optional
from utils import CONFIGS

TASK_CONFIGS = CONFIGS.tasks
WANDB_CONFIGS = CONFIGS.services.wandb
PARAMS_CONFIGS = CONFIGS.cdpo.params
HUGGINGFACE_CONFIGS = CONFIGS.services.huggingface
REWARD_CONFIGS = CONFIGS.models.reward


# Format the command line arguments
def format_args(value):
    if isinstance(value, str):
        return value
    elif isinstance(value, bool):
        return "yes" if value else "no"
    elif isinstance(value, int):
        return str(value)
    elif isinstance(value, float):
        if (value * 100).is_integer():
            return "{:.2f}".format(value)
        else:
            return "{:.2e}".format(value)
    return value


# Format the run name
def format_run_name(pipeline, model, dataset, extra_params):
    if pipeline == "SFT":
        configs = ""
    else:
        if pipeline not in PARAMS_CONFIGS:
            raise ValueError(f"Unknown pipeline name: {pipeline}")
        required_params = PARAMS_CONFIGS[pipeline]
        configs = "".join(
            [
                (param_name if param_name != "loss_type" else "") + str(extra_params[param_name])
                for param_name in required_params
            ]
        )

    # this part is for the reward evaluator, noise type and noise level
    if "resample_model" in extra_params and extra_params["resample_model"] is not None:
        configs += str(extra_params["resample_model"])
    if "noise_type" in extra_params and extra_params["noise_type"] is not None:
        configs +=  str(extra_params["noise_type"]) + str(extra_params["noise_level"])

    run_name = pipeline + "_" + model + "_" + dataset + ("_" if configs else "") + configs

    if len(run_name) > 96:
        raise ValueError(
            f"Run name '{run_name}' exceeds 96 characters. This will create a problem with HuggingFace. Please shorten the name."
        )
    
    return run_name


# Generate sweep tasks
def generate_sweep_tasks():
    dpo_tasks = []
    tag = TASK_CONFIGS["tag"]
    models = TASK_CONFIGS["model"]
    datasets = TASK_CONFIGS["dataset"]
    for task in TASK_CONFIGS["tasks"]:
        pipeline = task["pipeline"]
        extra_fields_list = [{}]
        for param, values in task.items():
            if param != "pipeline":
                new_extra_fields_list = []
                for extra_fields in extra_fields_list:
                    for value in values:
                        new_extra_fields = extra_fields.copy()
                        new_extra_fields[param] = value
                        new_extra_fields_list.append(new_extra_fields)
                extra_fields_list = new_extra_fields_list
        for model in models:
            for dataset in datasets:
                for extra_fields in extra_fields_list:
                    task_config = {
                        "pipeline": pipeline,
                        "model": model,
                        "dataset": dataset,
                        "tag": tag,
                        **extra_fields,
                    }
                    dpo_tasks.append(task_config)
    sft_tasks = [
        dict(t)
        for t in {
            tuple(d.items())
            for d in [
                {k: v for k, v in t.items() if k in ["model", "dataset", "tag"]}
                for t in dpo_tasks
            ]
        }
    ]
    commands = []
    if "SFT" in TASK_CONFIGS["pipelines"]:
        for task in sft_tasks:
            commands.append(
                "cdpo sft "
                + " ".join([f"--{k} {format_args(v)}" for k, v in task.items()]),
            )
    for task in dpo_tasks:
        for pipeline in [
            p
            for p in ["DPO", "GEN", "EVALREWARD", "EVALGPT"]
            if p in TASK_CONFIGS["pipelines"]
        ]:
            commands.append(
                f"cdpo {pipeline.lower()} "
                + " ".join([f"--{k} {format_args(v)}" for k, v in task.items()]),
            )
    return commands


# Initialize wandb
def wandb_init(run_name, script_args, training_args):
    wandb.init(
        # mode="offline",
        project=WANDB_CONFIGS["project"],
        entity=WANDB_CONFIGS["team"],
        name=script_args.tag + "-" + run_name,
        config={
            k: v
            for args in [
                script_args,
                training_args,
            ]
            for k, v in asdict(args).items()
        },
    )


# Sample every k samples from the dataset with batched sampling
def sample_every_k_batched(dataset, every_k, batch_size):
    # Ensure every_k is a positive float
    every_k = float(every_k)
    assert every_k > 0, "every_k must be a positive value"
    num_rows = len(dataset)

    if every_k >= 1:
        # Round every_k to the nearest integer
        step = int(np.round(every_k))
        total_samples = len(range(0, num_rows, step))
    else:
        # Calculate round(1 / every_k) and sample each row for that many times
        step = int(np.round(1 / every_k))
        total_samples = num_rows * step

    # Calculate the total number of iterations, considering the batch size
    num_iters = (total_samples + batch_size - 1) // batch_size

    def generator():
        if every_k >= 1:
            sampled_indices = list(range(0, num_rows, step))
        else:
            sampled_indices = list(np.repeat(np.arange(num_rows), step))

        # Yield batches
        for i in range(0, len(sampled_indices), batch_size):
            batch_indices = sampled_indices[i : i + batch_size]
            batch_samples = [
                dataset[int(idx)] if idx < num_rows else None for idx in batch_indices
            ]  # Handles out-of-bound indices and ensures proper key type
            # Filter out any None values that may appear if indices go out of bounds
            valid_batch_indices_samples = [
                (idx, samp)
                for idx, samp in zip(batch_indices, batch_samples)
                if samp is not None
            ]
            if valid_batch_indices_samples:
                yield zip(
                    *valid_batch_indices_samples
                )  # Unzip to separate indices and samples

    return generator(), num_iters


def sanitize_model_name(model_id: Optional[str]) -> Optional[str]:
    """Convert model ID to a safe column name suffix"""
    if model_id is None:
        return None
    model_name = model_id.split("/")[-1]
    # Replace problematic characters with underscores
    model_name = model_name.replace("-", "_").replace(".", "_")
    # Truncate if too long
    if len(model_name) > 20:
        model_name = model_name[:20]
    return model_name  


def apply_noise_and_resampling(dataset, script_args, is_training=True):
    """Helper function to handle noise injection and label resampling for both SFT and DPO"""
    
    # 1. We need to ensure the bt_noise column exists. First, check if a resampling model was given,
    # in which case we will use the column bt_prob_{resample_model} if it exists. Next, search if a default
    # bt_noise column exists, which is bt_prob_{dataset_name} where dataset_name is the name of the dataset.
    # lastly if such column does not exist, try just bt_prob, which is the backwards compatibility case.
    # If none of these columns exist, raise an error.

    # 1. Ensure bt_prob is directly assigned from bt_prob_{resample_model} if resample_model is provided
    sanitized_resample_model = None
    if hasattr(script_args, 'resample_model') and script_args.resample_model:
        sanitized_resample_model = sanitize_model_name(script_args.resample_model)
    else:
        # try default reward model of the dataset
        # replace numeric characters with ""
        dataset_name = re.sub(r'\d+', '', script_args.dataset)
        default_reward_model = REWARD_CONFIGS[dataset_name]['id']
        sanitized_resample_model = sanitize_model_name(default_reward_model)

    if sanitized_resample_model and f"bt_prob_{sanitized_resample_model}" in dataset.column_names:
        model_column = f"bt_prob_{sanitized_resample_model}"
        dataset = dataset.rename_column(model_column, "bt_prob")
    elif "bt_probs" in dataset.column_names:
        # check if bt_probs is present, which is the case for RMAB datasets
        dataset = dataset.rename_column("bt_probs", "bt_prob")
    else:
        # check if bt_prob is already present, backwards compatibility case
        if "bt_prob" not in dataset.column_names:
            raise ValueError("bt_prob column is missing in the dataset. Please ensure it is present.")
    
    # Now inject noise, only needed in training
    if is_training and script_args.noise_type:
        def apply_noise(example):
            prob = example["bt_prob"]
            if script_args.noise_type == "bt_noise_gauss":
                noise = np.random.normal(0, script_args.noise_level)
                example["bt_prob"] = float(expit(logit(prob + noise)))
            elif script_args.noise_type == "bt_noise_adv":
                noise = script_args.noise_level * np.random.uniform(0, 1) * np.sign(prob - 0.5)
                example["bt_prob"] = float(np.clip(prob - noise, 0, 1))
            elif script_args.noise_type == "bt_noise_flip":
                example["bt_prob"] = 1 - prob if np.random.random() < script_args.noise_level else prob
            elif script_args.noise_type == "label_switching":
                # This noise must be applied after sampling since it doesn't change bt_probs.
                pass
            else:
                raise ValueError(f"Unknown noise type: {script_args.noise_type}")
            return example
        dataset = dataset.map(apply_noise, num_proc=4, desc="Applying noise to bt_prob")

    # 2. Resample labels according to bt_prob
    if is_training and sanitized_resample_model is not None:
        def resample_labels(example):
            prob = example["bt_prob"]
            label = np.random.choice(["y1", "y2"], p=[prob, 1 - prob])

            # Flip again labels if noise type is label_switching
            if is_training and script_args.noise_type == "label_switching":
                if np.random.random() < script_args.noise_level:
                    label = "y2" if label == "y1" else "y1"

            if label == "y2":
                example["chosen"], example["rejected"] = example["rejected"], example["chosen"]
                example["bt_prob"] = 1 - prob

            return example
        
        dataset = dataset.map(resample_labels, num_proc=4, desc="Resampling labels based on bt_prob")

    # 3. Apply addition label switching noise if needed / does not affect bt_prob
    if is_training and script_args.noise_type == "label_switching":
        def apply_label_switching_noise(example):
            if np.random.random() < script_args.noise_level:
                example["chosen"], example["rejected"] = example["rejected"], example["chosen"]
                example["bt_prob"] = 1 - example["bt_prob"]
            return example
        
        dataset = dataset.map(apply_label_switching_noise, num_proc=4, desc="Applying label switching noise")

    return dataset


def load_and_format_dataset(script_args, format_type):
    """Load processed RLHF dataset and format for SFT or DPO training"""
    dataset = load_dataset(
        HUGGINGFACE_CONFIGS["prefix"]["datasets"] + script_args.dataset,
        cache_dir=getattr(script_args, 'dataset_cache_dir', None),
    )

    train_dataset = apply_noise_and_resampling(dataset["train"], script_args, is_training=True)
    eval_dataset = apply_noise_and_resampling(dataset["eval"], script_args, is_training=False)

    if format_type == "sft":
        def format_func(sample):
            sample["text"] = sample["prompt"] + sample["chosen"]
            return sample

        train_dataset = train_dataset.map(format_func, num_proc=4, desc="Formatting SFT train dataset")
        eval_dataset = eval_dataset.map(format_func, num_proc=4, desc="Formatting SFT eval dataset")
   
    return train_dataset, eval_dataset


def get_process_output_resource(process, script_args):
    """Get the output resource (model or dataset) for a given process"""
    if process == "sft":
        run_name = format_run_name(
            pipeline="SFT",
            model=script_args.model,
            dataset=script_args.dataset,
            extra_params={
                "resample_model": sanitize_model_name(script_args.resample_model) if script_args.resample_model else None,
                "noise_type": script_args.noise_type,
                "noise_level": script_args.noise_level,
            },
        )
        return {
            "type": "model",
            "id": HUGGINGFACE_CONFIGS["prefix"]["models"] + run_name
        }
    
    elif process == "dpo":
        run_name = format_run_name(
            pipeline=script_args.pipeline,
            model=script_args.model,
            dataset=script_args.dataset,
            extra_params={
                "beta": script_args.beta,
                "loss_type": script_args.loss_type,
                "resample_model": sanitize_model_name(script_args.resample_model) if script_args.resample_model else None,
                "noise_type": script_args.noise_type,
                "noise_level": script_args.noise_level,
            },
        )
        return {
            "type": "model",
            "id": HUGGINGFACE_CONFIGS["prefix"]["models"] + run_name
        }
    
    elif process in ["generate", "evalreward"]:
        run = script_args.run if hasattr(script_args, 'run') else script_args.run_name
        return {
            "type": "dataset",
            "id": HUGGINGFACE_CONFIGS["prefix"]["evaluations"] + run
        }
    
    else:
        raise ValueError(f"Unknown process: {process}")


def check_resource_exists(process, script_args, tag):
    """Check if the output resource for a process exists on HuggingFace Hub"""
    resource_info = get_process_output_resource(process, script_args)
    
    if process != "evalreward":
        api = HfApi()

        try:
            if resource_info["type"] == "model":
                api.model_info(resource_info["id"], revision=tag)
            elif resource_info["type"] == "dataset":
                api.dataset_info(resource_info["id"])
            
            return True
        except:
            return False
    else:
        # download dataset and check column exists
        dataset = load_dataset(
            HUGGINGFACE_CONFIGS["prefix"]["datasets"] + script_args.run_name,
            cache_dir=getattr(script_args, 'dataset_cache_dir', None),
        )

        return ("reward_score_generated" in dataset.column_names)
