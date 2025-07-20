#!/usr/bin/env python3


import argparse


# Direct imports from pipelines
from pipelines.sft import main as sft_main, ScriptArguments as ScriptArgumentsSFT, TrainingArguments as TrainingArgumentsSFT
from pipelines.dpo import main as dpo_main, ScriptArguments as ScriptArgumentsDPO, TrainingArguments as TrainingArgumentsDPO
from pipelines.generate import main as generate_main, ScriptArguments as ScriptArgumentsGenerate
from pipelines.evalreward import main as evalreward_main, ScriptArguments as ScriptArgumentsEvalReward

# Import utils
from utils import format_run_name, CONFIGS, check_resource_exists, sanitize_model_name

HUGGINGFACE_CONFIGS = CONFIGS.services.huggingface
DEVICE_CONFIGS = CONFIGS.devices.devices
OPTIM_CONFIGS = CONFIGS.pipelines.optim


def get_optimizer_params(pipeline, model, dataset):
    """Get optimizer parameters from config with model/dataset-specific multipliers."""
    # Get the default values for the pipeline
    defaults = OPTIM_CONFIGS["defaults"]["DPO" if pipeline != "SFT" else "SFT"]
    num_train_epochs = int(defaults["num_train_epochs"])
    learning_rate = float(defaults["learning_rate"])

    # Apply multipliers based on the dataset and model
    if dataset in OPTIM_CONFIGS["multipliers"]["num_train_epochs"]:
        num_train_epochs *= OPTIM_CONFIGS["multipliers"]["num_train_epochs"][dataset]

    if model in OPTIM_CONFIGS["multipliers"]["learning_rate"]:
        learning_rate *= OPTIM_CONFIGS["multipliers"]["learning_rate"][model]

    return {"num_train_epochs": int(num_train_epochs), "learning_rate": learning_rate}


def get_batch_size_params(pipeline, gres):
    """Get batch size parameters based on GPU device configuration."""
    device_type = gres.split(":")[0] if ":" in gres else gres
    return DEVICE_CONFIGS["pipelines"][pipeline][device_type]

def create_arguments(process, args):
    """Create arguments for any process from execute arguments."""
    
    extra_params = {
        "beta": args.beta,
        "noise_type": args.noise_type,
        "noise_level": args.noise_level,
        "loss_type": args.loss_type,
        "reward_model": sanitize_model_name(args.reward_model) if args.reward_model else None,
    }

    if process == "sft":
        # Get config parameters like cdpo_cli does
        optimizer_params = get_optimizer_params("SFT", args.model, args.dataset)
        batch_params = get_batch_size_params("SFT", args.gres)
        
        script_args = ScriptArgumentsSFT(
            model=args.model,
            dataset=args.dataset,
            tag=args.tag,
            reward_model=args.reward_model,
            noise_type=args.noise_type,
            noise_level=args.noise_level,
            lora_alpha= args.lora_alpha,
            lora_r=args.lora_r,
            use_flash_attn=args.use_flash_attn,
        )
        
        training_args = TrainingArgumentsSFT(
            num_train_epochs=optimizer_params["num_train_epochs"],
            learning_rate=optimizer_params["learning_rate"],
            per_device_train_batch_size=batch_params["per_device_train_batch_size"],
            per_device_eval_batch_size=batch_params["per_device_eval_batch_size"],
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10,
            bf16=True,
            remove_unused_columns=False
        )
        
        return script_args, training_args
    
    elif process == "dpo":
        # Get config parameters like cdpo_cli does
        optimizer_params = get_optimizer_params(args.pipeline, args.model, args.dataset)
        batch_params = get_batch_size_params("DPO", args.gres)
        
        script_args = ScriptArgumentsDPO(
            pipeline=args.pipeline,
            model=args.model,
            dataset=args.dataset,
            tag=args.tag,
            beta=args.beta,
            loss_type=args.loss_type,
            reward_model=args.reward_model,
            noise_type=args.noise_type,
            noise_level=args.noise_level,
            per_device_generation_batch_size=batch_params["per_device_generation_batch_size"],
            per_device_evalreward_batch_size=batch_params["per_device_evalreward_batch_size"],
            use_flash_attn=args.use_flash_attn,
        )
        
        training_args = TrainingArgumentsDPO(
            num_train_epochs=optimizer_params["num_train_epochs"],
            learning_rate=optimizer_params["learning_rate"],
            per_device_train_batch_size=batch_params["per_device_train_batch_size"],
            per_device_eval_batch_size=batch_params["per_device_eval_batch_size"],
            gradient_accumulation_steps=batch_params["gradient_accumulation_steps"],
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10,
            bf16=True,
            remove_unused_columns=False
        )
        
        return script_args, training_args
    
    elif process == "generate":
        # Get config parameters like cdpo_cli does
        batch_params = get_batch_size_params("GEN", args.gres)
        
        # Generate run name for DPO model
        if args.noise_type:
            extra_params["noise_type"] = args.noise_type
            extra_params["noise_level"] = args.noise_level
            
        run_name = format_run_name(args.pipeline, args.model, args.dataset, extra_params)
        
        script_args = ScriptArgumentsGenerate(
            run=run_name,
            tag=args.tag,
            per_device_generation_batch_size=batch_params["per_device_generation_batch_size"],
            eval_limit=args.eval_limit,
            use_flash_attn=args.use_flash_attn,
        )
        
        return script_args
    
    elif process == "evalreward":
        # Get config parameters like cdpo_cli does
        batch_params = get_batch_size_params("EVALREWARD", args.gres)
        
        # Generate run name for DPO model
        if args.noise_type:
            extra_params["noise_type"] = args.noise_type
            extra_params["noise_level"] = args.noise_level
            
        run_name = format_run_name(args.pipeline, args.model, args.dataset, extra_params)
        
        script_args = ScriptArgumentsEvalReward(
            run_name=run_name,
            pipeline=args.pipeline,
            tag=args.tag,
            per_device_evalreward_batch_size=batch_params["per_device_evalreward_batch_size"],
        )
        
        return script_args
    
    else:
        raise ValueError(f"Unknown process: {process}")



def main(args):
    """Execute the full pipeline with parsed arguments."""

    print(f"🎯 Selected processes: {', '.join(args.processes)}")
    print(f"🖥️  GPU resources: {args.gres}")
    if args.overwrite:
        print("🔄 Overwrite mode enabled - will skip smart checking and force re-run all steps")
    else:
        print("🧠 Smart checking enabled - will skip steps if resources already exist")

    # Step 1: SFT Training
    if "sft" in args.processes:
        # Check if we should skip SFT
        script_args, training_args = create_arguments("sft", args)
        if args.overwrite or not check_resource_exists("sft", script_args, args.tag):
            print("🚀 Starting SFT Training...")
            sft_main(script_args, training_args)
            print("✅ SFT Training completed successfully")
        else:
            print("⏭️ Skipping SFT Training - model already exists")
    else:
        print("⏭️ Skipping SFT Training")

    # Step 2: DPO Training
    if "dpo" in args.processes:
        # Check if we should skip DPO
        script_args, training_args = create_arguments("dpo", args)
        if args.overwrite or not check_resource_exists("dpo", script_args, args.tag):
            print("🚀 Starting DPO Training...")
            dpo_main(script_args, training_args)
            print("✅ DPO Training completed successfully")
        else:
            print("⏭️ Skipping DPO Training - model already exists")
    else:
        print("⏭️ Skipping DPO Training")

    # Step 3: Generation
    if "generate" in args.processes:
        script_args = create_arguments("generate", args)
        if args.overwrite or not check_resource_exists("generate", script_args, args.tag):
            print("🚀 Starting Generation...")
            generate_main(script_args)
            print("✅ Generation completed successfully")
        else:
            print("⏭️ Skipping Generation - dataset already exists")

    else:
        print("⏭️ Skipping Generation")

    # Step 4: EvalReward
    if "evalreward" in args.processes:
        script_args = create_arguments("evalreward", args)
        if args.overwrite or not check_resource_exists("evalreward", script_args, args.tag):
            print("🚀 Starting EvalReward...")
            evalreward_main(script_args)
            print("✅ EvalReward completed successfully")
        else:
            print("⏭️ Skipping EvalReward - dataset already exists")
    else:
        print("⏭️ Skipping EvalReward")

    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    completed_processes = [p for p in args.processes if p in args.processes]
    print(f"✅ Completed processes: {', '.join(completed_processes)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute the full pipeline: SFT -> DPO -> Generate -> EvalReward")
    
    # Required arguments
    parser.add_argument("--model", required=True, help="Model name (e.g., Q0.5B)")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., U0)")
    parser.add_argument("--tag", required=True, help="Tag for the experiment (e.g., tag1)")
    
    # Optional arguments
    parser.add_argument("--pipeline", help="DPO pipeline name (e.g., DPO, DPO-DRO)")
    parser.add_argument("--reward_model", default="openbmb/Eurus-RM-7b", help="Reward model name")
    parser.add_argument("--noise_type", help="Type of noise (e.g., label_switching, bt_noise_gauss)")
    parser.add_argument("--noise_level", type=float, help="Level of noise (e.g., 0.4, 0.5)")
    parser.add_argument("--beta", type=float, help="Beta parameter for DPO (e.g., 0.1)")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA r value (default: 64)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha value (default: 16)")
    parser.add_argument("--loss_type", help="Loss type (e.g., generalized_sigmoid)")
    parser.add_argument("--model_max_length", type=int, default=1024, help="Maximum sequence length for the model (default: 1024)")
    
    # GPU resource specification (required for config-driven parameters)
    parser.add_argument("-g", "--gres", required=True, help="GPU resources (e.g., A100:1, A100) for automatic parameter configuration")
    
    # Evaluation parameters
    parser.add_argument("--eval_limit", type=int, default=-1, help="Limit for evaluation samples (-1 for no limit)")

    # Flash attention usage
    parser.add_argument("--use_flash_attn", action="store_true", help="Use flash attention for training/generation")
    
    # Process selection``
    parser.add_argument("--processes", nargs="+", 
                       choices=["sft", "dpo", "generate", "evalreward"],
                       default=["sft", "dpo", "generate", "evalreward"],
                       help="Specify which processes to execute (default: all)")
    
    # Overwrite control
    parser.add_argument("--overwrite", action="store_true", 
                       help="Force overwrite existing models/datasets (skip smart checking)")
    
    args = parser.parse_args()
    main(args)
