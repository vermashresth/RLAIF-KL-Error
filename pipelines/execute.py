#!/usr/bin/env python3

import sys
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
CDPO_PARAMS_CONFIGS = CONFIGS.cdpo.params


def get_batch_size_params(pipeline, gres):
    """Get batch size parameters based on GPU device configuration."""
    device_type = gres.split(":")[0] if ":" in gres else gres
    return DEVICE_CONFIGS["pipelines"][pipeline][device_type]


def validate_extra_args(pipeline, extra_args_dict):
    """Validate that extra arguments match the cdpo/params config for the given pipeline."""
    if pipeline not in CDPO_PARAMS_CONFIGS:
        raise ValueError(f"Pipeline '{pipeline}' not found in cdpo/params config")
    
    required_params = CDPO_PARAMS_CONFIGS[pipeline]
    
    # Check that all provided extra args are valid for this pipeline
    for arg_name in extra_args_dict.keys():
        if arg_name not in required_params:
            raise ValueError(f"Argument '{arg_name}' is not valid for pipeline '{pipeline}'. "
                           f"Valid arguments are: {required_params}")
    
    return True


def parse_extra_args(unknown_args):
    """Parse unknown arguments into a dictionary."""
    extra_args = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith('--'):
            arg_name = arg[2:]  # Remove '--' prefix
            # Check if next item exists and doesn't start with '--'
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
                value = unknown_args[i + 1]
                # Try to convert to appropriate type
                try:
                    # Try int first
                    if '.' not in value:
                        extra_args[arg_name] = int(value)
                    else:
                        extra_args[arg_name] = float(value)
                except ValueError:
                    # Keep as string if conversion fails
                    extra_args[arg_name] = value
                i += 2
            else:
                # Boolean flag
                extra_args[arg_name] = True
                i += 1
        else:
            i += 1
    return extra_args


def create_arguments(process, args, extra_args=None):
    """Create arguments for any process from execute arguments."""
    
    # Set default values for args that might not be present
    beta = getattr(args, 'beta', 0.1)  # Default beta value
    
    # Override with extra_args if provided (for DPO)
    if extra_args:
        beta = extra_args.get('beta', beta)
    
    extra_params = {
        "beta": beta,
        "noise_type": args.noise_type,
        "noise_level": args.noise_level,
        "loss_type": args.loss_type,
        "reward_model": sanitize_model_name(args.reward_model) if args.reward_model else None,
    }
    
    # Add extra arguments to extra_params if provided
    if extra_args:
        extra_params.update(extra_args)

    if process == "sft":
        # Get config parameters like cdpo_cli does
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
            num_train_epochs=args.num_sft_train_epochs,
            learning_rate=args.sft_learning_rate,
            per_device_train_batch_size=batch_params["per_device_train_batch_size"],
            per_device_eval_batch_size=batch_params["per_device_eval_batch_size"],
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10,
            bf16=True,
            remove_unused_columns=False,
            model_max_length=args.model_max_length,
        )
        
        return script_args, training_args
    
    elif process == "dpo":
        # Get config parameters like cdpo_cli does
        batch_params = get_batch_size_params("DPO", args.gres)
        
        # Build DPO script arguments with base args and extra args
        dpo_script_args = {
            "pipeline": args.pipeline,
            "model": args.model,
            "dataset": args.dataset,
            "tag": args.tag,
            "beta": beta,
            "loss_type": args.loss_type,
            "reward_model": args.reward_model,
            "noise_type": args.noise_type,
            "noise_level": args.noise_level,
            "per_device_generation_batch_size": batch_params["per_device_generation_batch_size"],
            "per_device_evalreward_batch_size": batch_params["per_device_evalreward_batch_size"],
            "use_flash_attn": args.use_flash_attn,
        }
        
        # Add extra arguments if provided
        if extra_args:
            for key, value in extra_args.items():
                if hasattr(ScriptArgumentsDPO, key):
                    dpo_script_args[key] = value
        
        script_args = ScriptArgumentsDPO(**dpo_script_args)
        
        training_args = TrainingArgumentsDPO(
            num_train_epochs=args.num_dpo_train_epochs,
            learning_rate=args.dpo_learning_rate,
            per_device_train_batch_size=batch_params["per_device_train_batch_size"],
            per_device_eval_batch_size=batch_params["per_device_eval_batch_size"],
            gradient_accumulation_steps=batch_params["gradient_accumulation_steps"],
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10,
            bf16=True,
            remove_unused_columns=False,
            model_max_length=args.model_max_length,
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
            tag=args.tag,
            per_device_evalreward_batch_size=batch_params["per_device_evalreward_batch_size"],
        )
        
        return script_args
    
    else:
        raise ValueError(f"Unknown process: {process}")



def main(args, extra_args=None):
    """Execute the full pipeline with parsed arguments."""

    print(f"🎯 Selected processes: {', '.join(args.processes)}")
    print(f"🖥️  GPU resources: {args.gres}")
    if extra_args:
        print(f"🔧 Extra arguments: {extra_args}")
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
        script_args, training_args = create_arguments("dpo", args, extra_args)
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
        script_args = create_arguments("generate", args, extra_args)
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
        script_args = create_arguments("evalreward", args, extra_args)
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
    
    # Optional arguments - Remove beta since it will be handled as extra arg
    parser.add_argument("--pipeline", help="DPO pipeline name (e.g., DPO, DPO-DRO)")
    parser.add_argument("--reward_model", default="openbmb/Eurus-RM-7b", help="Reward model name")
    parser.add_argument("--noise_type", help="Type of noise (e.g., label_switching, bt_noise_gauss)")
    parser.add_argument("--noise_level", type=float, help="Level of noise (e.g., 0.4, 0.5)")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA r value (default: 64)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha value (default: 16)")
    parser.add_argument("--loss_type", help="Loss type (e.g., generalized_sigmoid)")
    parser.add_argument("--model_max_length", type=int, default=1024, help="Maximum sequence length for the model (default: 1024)")
    parser.add_argument("--num_sft_train_epochs", type=int, default=5, help="Number of SFT training epochs (default: 1)")
    parser.add_argument("--num_dpo_train_epochs", type=int, default=5, help="Number of DPO training epochs (default: 1)")
    parser.add_argument("--sft_learning_rate", type=float, default=2e-5, help="Learning rate for training (default: 2e-5)")
    parser.add_argument("--dpo_learning_rate", type=float, default=1e-5, help="Learning rate for DPO training (default: 1e-5)")
        
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
    
    args, unknown_args = parser.parse_known_args()
    
    # Parse extra arguments
    extra_args = parse_extra_args(unknown_args) if unknown_args else {}
    
    # Validate extra arguments if pipeline is specified and DPO is in processes
    if args.pipeline and "dpo" in args.processes and extra_args:
        try:
            validate_extra_args(args.pipeline, extra_args)
        except ValueError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    main(args, extra_args)
