import argparse

from pipelines.dpo import ScriptArguments as ScriptArgumentsDPO
from pipelines.dpo import TrainingArguments as TrainingArgumentsDPO
from pipelines.dpo import main as dpo_main
from pipelines.evalreward import ScriptArguments as ScriptArgumentsEvalReward
from pipelines.evalreward import main as evalreward_main
from pipelines.generate import ScriptArguments as ScriptArgumentsGenerate
from pipelines.generate import main as generate_main
# Direct imports from pipelines
from pipelines.sft import ScriptArguments as ScriptArgumentsSFT
from pipelines.sft import TrainingArguments as TrainingArgumentsSFT
from pipelines.sft import main as sft_main
# Import utils
from utils import (CONFIGS, check_resource_exists, format_run_name,
                   sanitize_model_name)

HUGGINGFACE_CONFIGS = CONFIGS.services.huggingface
DEVICE_CONFIGS = CONFIGS.devices.devices
OPTIM_CONFIGS = CONFIGS.pipelines.optim


# def get_batch_size_params(pipeline, gres):
#     """Get batch size parameters based on GPU device configuration."""
#     device_type = gres.split(":")[0] if ":" in gres else gres
#     return DEVICE_CONFIGS["pipelines"][pipeline][device_type]


def create_arguments(process, args):
    """Create arguments for any process from execute arguments."""
    
    extra_params = {
        "beta": args.beta,
        "noise_type": args.noise_type,
        "noise_level": args.noise_level,
        "loss_type": args.loss_type,
        "dro_divergence_type": args.dro_divergence_type,
        "resample_model": sanitize_model_name(args.resample_model) if args.resample_model else None,
    }

    if process == "sft":
        # Get config parameters like cdpo_cli does
        # batch_params = get_batch_size_params("SFT", args.gres)
        
        script_args = ScriptArgumentsSFT(
            model=args.model,
            dataset=args.dataset,
            tag=args.tag,
            resample_model=args.resample_model,
            noise_type=args.noise_type,
            noise_level=args.noise_level,
            lora_alpha= args.lora_alpha,
            lora_r=args.lora_r,
            use_flash_attn=args.use_flash_attn,
        )
        
        training_args = TrainingArgumentsSFT(
            num_train_epochs=args.num_sft_train_epochs,
            learning_rate=args.sft_learning_rate,
            # per_device_train_batch_size=batch_params["per_device_train_batch_size"],
            # per_device_eval_batch_size=batch_params["per_device_eval_batch_size"],
            per_device_train_batch_size=args.sft_per_device_train_batch_size,
            per_device_eval_batch_size=args.sft_per_device_eval_batch_size,
            gradient_accumulation_steps=args.sft_gradient_accumulation_steps,
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
        # batch_params = get_batch_size_params("DPO", args.gres)
        
        script_args = ScriptArgumentsDPO(
            pipeline=args.pipeline,
            model=args.model,
            dataset=args.dataset,
            tag=args.tag,
            beta=args.beta,
            loss_type=args.loss_type,
            dro_divergence_type=args.dro_divergence_type,
            resample_model=args.resample_model,
            noise_type=args.noise_type,
            noise_level=args.noise_level,
            use_flash_attn=args.use_flash_attn,
            per_device_generation_batch_size=args.per_device_generation_batch_size,
            per_device_evalreward_batch_size=args.per_device_evalreward_batch_size,
            logit_clipping=args.logit_clipping,
            sft_tag=args.sft_tag,
        )

        training_kwargs = {}
        if args.lr_scheduler_type:
            training_kwargs["lr_scheduler_type"] = args.lr_scheduler_type
        if args.lr_scheduler_kwargs:
            training_kwargs["lr_scheduler_kwargs"] = eval(args.lr_scheduler_kwargs)
        if args.warmup_steps:
            training_kwargs["warmup_steps"] = args.warmup_steps
        if args.optim:
            training_kwargs["optim"] = args.optim
        if args.max_grad_norm:
            training_kwargs["max_grad_norm"] = args.max_grad_norm
        if args.weight_decay:
            training_kwargs["weight_decay"] = args.weight_decay
        if args.logit_clipping:
            training_kwargs["logit_clipping"] = args.logit_clipping
        
        training_args = TrainingArgumentsDPO(
            num_train_epochs=args.num_dpo_train_epochs,
            learning_rate=args.dpo_learning_rate,
            per_device_train_batch_size=args.dpo_per_device_train_batch_size,
            per_device_eval_batch_size=args.dpo_per_device_eval_batch_size,
            gradient_accumulation_steps=args.dpo_gradient_accumulation_steps,
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10,
            bf16=True,
            remove_unused_columns=False,
            model_max_length=args.model_max_length,
            **training_kwargs
        )
        
        return script_args, training_args
    
    elif process == "generate":
        # Get config parameters like cdpo_cli does
        # batch_params = get_batch_size_params("GEN", args.gres)
        
        # Generate run name for DPO model
        if args.noise_type:
            extra_params["noise_type"] = args.noise_type
            extra_params["noise_level"] = args.noise_level
            
        run_name = format_run_name(args.pipeline, args.model, args.dataset, extra_params)
        
        script_args = ScriptArgumentsGenerate(
            run=run_name,
            tag=args.tag,
            per_device_generation_batch_size=args.per_device_generation_batch_size,
            eval_limit=args.eval_limit,
            use_flash_attn=args.use_flash_attn,
            do_sample=args.do_sample,
            temperature=args.temperature,
            max_new_tokens=args.generate_max_new_tokens,
        )
        
        return script_args
    
    elif process == "evalreward":
        # Get config parameters like cdpo_cli does
        # batch_params = get_batch_size_params("EVALREWARD", args.gres)
        
        # Generate run name for DPO model
        if args.noise_type:
            extra_params["noise_type"] = args.noise_type
            extra_params["noise_level"] = args.noise_level
            
        run_name = format_run_name(args.pipeline, args.model, args.dataset, extra_params)
        
        script_args = ScriptArgumentsEvalReward(
            run_name=run_name,
            tag=args.tag,
            # per_device_evalreward_batch_size=batch_params["per_device_evalreward_batch_size"],
            per_device_evalreward_batch_size=args.per_device_evalreward_batch_size,
        )
        
        return script_args
    
    else:
        raise ValueError(f"Unknown process: {process}")



def main(args):
    """Execute the full pipeline with parsed arguments."""

    print(f"🎯 Selected processes: {', '.join(args.processes)}")
    # print(f"🖥️  GPU resources: {args.gres}")
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
    parser.add_argument("--sft_tag", default=None, type=str, help="Tag for the SFT experiment (if different from main tag)")
    
    # Optional arguments
    parser.add_argument("--pipeline", help="DPO pipeline name (e.g., DPO, DPO-DRO)")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta value for DPO (default: 0.1)")
    parser.add_argument("--resample_model", default=None, help="Model to resample from (if applicable)")
    parser.add_argument("--noise_type", help="Type of noise (e.g., label_switching, bt_noise_gauss)")
    parser.add_argument("--noise_level", type=float, help="Level of noise (e.g., 0.4, 0.5)")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA r value (default: 64)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha value (default: 16)")
    parser.add_argument("--loss_type", help="Loss type (e.g., generalized_sigmoid)")
    parser.add_argument("--dro_divergence_type", default="chi_squared", 
                       choices=["kl_div", "chi_squared"],
                       help="Divergence type for DRO method (default: chi_squared)")
    parser.add_argument("--model_max_length", type=int, default=1024, help="Maximum sequence length for the model (default: 1024)")
    parser.add_argument("--num_sft_train_epochs", type=int, default=5, help="Number of SFT training epochs (default: 1)")
    parser.add_argument("--num_dpo_train_epochs", type=int, default=5, help="Number of DPO training epochs (default: 1)")

    # SFT Training arguments
    parser.add_argument("--sft_learning_rate", type=float, default=1e-5, help="Learning rate for training (default: 2e-5)")
    parser.add_argument("--sft_per_device_train_batch_size", type=int, default=16, help="SFT per device train batch size (default: 16)")
    parser.add_argument("--sft_per_device_eval_batch_size", type=int, default=16, help="SFT per device eval batch size (default: 16)")
    parser.add_argument("--sft_gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps (default: 1)")

    # DPO Training arguments
    parser.add_argument("--dpo_learning_rate", type=float, default=2e-6, help="Learning rate for DPO training (default: 2e-6)")
    parser.add_argument("--dpo_per_device_train_batch_size", type=int, default=4, help="DPO per device train batch size (default: 4)")
    parser.add_argument("--dpo_per_device_eval_batch_size", type=int, default=4, help="DPO per device eval batch size (default: 4)")
    parser.add_argument("--dpo_gradient_accumulation_steps", type=int, default=4, help="DPO gradient accumulation steps (default: 4)")

    # Generation arguments
    parser.add_argument("--per_device_generation_batch_size", type=int, default=8, help="Per device generation batch size (default: 8)")
    parser.add_argument("--per_device_evalreward_batch_size", type=int, default=8, help="Per device evalreward batch size (default: 8)")
    parser.add_argument("--do_sample", default=False, action="store_true", help="Use sampling for generation (default: False)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling (default: 1.0)")
    parser.add_argument("--generate_max_new_tokens", type=int, default=None, help="Maximum number of new tokens to generate (default: None, which uses model_max_length)")

    # GPU resource specification (required for config-driven parameters)
    # parser.add_argument("-g", "--gres", required=True, help="GPU resources (e.g., A100:1, A100) for automatic parameter configuration")
    
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
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite existing models/datasets (skip smart checking)")
    
    # Parmereeters for DPO training
    parser.add_argument('--logit_clipping', type=float, default=None, help='Logit clipping value (optional)')  # New argument for logit clippings
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay for optimizer')
    parser.add_argument('--optim', type=str, default=None, help='Optimizer to use')
    parser.add_argument('--max_grad_norm', type=float, default=None, help='Maximum gradient norm for clipping')
    parser.add_argument('--lr_scheduler_type', type=str, default=None, help='Learning rate scheduler type (e.g., linear, cosine, cosine_with_min_lr)')
    parser.add_argument('--lr_scheduler_kwargs', type=str, default=None, help='Additional arguments for the learning rate scheduler')
    parser.add_argument('--warmup_steps', type=int, default=None, help='Number of warmup steps for learning rate scheduler')

    args = parser.parse_args()

    main(args)
